"""Multi-product portfolio backtest engine.

Each product trades independently using its own signal series (main contract
discovery per product). All products share a single capital pool.

Position sizing per product: MARGIN_FRACTION × equity (10%)
Total margin across all products: ≤ 100% of equity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config.strategy import (
    INITIAL_CAPITAL,
    MA_SHORT,
    SLIPPAGE_RATE,
    MARGIN_FRACTION,
    LEVERAGE,
    COMMISSION_PER_CONTRACT,
)
from data.discovery import ContractPeriod
from data.contract_data import download_all_contracts
from engine.multi_v3 import add_indicators, detect_session_type, ohlc_fill_price
from stats.journal import TradeJournal


PROTECTIVE_STOP_BARS = 0  # match GPT no-protective-stop backtest


@dataclass
class PositionState:
    product: str = ""
    direction: int = 0
    qty: float = 0.0
    lots: float = 0.0
    entry_price_slipped: float = 0.0
    entry_time: datetime | None = None
    entry_bar: int = 0
    entry_margin: float = 0.0
    entry_fee: float = 0.0
    current_price: float = 0.0
    symbol: str = ""
    ma_long: int = 75
    session_bucket: str = "day_only"


@dataclass
class PortfolioResult:
    equity_curve: list[float]
    equity_dates: list[datetime]
    journals: dict[str, TradeJournal]
    positions_held: list[int]
    final_equity: float
    total_return: float


def _slippage(price: float, direction: int, action: str) -> float:
    if action == "entry":
        return price * (1 + SLIPPAGE_RATE) if direction == 1 else price * (1 - SLIPPAGE_RATE)
    return price * (1 - SLIPPAGE_RATE) if direction == 1 else price * (1 + SLIPPAGE_RATE)


def calc_fee(lots: float, price: float, side: str = "open") -> float:
    return float(lots * COMMISSION_PER_CONTRACT)


def compute_lots(equity: float, price: float, multiplier: float = 10.0) -> int:
    """Number of contracts: notional / (price × multiplier)."""
    if price <= 0 or equity <= 0:
        return 0
    notional = equity * MARGIN_FRACTION * LEVERAGE
    total_qty = notional / price
    lots = int(total_qty // multiplier)
    return max(lots, 0)


def total_equity(cash: float, positions: dict[str, PositionState]) -> float:
    unrealized = sum(
        (pos.current_price - pos.entry_price_slipped) * pos.qty * pos.direction
        for pos in positions.values()
    )
    return cash + unrealized


def available_margin(cash: float, positions: dict[str, PositionState]) -> float:
    eq = total_equity(cash, positions)
    held = sum(pos.entry_margin for pos in positions.values())
    return eq - held


def compute_contract_multiplier(product: str) -> float:
    """Return contract multiplier (tons/lot). Most Chinese futures = 10."""
    # DCE products: jd=10, p=10, m=10, i=100, c=10, etc.
    overrides = {
        "DCE.i": 100.0,     # iron ore
        "SHFE.rb": 10.0,    # rebar
        "SHFE.hc": 10.0,    # hot coil
        "SHFE.ag": 15.0,    # silver (kg)
        "SHFE.au": 1000.0,  # gold (grams)
        "SHFE.cu": 5.0,     # copper
        "CZCE.MA": 10.0,    # methanol
        "CZCE.TA": 5.0,     # PTA
        "CFFEX.IF": 300.0,  # index futures (yuan per point)
        "CFFEX.IC": 200.0,
    }
    return overrides.get(product, 10.0)


# ---------------------------------------------------------------------------
# Build unified signal series per product
# ---------------------------------------------------------------------------

def build_product_series(
    product: str,
    periods: list[ContractPeriod],
    contract_data: dict[str, pd.DataFrame],
    short_ma: int = MA_SHORT,
) -> pd.DataFrame | None:
    """Build a unified hourly signal series for one product.

    For each hour, finds which contract is the main contract, and uses
    that contract's pre-computed signal data. Returns a DataFrame with
    columns: [datetime, open, high, low, close, volume, zone, prev_zone,
    upper, lower, ma20_slope, symbol, product, session_bucket, ma_long_len]
    """
    if not periods or not contract_data:
        return None

    # Pre-compute signals per contract
    contract_sigs: dict[str, pd.DataFrame] = {}
    for c, df in contract_data.items():
        sig = add_indicators(df, short_ma, 75)  # will be overridden with proper ma_long
        contract_sigs[c] = sig

    # Build unified timeline
    all_hours: list[datetime] = []
    for p in periods:
        if p.contract not in contract_data:
            continue
        cdf = contract_data[p.contract]
        mask = (cdf.index >= p.start) & (cdf.index <= p.end)
        hours = cdf.index[mask]
        all_hours.extend(hours.tolist())
    all_hours = sorted(set(all_hours))

    # Detect session
    first_contract = list(contract_data.keys())[0]
    ma_long = detect_session_type(contract_data[first_contract])

    # Recompute signals with correct ma_long
    contract_sigs = {}
    for c, df in contract_data.items():
        sig = add_indicators(df, short_ma, ma_long)
        contract_sigs[c] = sig

    # Build series
    rows: list[dict] = []
    for hour_dt in all_hours:
        # Find main contract for this bar
        main_contract = ""
        for p in periods:
            if p.start <= hour_dt <= p.end:
                main_contract = p.contract
                break
        if not main_contract or main_contract not in contract_sigs:
            continue

        sig = contract_sigs[main_contract]
        if hour_dt not in sig.index:
            continue

        row = sig.loc[hour_dt]
        raw = contract_data[main_contract].loc[hour_dt]

        rows.append({
            "datetime": hour_dt,
            "open": float(raw["open"]),
            "high": float(raw["high"]),
            "low": float(raw["low"]),
            "close": float(raw["close"]),
            "volume": float(raw.get("volume", 0)),
            "zone": str(row.get("zone", "middle")),
            "prev_zone": str(row.get("prev_zone", "")),
            "upper": float(row["upper"]) if pd.notna(row.get("upper")) else 0.0,
            "lower": float(row["lower"]) if pd.notna(row.get("lower")) else 0.0,
            "ma20_slope": float(row.get("ma20_slope", 0)),
            "bar_seq": len(rows),
            "symbol": main_contract,
            "product": product,
            "ma_long": ma_long,
            "session_bucket": "day_only",
        })

    if not rows:
        return None

    series = pd.DataFrame(rows)
    series["prev_zone"] = series["zone"].shift(1)
    return series


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

AVAILABLE_PRODUCTS = [
    "DCE.jd", "DCE.p", "DCE.m", "DCE.i",
    "SHFE.rb", "SHFE.hc",
    "CZCE.MA", "CZCE.TA", "CZCE.SR",
]


def run_portfolio(
    product_series: dict[str, pd.DataFrame],
    product_order: list[str],
    initial_capital: float = INITIAL_CAPITAL,
) -> PortfolioResult:
    """Run multi-product portfolio backtest.

    Args:
        product_series: {product: DataFrame with signal columns}
        product_order: ordered list of products (determines entry priority)
        initial_capital: starting capital

    Returns:
        PortfolioResult with equity curve and trade journals per product.
    """
    # Concatenate all products, sort by datetime
    combined: list[pd.DataFrame] = []
    for product, frame in product_series.items():
        part = frame.copy()
        part["product"] = product
        combined.append(part)
    all_rows = pd.concat(combined, ignore_index=True)
    all_rows = all_rows.sort_values(["datetime", "product"]).reset_index(drop=True)

    cash = initial_capital
    positions: dict[str, PositionState] = {}
    journals: dict[str, TradeJournal] = {p: TradeJournal() for p in product_order}
    curve_rows: list[dict] = []
    warmup = 75

    print(f"[PORT] {len(product_series)} products, {all_rows['datetime'].nunique()} hours")
    print(f"[PORT] MARGIN_FRACTION={MARGIN_FRACTION}, LEVERAGE={LEVERAGE}")
    print(f"[PORT] Max concurrent positions: {int(1.0 / MARGIN_FRACTION)}")

    for dt, block in all_rows.groupby("datetime", sort=True):
        block = block.sort_values("product")
        bar_seq = int(block["bar_seq"].iloc[0]) if len(block) > 0 else 0

        # 1. Roll exits: symbol change within a product
        for _, row in block.iterrows():
            prod = str(row["product"])
            if prod in positions:
                pos = positions[prod]
                if str(row["symbol"]) != pos.symbol:
                    # Force close on roll
                    fill = _slippage(float(row["close"]), pos.direction, "exit")
                    pnl = (fill - pos.entry_price_slipped) * pos.qty * pos.direction
                    fee = calc_fee(pos.lots, fill, "close")
                    pnl -= fee
                    cash += pnl
                    journals[prod].record(bar_seq, f"close_{'long' if pos.direction==1 else 'short'}", fill, pos.lots, pnl, "roll_force_close")
                    del positions[prod]

        # 2. Mark to market
        for _, row in block.iterrows():
            prod = str(row["product"])
            if prod in positions:
                positions[prod].current_price = float(row["close"])

        # 3. Exit signals (check each product with a position)
        for _, row in block.iterrows():
            prod = str(row["product"])
            if prod not in positions:
                continue
            pos = positions[prod]
            zone = str(row.get("zone", "middle"))
            multiplier = compute_contract_multiplier(prod)

            # Protective stop
            if bar_seq <= pos.entry_bar + PROTECTIVE_STOP_BARS:
                stop_fill = ohlc_fill_price(
                    float(row["low"]) if pos.direction == 1 else float(row["high"]),
                    pos.direction, row, "exit",
                )
                if stop_fill is not None:
                    pnl = (stop_fill - pos.entry_price_slipped) * pos.qty * pos.direction
                    fee = calc_fee(pos.lots, stop_fill, "close")
                    pnl -= fee
                    cash += pnl
                    journals[prod].record(bar_seq, f"close_{'long' if pos.direction==1 else 'short'}", stop_fill, pos.lots, pnl, "protective_stop")
                    del positions[prod]
                    continue

            # Zone exit
            exit_bound = None
            if pos.direction == 1 and zone != "long":
                exit_bound = float(row.get("upper", row["close"]))
            elif pos.direction == -1 and zone != "short":
                exit_bound = float(row.get("lower", row["close"]))

            if exit_bound is not None:
                fill = _slippage(exit_bound, pos.direction, "exit")
                pnl = (fill - pos.entry_price_slipped) * pos.qty * pos.direction
                fee = calc_fee(pos.lots, fill, "close")
                pnl -= fee
                cash += pnl
                journals[prod].record(bar_seq, f"close_{'long' if pos.direction==1 else 'short'}", fill, pos.lots, pnl, "zone_exit")
                del positions[prod]

        # 4. Entry signals
        row_map = {str(r["product"]): r for _, r in block.iterrows()}
        for prod in product_order:
            row = row_map.get(prod)
            if row is None or prod in positions:
                continue

            zone = str(row.get("zone", "middle"))
            prev_zone = str(row.get("prev_zone", ""))
            if pd.isna(prev_zone) or prev_zone == "nan" or prev_zone == "":
                continue
            if zone == prev_zone:
                continue

            direction = 0
            entry_bound = None
            if prev_zone != "long" and zone == "long" and float(row.get("upper", 0)) > 0:
                direction = 1
                entry_bound = float(row["upper"])
            elif prev_zone != "short" and zone == "short" and float(row.get("lower", 0)) > 0:
                direction = -1
                entry_bound = float(row["lower"])

            if direction == 0 or entry_bound is None:
                continue

            # MA20 direction filter
            ma20_slope = float(row.get("ma20_slope", 0))
            if direction == 1 and ma20_slope <= 0:
                continue
            if direction == -1 and ma20_slope >= 0:
                continue

            # Check global margin
            eq_now = total_equity(cash, positions)
            if eq_now <= 0:
                continue
            margin_needed = eq_now * MARGIN_FRACTION
            avail = available_margin(cash, positions)
            if avail + 1e-9 < margin_needed:
                continue

            # Compute lots
            multiplier = compute_contract_multiplier(prod)
            fill_px = _slippage(entry_bound, direction, "entry")
            lots = compute_lots(eq_now, fill_px, multiplier)
            if lots <= 0:
                continue

            qty = lots * multiplier
            fee = calc_fee(lots, fill_px, "open")
            cash -= fee

            pos = PositionState(
                product=prod,
                direction=direction,
                qty=qty,
                lots=float(lots),
                entry_price_slipped=fill_px,
                entry_time=pd.Timestamp(dt).to_pydatetime(),
                entry_bar=bar_seq,
                entry_margin=margin_needed,
                entry_fee=fee,
                current_price=float(row["close"]),
                symbol=str(row["symbol"]),
                ma_long=int(row.get("ma_long", 75)),
            )
            positions[prod] = pos
            journals[prod].record(bar_seq, "open_long" if direction == 1 else "open_short", fill_px, lots, -fee, "signal_entry")

        # Record equity
        eq = total_equity(cash, positions)
        curve_rows.append({"datetime": dt, "equity": eq, "positions": len(positions)})

    # Force close all at end
    for prod in list(positions.keys()):
        pos = positions[prod]
        fill = _slippage(pos.current_price, pos.direction, "exit")
        pnl = (fill - pos.entry_price_slipped) * pos.qty * pos.direction
        fee = calc_fee(pos.lots, fill, "close")
        pnl -= fee
        cash += pnl
        journals[prod].record(len(curve_rows), f"close_{'long' if pos.direction==1 else 'short'}", fill, pos.lots, pnl, "final_force_close")
        del positions[prod]

    if curve_rows:
        curve_rows[-1]["equity"] = cash
        curve_rows[-1]["positions"] = 0

    eq_curve = [r["equity"] for r in curve_rows]
    eq_dates = [r["datetime"] for r in curve_rows]
    positions_held = [r["positions"] for r in curve_rows]
    final_eq = eq_curve[-1] if eq_curve else initial_capital
    total_ret = (final_eq / initial_capital) - 1.0

    # Count trades
    total_trades = sum(len(j.records) for j in journals.values())
    close_trades = sum(
        len([r for r in j.records if "close" in r.get("action", "")])
        for j in journals.values()
    )

    print(f"[PORT] DONE: {close_trades} trades, {final_eq:,.0f} final, {total_ret*100:+.2f}%")
    return PortfolioResult(
        equity_curve=eq_curve,
        equity_dates=eq_dates,
        journals=journals,
        positions_held=positions_held,
        final_equity=final_eq,
        total_return=total_ret,
    )
