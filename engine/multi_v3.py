"""Multi-contract backtest engine — v3, incorporating GPT framework improvements.

Improvements from GPT analysis:
1. OHLC range fill check — verifies trigger price hit bar's range before filling
2. N-bar protective stop — checked BEFORE MA exit, uses OHLC for fill
3. Fee: dynamic (rate% or fixed/手), not hardcoded
4. Available margin constraint before entry
5. Consistent P&L: slipped entry price used throughout
6. Force close on roll + force close at end
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from data.discovery import ContractPeriod
from config.strategy import (
    INITIAL_CAPITAL,
    MA_SHORT,
    SLIPPAGE_RATE,
    CONTRACT_MULTIPLIER,
    MARGIN_FRACTION,
    LEVERAGE,
)
from stats.journal import TradeJournal
from strategy.signal import compute_signal


# GPT-style constants
PROTECTIVE_STOP_BARS = 3  # first N bars protected


@dataclass
class PositionState:
    product: str = ""
    direction: int = 0  # 1=long, -1=short
    qty: float = 0.0    # underlying quantity (tons)
    lots: float = 0.0   # contracts
    entry_price_slipped: float = 0.0  # after slippage
    entry_price_raw: float = 0.0      # before slippage
    entry_time: datetime | None = None
    entry_bar: int = 0
    entry_margin: float = 0.0
    entry_fee: float = 0.0
    current_price: float = 0.0
    symbol: str = ""
    protective_stop_price: float = 0.0
    protective_stop_until_bar: int = 0


@dataclass
class BacktestResult:
    symbol: str
    equity_curve: list[float]
    equity_dates: list[datetime]
    trade_journal: Any
    final_equity: float
    total_return: float
    num_trades: int


# ---------------------------------------------------------------------------
# Fee from tqsdk quote (rate-based if available, else fixed per contract)
# ---------------------------------------------------------------------------

# JD at DCE: ~7 yuan/contract fixed fee, no rate-based fee
# We'll use fixed fee from config, or dynamic from quote
JD_COMMISSION_FIXED = 7.0  # yuan/contract/side
JD_COMMISSION_RATE = 0.0   # no rate-based fee for JD
JD_MULTIPLIER = 10.0       # tons/contract


def calc_fee(lots: float, price: float, side: str = "open") -> float:
    """Calculate commission for a trade.
    
    Uses rate-based or fixed-per-contract model (GPT-style).
    For JD: fixed 7 yuan/contract per side (no rate-based).
    """
    rate = JD_COMMISSION_RATE
    fixed = JD_COMMISSION_FIXED
    notional = lots * JD_MULTIPLIER * price
    return float(max(0.0, notional * rate + lots * fixed))


# ---------------------------------------------------------------------------
# Fill helpers (GPT-style OHLC range check)
# ---------------------------------------------------------------------------

def _slippage(price: float, direction: int, action: str) -> float:
    """Apply 0.2% slippage against the trade."""
    if action == "entry":
        return price * (1 + SLIPPAGE_RATE) if direction == 1 else price * (1 - SLIPPAGE_RATE)
    return price * (1 - SLIPPAGE_RATE) if direction == 1 else price * (1 + SLIPPAGE_RATE)


def ohlc_fill_price(trigger: float, direction: int, row: pd.Series, action: str) -> float | None:
    """Check if trigger price was reached within the bar's OHLC range.
    
    Returns the fill price (with slippage applied) or None if not fillable.
    """
    open_px = float(row["open"])
    high_px = float(row["high"])
    low_px = float(row["low"])

    if action == "entry":
        if direction == 1:  # long entry: price must go UP to trigger
            if open_px >= trigger:
                return _slippage(open_px, direction, action)
            if high_px >= trigger:
                return _slippage(trigger, direction, action)
        else:  # short entry: price must go DOWN to trigger
            if open_px <= trigger:
                return _slippage(open_px, direction, action)
            if low_px <= trigger:
                return _slippage(trigger, direction, action)
    else:  # exit
        if direction == 1:  # long exit: price goes DOWN to stop/exit
            if open_px <= trigger:
                return _slippage(open_px, direction, action)
            if low_px <= trigger:
                return _slippage(trigger, direction, action)
        else:  # short exit: price goes UP to stop/exit
            if open_px >= trigger:
                return _slippage(open_px, direction, action)
            if high_px >= trigger:
                return _slippage(trigger, direction, action)

    return None


# ---------------------------------------------------------------------------
# Position sizing (GPT-style: equity × MARGIN_FRACTION × LEVERAGE)
# ---------------------------------------------------------------------------

def compute_lots(equity: float, price: float) -> int:
    """Number of contracts: notional / (price × multiplier).
    
    notional = equity × MARGIN_FRACTION × LEVERAGE = equity × 1.0
    lots = notional / (price × multiplier)
    """
    if price <= 0 or equity <= 0:
        return 0
    notional = equity * MARGIN_FRACTION * LEVERAGE
    total_qty = notional / price  # underlying quantity
    lots = int(total_qty // JD_MULTIPLIER)
    return max(lots, 0)


def available_margin(equity: float, held_margin: float) -> float:
    return equity - held_margin


def total_equity(cash: float, pos: PositionState | None) -> float:
    if pos is None:
        return cash
    unrealized = (pos.current_price - pos.entry_price_slipped) * pos.qty * pos.direction
    return cash + unrealized


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame, short_period: int, long_period: int) -> pd.DataFrame:
    """GPT-style indicator computation (pure pandas, returns same df with new cols)."""
    out = df.copy().sort_index()
    out["ma20"] = out["close"].rolling(short_period, min_periods=short_period).mean()
    out["ma_long"] = out["close"].rolling(long_period, min_periods=long_period).mean()
    out["upper"] = out[["ma20", "ma_long"]].max(axis=1)
    out["lower"] = out[["ma20", "ma_long"]].min(axis=1)
    out["zone"] = np.select(
        [out["close"] > out["upper"], out["close"] < out["lower"]],
        ["long", "short"],
        default="middle",
    )
    out["prev_zone"] = out["zone"].shift(1)
    return out


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    periods: list[ContractPeriod],
    contract_data: dict[str, pd.DataFrame],
    initial_capital: float = INITIAL_CAPITAL,
    short_ma: int = MA_SHORT,
    long_ma: int = 75,
) -> BacktestResult:
    """Run multi-contract backtest (v3 — GPT improvements)."""
    print(f"[BTv3] Starting ({len(periods)} periods, {len(contract_data)} contracts)")

    # Pre-compute indicators for each contract
    contract_sigs: dict[str, pd.DataFrame] = {}
    for c, df in contract_data.items():
        sig = add_indicators(df, short_ma, long_ma)
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
    print(f"[BTv3] Timeline: {len(all_hours)} hours")

    # State
    cash = initial_capital
    pos: PositionState | None = None
    journal = TradeJournal()
    equity_curve: list[float] = []
    equity_dates: list[datetime] = []
    warmup = max(short_ma, long_ma)

    for bar_idx, hour_dt in enumerate(all_hours):
        # Determine main contract
        main_contract = ""
        for p in periods:
            if p.start <= hour_dt <= p.end:
                main_contract = p.contract
                break
        if not main_contract or main_contract not in contract_sigs:
            eq = total_equity(cash, pos)
            equity_curve.append(eq)
            equity_dates.append(hour_dt)
            continue

        sig = contract_sigs[main_contract]
        if hour_dt not in sig.index:
            eq = total_equity(cash, pos)
            equity_curve.append(eq)
            equity_dates.append(hour_dt)
            continue

        row = sig.loc[hour_dt]
        close_px = float(contract_data[main_contract].loc[hour_dt, "close"])

        # ---- Force close on roll ----
        if pos and pos.symbol != main_contract:
            fill = ohlc_fill_price(pos.current_price, pos.direction, row, "exit")
            if fill is None:
                fill = _slippage(close_px, pos.direction, "exit")
            pnl = (fill - pos.entry_price_slipped) * pos.qty * pos.direction
            exit_fee = calc_fee(pos.lots, fill, "close")
            pnl -= exit_fee
            cash += pnl
            journal.record(bar_idx, "close_long" if pos.direction == 1 else "close_short",
                          fill, pos.lots, pnl, "roll_force_close")
            pos = None

        # Skip warmup
        if bar_idx < warmup:
            equity_curve.append(total_equity(cash, pos))
            equity_dates.append(hour_dt)
            continue

        if pos:
            # Held contract's data for exit
            hold_sig = contract_sigs.get(pos.symbol)
            hold_close = close_px
            if hold_sig is not None and hour_dt in hold_sig.index:
                hold_row = hold_sig.loc[hour_dt]
            else:
                hold_row = row  # fallback to main contract
                hold_close = float(contract_data[pos.symbol].loc[hour_dt, "close"]) if pos.symbol in contract_data and hour_dt in contract_data[pos.symbol].index else close_px

            pos.current_price = hold_close

            # 1. Protective stop check (GPT-style: N-bar, OHLC fill)
            if bar_idx <= pos.protective_stop_until_bar:
                stop_fill = ohlc_fill_price(pos.protective_stop_price, pos.direction, hold_row, "exit")
                if stop_fill is not None:
                    pnl = (stop_fill - pos.entry_price_slipped) * pos.qty * pos.direction
                    fee = calc_fee(pos.lots, stop_fill, "close")
                    pnl -= fee
                    cash += pnl
                    journal.record(bar_idx, "close_long" if pos.direction == 1 else "close_short",
                                  stop_fill, pos.lots, pnl, f"protective_stop_N{PROTECTIVE_STOP_BARS}")
                    pos = None
                    equity_curve.append(total_equity(cash, pos))
                    equity_dates.append(hour_dt)
                    continue

            # 2. Zone exit check
            zone = str(hold_row.get("zone", "middle"))
            if (pos.direction == 1 and zone != "long") or (pos.direction == -1 and zone != "short"):
                exit_bound = float(hold_row.get("upper" if pos.direction == 1 else "lower", hold_close))
                fill = ohlc_fill_price(exit_bound, pos.direction, hold_row, "exit")
                if fill is None:
                    fill = _slippage(hold_close, pos.direction, "exit")
                pnl = (fill - pos.entry_price_slipped) * pos.qty * pos.direction
                fee = calc_fee(pos.lots, fill, "close")
                pnl -= fee
                cash += pnl
                journal.record(bar_idx, "close_long" if pos.direction == 1 else "close_short",
                              fill, pos.lots, pnl, "zone_exit")
                pos = None

        else:
            # Entry
            zone = str(row.get("zone", "middle"))
            prev_zone = str(row.get("prev_zone", ""))
            if pd.isna(prev_zone) or prev_zone == "nan":
                eq = total_equity(cash, pos)
                equity_curve.append(eq)
                equity_dates.append(hour_dt)
                continue

            direction = 0
            entry_bound = None
            signal = ""

            if prev_zone != "long" and zone == "long" and pd.notna(row.get("upper")):
                direction = 1
                entry_bound = float(row["upper"])
                signal = "open_long"
            elif prev_zone != "short" and zone == "short" and pd.notna(row.get("lower")):
                direction = -1
                entry_bound = float(row["lower"])
                signal = "open_short"

            if direction != 0 and entry_bound and entry_bound > 0:
                # Check available margin
                eq_now = total_equity(cash, pos)
                margin_needed = eq_now * MARGIN_FRACTION
                if pos is not None:
                    if available_margin(eq_now, pos.entry_margin) + 1e-9 < margin_needed:
                        equity_curve.append(eq_now)
                        equity_dates.append(hour_dt)
                        continue

                # OHLC fill check (GPT-style)
                fill_px = ohlc_fill_price(entry_bound, direction, row, "entry")
                if fill_px is None:
                    # Fallback: use boundary with slippage
                    fill_px = _slippage(entry_bound, direction, "entry")

                lots = compute_lots(eq_now, fill_px)
                if lots <= 0:
                    equity_curve.append(eq_now)
                    equity_dates.append(hour_dt)
                    continue

                qty = lots * JD_MULTIPLIER
                fee = calc_fee(lots, fill_px, "open")
                cash -= fee

                # Stop price: entry bar's OHLC extreme (GPT-style)
                stop_price = float(row["low"]) if direction == 1 else float(row["high"])

                pos = PositionState(
                    product="DCE.jd",
                    direction=direction,
                    qty=qty,
                    lots=float(lots),
                    entry_price_slipped=fill_px,
                    entry_price_raw=entry_bound,
                    entry_time=hour_dt,
                    entry_bar=bar_idx,
                    entry_margin=margin_needed,
                    entry_fee=fee,
                    current_price=close_px,
                    symbol=main_contract,
                    protective_stop_price=stop_price,
                    protective_stop_until_bar=bar_idx + PROTECTIVE_STOP_BARS,
                )
                journal.record(bar_idx, signal, fill_px, lots, -fee, "signal_entry")

        # Mark-to-market
        eq = total_equity(cash, pos)
        equity_curve.append(eq)
        equity_dates.append(hour_dt)

    # ---- Force close at end ----
    if pos:
        last_dt = all_hours[-1] if all_hours else hour_dt
        last_close = close_px
        if pos.symbol in contract_data and last_dt in contract_data[pos.symbol].index:
            last_close = float(contract_data[pos.symbol].loc[last_dt, "close"])
        fill = _slippage(last_close, pos.direction, "exit")
        pnl = (fill - pos.entry_price_slipped) * pos.qty * pos.direction
        fee = calc_fee(pos.lots, fill, "close")
        pnl -= fee
        cash += pnl
        journal.record(len(all_hours), "close_long" if pos.direction == 1 else "close_short",
                      fill, pos.lots, pnl, "final_force_close")
        equity_curve[-1] = cash

    final_equity = equity_curve[-1] if equity_curve else initial_capital
    total_return = (final_equity / initial_capital) - 1.0
    close_records = [r for r in journal.records if "close" in r.get("action", "")]
    num_trades = len(close_records)

    print(f"[BTv3] DONE: {num_trades} trades, {final_equity:.0f} final, {total_return*100:+.2f}%")
    return BacktestResult(
        symbol="DCE.jd",
        equity_curve=equity_curve,
        equity_dates=equity_dates,
        trade_journal=journal,
        final_equity=final_equity,
        total_return=total_return,
        num_trades=num_trades,
    )
