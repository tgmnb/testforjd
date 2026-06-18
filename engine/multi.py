"""Multi-contract backtest engine — v2, TGM's confirmed spec.

Key changes from v1:
1. Main contract from volume-based detection (not OI ratio)
2. Force-close when main contract rolls while holding
3. Slippage 0.2%
4. Commission 7 yuan/contract/side
5. Dynamic sizing: equity × 10% / (price × 10 × 10%)
6. Floating P&L with slippage-adjusted entry
7. Protective 2% stop-loss
8. Force-close all at end
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from data.discovery import ContractPeriod
from config.strategy import (
    INITIAL_CAPITAL,
    MA_SHORT,
    SLIPPAGE,
    COMMISSION_PER_CONTRACT,
    MARGIN_RATIO,
    CONTRACT_MULTIPLIER,
    STOP_LOSS_PCT,
    RESOLUTION,
)
from stats.journal import TradeJournal
from strategy.signal import compute_signal


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PositionState:
    has_position: bool = False
    direction: str = "none"
    entry_price: float = 0.0  # raw entry price (before slippage)
    entry_price_slipped: float = 0.0  # after slippage
    entry_bar: int = 0
    position_size: float = 0.0
    entry_dt: datetime | None = None
    contract: str = ""


@dataclass
class BacktestResult:
    symbol: str
    equity_curve: list[float]
    equity_dates: list[datetime]
    trade_journal: Any
    final_equity: float
    total_return: float
    num_trades: int
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    contract_history: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------


def _apply_slippage(price: float, action: str) -> float:
    if action in ("open_long", "close_short"):
        return price * (1.0 + SLIPPAGE)
    return price * (1.0 - SLIPPAGE)


def _compute_margin_per_contract(price: float) -> float:
    """Margin required per contract = price × multiplier × margin_ratio."""
    return price * CONTRACT_MULTIPLIER * MARGIN_RATIO


def _compute_position_size(equity: float, price: float) -> int:
    """Number of contracts = margin_budget / margin_per_contract.
    
    margin_budget = equity × MARGIN_RATIO
    margin_per_contract = price × multiplier × MARGIN_RATIO
    → contracts = equity / (price × multiplier)
    """
    if price <= 0 or equity <= 0:
        return 0
    margin_per = _compute_margin_per_contract(price)
    margin_budget = equity * MARGIN_RATIO
    if margin_budget < margin_per:
        return 0
    return int(margin_budget // margin_per)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _compute_contract_signals(
    df: pd.DataFrame,
    short_period: int,
    long_period: int,
) -> pd.DataFrame:
    return compute_signal(df, short_period=short_period, long_period=long_period)


# ---------------------------------------------------------------------------
# Bar-level processing
# ---------------------------------------------------------------------------


def _process_signal(
    state: PositionState,
    bar_idx: int,
    bar_dt: datetime,
    close: float,
    upper_bound: float,
    lower_bound: float,
    entry_price: float,
    signal: str,
    zone: str,
    equity: float,
) -> tuple[PositionState, list[dict], float]:
    """Process one bar's signal. Returns (new_state, trades, updated_equity).
    
    Updates equity as trades happen (costs deducted immediately).
    """
    new_state = copy.deepcopy(state)
    trades: list[dict] = []

    if new_state.has_position:
        # --- Check stop-loss first ---
        sl_price = None
        sl_action = None
        if new_state.direction == "long":
            sl_price = new_state.entry_price * (1.0 - STOP_LOSS_PCT)
            if close <= sl_price:
                # Stop-loss triggered
                exit_px = _apply_slippage(sl_price, "close_long")
                pnl = (
                    exit_px - new_state.entry_price_slipped
                ) * new_state.position_size
                commission = COMMISSION_PER_CONTRACT * new_state.position_size
                pnl -= commission
                trades.append({
                    "bar_index": bar_idx,
                    "datetime": bar_dt,
                    "action": "close_long",
                    "price": exit_px,
                    "size": new_state.position_size,
                    "pnl": pnl,
                    "reason": "stop_loss",
                    "contract": new_state.contract,
                })
                equity += pnl
                new_state.has_position = False
                new_state.direction = "none"
                new_state.position_size = 0.0
                return new_state, trades, equity

        elif new_state.direction == "short":
            sl_price = new_state.entry_price * (1.0 + STOP_LOSS_PCT)
            if close >= sl_price:
                exit_px = _apply_slippage(sl_price, "close_short")
                pnl = (
                    new_state.entry_price_slipped - exit_px
                ) * new_state.position_size
                commission = COMMISSION_PER_CONTRACT * new_state.position_size
                pnl -= commission
                trades.append({
                    "bar_index": bar_idx,
                    "datetime": bar_dt,
                    "action": "close_short",
                    "price": exit_px,
                    "size": new_state.position_size,
                    "pnl": pnl,
                    "reason": "stop_loss",
                    "contract": new_state.contract,
                })
                equity += pnl
                new_state.has_position = False
                new_state.direction = "none"
                new_state.position_size = 0.0
                return new_state, trades, equity

        # --- Check zone exit ---
        if new_state.direction == "long" and zone != "bullish":
            exit_px = _apply_slippage(upper_bound, "close_long")
            pnl = (
                exit_px - new_state.entry_price_slipped
            ) * new_state.position_size
            commission = COMMISSION_PER_CONTRACT * new_state.position_size
            pnl -= commission
            trades.append({
                "bar_index": bar_idx,
                "datetime": bar_dt,
                "action": "close_long",
                "price": exit_px,
                "size": new_state.position_size,
                "pnl": pnl,
                "reason": "zone_exit",
                "contract": new_state.contract,
            })
            equity += pnl
            new_state.has_position = False
            new_state.direction = "none"
            new_state.position_size = 0.0

        elif new_state.direction == "short" and zone != "bearish":
            exit_px = _apply_slippage(lower_bound, "close_short")
            pnl = (
                new_state.entry_price_slipped - exit_px
            ) * new_state.position_size
            commission = COMMISSION_PER_CONTRACT * new_state.position_size
            pnl -= commission
            trades.append({
                "bar_index": bar_idx,
                "datetime": bar_dt,
                "action": "close_short",
                "price": exit_px,
                "size": new_state.position_size,
                "pnl": pnl,
                "reason": "zone_exit",
                "contract": new_state.contract,
            })
            equity += pnl
            new_state.has_position = False
            new_state.direction = "none"
            new_state.position_size = 0.0

    else:
        # No position — check entry
        if signal == "open_long":
            size = _compute_position_size(equity, entry_price)
            if size and size > 0:
                filled_px = _apply_slippage(entry_price, signal)
                commission = COMMISSION_PER_CONTRACT * size
                equity -= commission
                new_state.has_position = True
                new_state.direction = "long"
                new_state.entry_price = entry_price
                new_state.entry_price_slipped = filled_px
                new_state.entry_bar = bar_idx
                new_state.entry_dt = bar_dt
                new_state.position_size = size
                trades.append({
                    "bar_index": bar_idx,
                    "datetime": bar_dt,
                    "action": "open_long",
                    "price": filled_px,
                    "size": size,
                    "pnl": -commission,
                    "reason": "signal_entry",
                    "contract": "",
                })

        elif signal == "open_short":
            size = _compute_position_size(equity, entry_price)
            if size and size > 0:
                filled_px = _apply_slippage(entry_price, signal)
                commission = COMMISSION_PER_CONTRACT * size
                equity -= commission
                new_state.has_position = True
                new_state.direction = "short"
                new_state.entry_price = entry_price
                new_state.entry_price_slipped = filled_px
                new_state.entry_bar = bar_idx
                new_state.entry_dt = bar_dt
                new_state.position_size = size
                trades.append({
                    "bar_index": bar_idx,
                    "datetime": bar_dt,
                    "action": "open_short",
                    "price": filled_px,
                    "size": size,
                    "pnl": -commission,
                    "reason": "signal_entry",
                    "contract": "",
                })

    return new_state, trades, equity


def _force_close(
    state: PositionState,
    bar_idx: int,
    bar_dt: datetime,
    close_price: float,
    reason: str,
) -> tuple[PositionState, list[dict], float]:
    """Force-close an open position at market (close price with slippage)."""
    if not state.has_position:
        return state, [], 0.0

    trades: list[dict] = []
    pnl = 0.0

    if state.direction == "long":
        exit_px = _apply_slippage(close_price, "close_long")
        pnl = (
            exit_px - state.entry_price_slipped
        ) * state.position_size
        commission = COMMISSION_PER_CONTRACT * state.position_size
        pnl -= commission
        trades.append({
            "bar_index": bar_idx,
            "datetime": bar_dt,
            "action": "close_long",
            "price": exit_px,
            "size": state.position_size,
            "pnl": pnl,
            "reason": reason,
            "contract": state.contract,
        })

    elif state.direction == "short":
        exit_px = _apply_slippage(close_price, "close_short")
        pnl = (
            state.entry_price_slipped - exit_px
        ) * state.position_size
        commission = COMMISSION_PER_CONTRACT * state.position_size
        pnl -= commission
        trades.append({
            "bar_index": bar_idx,
            "datetime": bar_dt,
            "action": "close_short",
            "price": exit_px,
            "size": state.position_size,
            "pnl": pnl,
            "reason": reason,
            "contract": state.contract,
        })

    new_state = PositionState()
    return new_state, trades, pnl


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def run_backtest(
    periods: list[ContractPeriod],
    contract_data: dict[str, pd.DataFrame],
    initial_capital: float = INITIAL_CAPITAL,
    short_ma: int = MA_SHORT,
    long_ma: int = 75,
) -> BacktestResult:
    """Run the multi-contract backtest (v2 — TGM spec).

    Args:
        periods: Contract periods from volume-based discovery.
        contract_data: Dict mapping contract → hourly OHLC DataFrame.
        initial_capital: Starting capital.
        short_ma: Short MA period.
        long_ma: Long MA period.

    Returns:
        BacktestResult with equity curve and trade journal.
    """
    print(f"[BTv2] Starting backtest ({len(periods)} periods, "
          f"{len(contract_data)} contracts)")

    # Pre-compute signals for each contract
    contract_signals: dict[str, pd.DataFrame] = {}
    for c, df in contract_data.items():
        sig = _compute_contract_signals(df, short_ma, long_ma)
        contract_signals[c] = sig

    # Build unified time index from all contracts' data within periods
    all_hours: list[datetime] = []
    for p in periods:
        if p.contract not in contract_data:
            continue
        cdf = contract_data[p.contract]
        mask = (cdf.index >= p.start) & (cdf.index <= p.end)
        hours = cdf.index[mask]
        all_hours.extend(hours.tolist())

    all_hours = sorted(set(all_hours))
    print(f"[BTv2] Timeline: {len(all_hours)} hours, "
          f"{all_hours[0]} to {all_hours[-1]}")

    # State
    state = PositionState()
    journal = TradeJournal()
    equity_curve: list[float] = []
    equity_dates: list[datetime] = []
    balance = initial_capital
    warmup = max(short_ma, long_ma)
    contract_history: list[str] = []

    for bar_idx, hour_dt in enumerate(all_hours):
        # Determine current main contract
        main_contract = ""
        for p in periods:
            if p.start <= hour_dt <= p.end:
                main_contract = p.contract
                break

        if not main_contract or main_contract not in contract_data:
            equity_curve.append(balance)
            equity_dates.append(hour_dt)
            contract_history.append(main_contract if main_contract else "unknown")
            continue

        sig_df = contract_signals[main_contract]
        if hour_dt not in sig_df.index:
            equity_curve.append(balance)
            equity_dates.append(hour_dt)
            contract_history.append(main_contract)
            continue

        sig_row = sig_df.loc[hour_dt]
        close_price = float(
            contract_data[main_contract].loc[hour_dt, "close"]
        )

        # --- Force close on roll: if holding and main contract changed ---
        if state.has_position and state.contract != main_contract:
            new_state, trades, pnl = _force_close(
                state, bar_idx, hour_dt, close_price,
                "roll_force_close",
            )
            balance += pnl
            for t in trades:
                journal.record(
                    t["bar_index"], t["action"], t["price"],
                    t["size"], t["pnl"], t["reason"],
                )
            state = new_state
            # Fall through to check entry signals on the new contract

        # Skip warmup
        if bar_idx < warmup:
            equity_curve.append(balance)
            equity_dates.append(hour_dt)
            contract_history.append(main_contract)
            continue

        if state.has_position:
            # Use HELD contract's data for exit/stop signals
            hold_contract = state.contract
            if hold_contract in contract_signals:
                hold_sig = contract_signals[hold_contract]
                if hour_dt in hold_sig.index:
                    hold_row = hold_sig.loc[hour_dt]
                    hold_close = float(
                        contract_data[hold_contract].loc[hour_dt, "close"]
                    )

                    new_state, trades, balance = _process_signal(
                        state, bar_idx, hour_dt,
                        close=hold_close,
                        upper_bound=hold_row.get("upper_bound", 0),
                        lower_bound=hold_row.get("lower_bound", 0),
                        entry_price=hold_row.get("entry_price", 0),
                        signal=hold_row.get("signal", "hold"),
                        zone=hold_row.get("zone", ""),
                        equity=balance,
                    )

                    for t in trades:
                        journal.record(
                            t["bar_index"], t["action"], t["price"],
                            t["size"], t["pnl"], t["reason"],
                        )

                    state = new_state

        else:
            # Use main contract's signal for entry
            new_state, trades, balance = _process_signal(
                state, bar_idx, hour_dt,
                close=close_price,
                upper_bound=sig_row.get("upper_bound", 0),
                lower_bound=sig_row.get("lower_bound", 0),
                entry_price=sig_row.get("entry_price", 0),
                signal=sig_row.get("signal", "hold"),
                zone=sig_row.get("zone", ""),
                equity=balance,
            )

            for t in trades:
                t["contract"] = main_contract
                journal.record(
                    t["bar_index"], t["action"], t["price"],
                    t["size"], t["pnl"], t["reason"],
                )

            if new_state.has_position:
                new_state.contract = main_contract

            state = new_state

        # Mark-to-market equity
        if state.has_position:
            current_price = 0.0
            if (
                state.contract in contract_data
                and hour_dt in contract_data[state.contract].index
            ):
                current_price = float(
                    contract_data[state.contract].loc[hour_dt, "close"]
                )

            if state.direction == "long" and state.entry_price_slipped > 0:
                # Floating P&L uses the SLIPPAGE-ADJUSTED entry price
                unrealized = (
                    current_price - state.entry_price_slipped
                ) * state.position_size
            elif (
                state.direction == "short"
                and state.entry_price_slipped > 0
            ):
                unrealized = (
                    state.entry_price_slipped - current_price
                ) * state.position_size
            else:
                unrealized = 0.0
            current_equity = balance + unrealized
        else:
            current_equity = balance

        equity_curve.append(current_equity)
        equity_dates.append(hour_dt)
        contract_history.append(main_contract)

    # --- Force close any remaining position at end ---
    if state.has_position:
        last_dt = all_hours[-1] if all_hours else None
        last_close = 0.0
        if (
            state.contract in contract_data
            and last_dt in contract_data[state.contract].index
        ):
            last_close = float(
                contract_data[state.contract].loc[last_dt, "close"]
            )

        if last_close > 0 and last_dt:
            new_state, trades, pnl = _force_close(
                state, len(all_hours), last_dt, last_close,
                "final_force_close",
            )
            balance += pnl
            for t in trades:
                journal.record(
                    t["bar_index"], t["action"], t["price"],
                    t["size"], t["pnl"], t["reason"],
                )
            state = new_state
            # Update final equity
            equity_curve[-1] = balance

    # Results
    final_equity = equity_curve[-1] if equity_curve else initial_capital
    total_return = (final_equity / initial_capital) - 1.0
    close_records = [
        r for r in journal.records if "close" in r.get("action", "")
    ]
    num_trades = len(close_records)

    print(f"[BTv2] DONE: {num_trades} trades, "
          f"{final_equity:.0f} final equity, "
          f"{total_return * 100:+.2f}% return")

    return BacktestResult(
        symbol="multi-contract",
        equity_curve=equity_curve,
        equity_dates=equity_dates,
        trade_journal=journal,
        final_equity=final_equity,
        total_return=total_return,
        num_trades=num_trades,
        contract_history=contract_history,
    )
