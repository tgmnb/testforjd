"""Multi-contract backtest engine — uses individual contract data per period.

Key design:
1. At each bar, the "main contract" period determines which contract's
   OHLC data to use for MA calculations.
2. Entry: signal is computed on the current main contract's individual data.
3. When holding a contract, EXIT signals use that held contract's own data.
4. After exit, the next entry re-evaluates the current main contract.
5. No cross-contract data splicing — each contract's MAs are its own.
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
    SLIPPAGE,
    MARGIN_RATIO,
)
from risk.sizing import compute_position_size
from stats.journal import TradeJournal
from strategy.signal import compute_signal


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PositionState:
    has_position: bool = False
    direction: str = "none"
    entry_price: float = 0.0
    entry_bar: int = 0
    position_size: float = 0.0
    contract: str = ""


@dataclass
class BacktestResult:
    symbol: str
    equity_curve: list[float]
    trade_journal: Any
    final_equity: float
    total_return: float
    num_trades: int
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    contract_history: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _compute_contract_signals(
    df: pd.DataFrame,
    short_period: int,
    long_period: int,
) -> pd.DataFrame:
    return compute_signal(df, short_period=short_period, long_period=long_period)


def _apply_slippage(price: float, action: str) -> float:
    if action in ("open_long", "close_short"):
        return price * (1.0 + SLIPPAGE)
    return price * (1.0 - SLIPPAGE)


# ---------------------------------------------------------------------------
# Bar-level processing
# ---------------------------------------------------------------------------


def _process_signal(
    state: PositionState,
    bar_index: int,
    close: float,
    upper_bound: float,
    lower_bound: float,
    entry_price: float,
    signal: str,
    zone: str,
    sizing_fn,
    margin_per_contract: float = 3000.0,
) -> tuple[PositionState, list[dict]]:
    """Process one bar's signal. Returns (new_state, trades_taken)."""
    new_state = copy.deepcopy(state)
    trades: list[dict] = []

    if new_state.has_position:
        # Check exit
        if new_state.direction == "long" and zone != "bullish":
            exit_px = _apply_slippage(upper_bound, "close_long")
            entry_px = _apply_slippage(
                new_state.entry_price, "open_long"
            )
            pnl = (exit_px - entry_px) * new_state.position_size
            trades.append(
                {
                    "bar_index": bar_index,
                    "action": "close_long",
                    "price": exit_px,
                    "size": new_state.position_size,
                    "pnl": pnl,
                    "reason": "zone_exit",
                    "contract": new_state.contract,
                }
            )
            new_state.has_position = False
            new_state.direction = "none"
            new_state.position_size = 0.0

        elif new_state.direction == "short" and zone != "bearish":
            exit_px = _apply_slippage(lower_bound, "close_short")
            entry_px = _apply_slippage(
                new_state.entry_price, "open_short"
            )
            pnl = (entry_px - exit_px) * new_state.position_size
            trades.append(
                {
                    "bar_index": bar_index,
                    "action": "close_short",
                    "price": exit_px,
                    "size": new_state.position_size,
                    "pnl": pnl,
                    "reason": "zone_exit",
                    "contract": new_state.contract,
                }
            )
            new_state.has_position = False
            new_state.direction = "none"
            new_state.position_size = 0.0

    else:
        # No position — check entry
        if signal == "open_long":
            size = sizing_fn(margin_per_contract)
            if size and size > 0:
                filled_px = _apply_slippage(entry_price, signal)
                new_state.has_position = True
                new_state.direction = "long"
                new_state.entry_price = entry_price
                new_state.entry_bar = bar_index
                new_state.position_size = size
                trades.append(
                    {
                        "bar_index": bar_index,
                        "action": "open_long",
                        "price": filled_px,
                        "size": size,
                        "pnl": 0.0,
                        "reason": "signal_entry",
                        "contract": "",
                    }
                )

        elif signal == "open_short":
            size = sizing_fn(margin_per_contract)
            if size and size > 0:
                filled_px = _apply_slippage(entry_price, signal)
                new_state.has_position = True
                new_state.direction = "short"
                new_state.entry_price = entry_price
                new_state.entry_bar = bar_index
                new_state.position_size = size
                trades.append(
                    {
                        "bar_index": bar_index,
                        "action": "open_short",
                        "price": filled_px,
                        "size": size,
                        "pnl": 0.0,
                        "reason": "signal_entry",
                        "contract": "",
                    }
                )

    return new_state, trades


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def run_backtest(
    periods: list[ContractPeriod],
    contract_data: dict[str, pd.DataFrame],
    initial_capital: float = INITIAL_CAPITAL,
    short_ma: int = MA_SHORT,
    long_ma: int = 75,
    margin_ratio: float = MARGIN_RATIO,
) -> BacktestResult:
    """Run the multi-contract backtest.

    Args:
        periods: Contract periods from discovery.
        contract_data: Dict mapping contract -> hourly OHLC DataFrame.
        initial_capital: Starting capital.
        short_ma: Short MA period (default 20).
        long_ma: Long MA period (default 75 for no-night-session).
        margin_ratio: Fraction of equity used per position.

    Returns:
        BacktestResult with equity curve and trade journal.
    """
    print(
        f"[BT] Starting multi-contract backtest "
        f"({len(periods)} periods, {len(contract_data)} contracts)"
    )

    # Pre-compute signals for each contract
    contract_signals: dict[str, pd.DataFrame] = {}
    for c, df in contract_data.items():
        sig = _compute_contract_signals(df, short_ma, long_ma)
        contract_signals[c] = sig

    # Build a unified time index across all periods
    all_hours: list[datetime] = []
    for p in periods:
        if p.contract not in contract_data:
            continue
        cdf = contract_data[p.contract]
        mask = (cdf.index >= p.start) & (cdf.index <= p.end)
        hours = cdf.index[mask]
        all_hours.extend(hours.tolist())

    all_hours = sorted(set(all_hours))
    print(
        f"[BT] Unified timeline: {len(all_hours)} hours from "
        f"{all_hours[0]} to {all_hours[-1]}"
    )

    # State
    state = PositionState()
    journal = TradeJournal()
    equity_curve: list[float] = []
    balance = initial_capital
    warmup = max(short_ma, long_ma)
    contract_history: list[str] = []

    for bar_idx, hour_dt in enumerate(all_hours):
        # Determine which contract is main at this hour
        main_contract = ""
        for p in periods:
            if p.start <= hour_dt <= p.end:
                main_contract = p.contract
                break

        # For exit: use the HELD contract's own signals
        exit_contract = main_contract
        if state.has_position:
            exit_contract = state.contract

        exit_sig = exit_contract in contract_signals
        main_sig = main_contract in contract_signals

        if not exit_sig or not main_sig:
            equity_curve.append(balance)
            contract_history.append(
                main_contract if main_contract else "unknown"
            )
            continue

        sig_df = contract_signals[main_contract]
        if hour_dt not in sig_df.index:
            equity_curve.append(balance)
            contract_history.append(main_contract)
            continue

        entry_sig_row = sig_df.loc[hour_dt]

        if state.has_position:
            hold_sig_df = contract_signals[exit_contract]
            if hour_dt not in hold_sig_df.index:
                equity_curve.append(balance)
                contract_history.append(main_contract)
                continue

            exit_sig_row = hold_sig_df.loc[hour_dt]

            signal_dict = {
                "zone": exit_sig_row.get("zone"),
                "signal": exit_sig_row.get("signal"),
                "entry_price": exit_sig_row.get("entry_price"),
                "upper_bound": exit_sig_row.get("upper_bound"),
                "lower_bound": exit_sig_row.get("lower_bound"),
                "close": (
                    float(contract_data[exit_contract].loc[hour_dt, "close"])
                    if hour_dt in contract_data[exit_contract].index
                    else 0.0
                ),
            }

            if bar_idx < warmup:
                equity_curve.append(balance)
                contract_history.append(main_contract)
                continue

            def _sizing(margin_val):
                return compute_position_size(
                    balance, margin_val, margin_ratio
                )

            new_state, trades = _process_signal(
                state,
                bar_idx,
                close=signal_dict.get("close", 0.0),
                upper_bound=signal_dict.get("upper_bound", 0.0),
                lower_bound=signal_dict.get("lower_bound", 0.0),
                entry_price=signal_dict.get("entry_price", 0.0),
                signal=signal_dict.get("signal", "hold"),
                zone=signal_dict.get("zone", ""),
                sizing_fn=_sizing,
            )

            for t in trades:
                journal.record(
                    t["bar_index"],
                    t["action"],
                    t["price"],
                    t["size"],
                    t["pnl"],
                    t["reason"],
                )
                if "close" in t["action"]:
                    balance += t["pnl"]

            state = new_state

        else:
            entry_sig = entry_sig_row.get("signal", "hold")

            if bar_idx < warmup:
                equity_curve.append(balance)
                contract_history.append(main_contract)
                continue

            def _sizing(margin_val):
                return compute_position_size(
                    balance, margin_val, margin_ratio
                )

            signal_dict = {
                "zone": entry_sig_row.get("zone"),
                "signal": entry_sig,
                "entry_price": entry_sig_row.get("entry_price"),
                "upper_bound": entry_sig_row.get("upper_bound"),
                "lower_bound": entry_sig_row.get("lower_bound"),
                "close": (
                    float(contract_data[main_contract].loc[hour_dt, "close"])
                    if hour_dt in contract_data[main_contract].index
                    else 0.0
                ),
            }

            new_state, trades = _process_signal(
                state,
                bar_idx,
                close=signal_dict.get("close", 0.0),
                upper_bound=signal_dict.get("upper_bound", 0.0),
                lower_bound=signal_dict.get("lower_bound", 0.0),
                entry_price=signal_dict.get("entry_price", 0.0),
                signal=signal_dict.get("signal", "hold"),
                zone=signal_dict.get("zone", ""),
                sizing_fn=_sizing,
            )

            for t in trades:
                t["contract"] = main_contract
                journal.record(
                    t["bar_index"],
                    t["action"],
                    t["price"],
                    t["size"],
                    t["pnl"],
                    t["reason"],
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

            if state.direction == "long" and state.entry_price > 0:
                unrealized = (
                    current_price - state.entry_price
                ) * state.position_size
            elif state.direction == "short" and state.entry_price > 0:
                unrealized = (
                    state.entry_price - current_price
                ) * state.position_size
            else:
                unrealized = 0.0
            current_equity = balance + unrealized
        else:
            current_equity = balance

        equity_curve.append(current_equity)
        contract_history.append(main_contract)

        if bar_idx > 0 and bar_idx % 5000 == 0:
            print(
                f"[BT] {bar_idx}/{len(all_hours)} bars, "
                f"equity={current_equity:.0f}, "
                f"trades={len(journal.records)}"
            )

    # Results
    final_equity = (
        equity_curve[-1] if equity_curve else initial_capital
    )
    total_return = (final_equity / initial_capital) - 1.0
    close_records = [
        r for r in journal.records if "close" in r.get("action", "")
    ]
    num_trades = len(close_records)

    print(
        f"[BT] DONE: {num_trades} trades, {final_equity:.0f} final equity, "
        f"{total_return * 100:+.2f}% return"
    )

    return BacktestResult(
        symbol="multi-contract",
        equity_curve=equity_curve,
        trade_journal=journal,
        final_equity=final_equity,
        total_return=total_return,
        num_trades=num_trades,
        contract_history=contract_history,
    )
