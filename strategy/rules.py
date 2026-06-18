"""Trade rules — position state machine, slippage, and bar-level processing.

This module is pure logic with no tqsdk dependency. It manages the
open/close state machine that turns signal rows into trade actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from config.strategy import SLIPPAGE


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """Current position state.

    Attributes:
        has_position: Whether a position is currently held.
        direction: ``"long"``, ``"short"``, or ``"none"``.
        entry_price: Price at which the position was entered (before slippage).
        entry_bar: Bar index of the entry.
        position_size: Number of contracts held.
    """
    has_position: bool = False
    direction: str = "none"
    entry_price: float = 0.0
    entry_bar: int = 0
    position_size: float = 0.0


@dataclass
class TradeJournal:
    """Accumulates trade records as a list of dicts.

    Each record has keys: ``bar_index``, ``action``, ``price``, ``size``,
    ``pnl``, ``reason``.
    """
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        bar_index: int,
        action: str,
        price: float,
        size: float,
        pnl: float = 0.0,
        reason: str = "",
    ) -> None:
        self.records.append({
            "bar_index": bar_index,
            "action": action,
            "price": price,
            "size": size,
            "pnl": pnl,
            "reason": reason,
        })


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------

def apply_slippage(price: float, action: str) -> float:
    """Apply slippage to *price* based on the trade *action*.

    Rules:
        - Buy operations (open_long, close_short): ``price * (1 + SLIPPAGE)``
        - Sell operations (close_long, open_short): ``price * (1 - SLIPPAGE)``
    """
    if action in ("open_long", "close_short"):
        return price * (1.0 + SLIPPAGE)
    # close_long, open_short
    return price * (1.0 - SLIPPAGE)


# ---------------------------------------------------------------------------
# Bar processing
# ---------------------------------------------------------------------------

def process_bar(
    state: PositionState,
    signal_row: dict[str, Any],
    bar_index: int,
    position_size_fn: Callable,
) -> tuple[PositionState, TradeJournal]:
    """Process one bar's signal and update the position state.

    Args:
        state: Current position state (will be copied on modification).
        signal_row: Dict with keys ``zone``, ``signal``, ``entry_price``,
            ``close`` (as returned by ``compute_signal``).
        bar_index: Index of the current bar (for journaling).
        position_size_fn: Callable ``(equity, margin_per_contract) → int``
            that returns the number of contracts to trade.

    Returns:
        ``(new_state, journal)`` where *new_state* is the updated position
        and *journal* contains any trade actions taken this bar.
    """
    new_state = PositionState(
        has_position=state.has_position,
        direction=state.direction,
        entry_price=state.entry_price,
        entry_bar=state.entry_bar,
        position_size=state.position_size,
    )
    journal = TradeJournal()

    zone = signal_row.get("zone")
    sig = signal_row.get("signal")
    close = signal_row.get("close", 0.0)
    entry_price = signal_row.get("entry_price", 0.0)
    upper_bound = signal_row.get("upper_bound", 0.0)
    lower_bound = signal_row.get("lower_bound", 0.0)

    if new_state.has_position:
        # --- Position management: check for exit ---
        if new_state.direction == "long" and zone != "bullish":
            # Close long at UPPER bound per spec: "按上边界价格平多"
            exit_price = apply_slippage(upper_bound, "close_long")
            pnl = (exit_price - apply_slippage(new_state.entry_price, "open_long")) * new_state.position_size
            journal.record(
                bar_index, "close_long", exit_price,
                new_state.position_size, pnl, "zone_exit",
            )
            new_state.has_position = False
            new_state.direction = "none"
            new_state.position_size = 0.0

        elif new_state.direction == "short" and zone != "bearish":
            # Close short at LOWER bound per spec: "按下边界价格平空"
            exit_price = apply_slippage(lower_bound, "close_short")
            pnl = (apply_slippage(new_state.entry_price, "open_short") - exit_price) * new_state.position_size
            journal.record(
                bar_index, "close_short", exit_price,
                new_state.position_size, pnl, "zone_exit",
            )
            new_state.has_position = False
            new_state.direction = "none"
            new_state.position_size = 0.0

    else:
        # --- No position: check for entry ---
        if sig == "open_long":
            size = position_size_fn()
            if size and size > 0:
                filled_price = apply_slippage(entry_price, sig)
                new_state.has_position = True
                new_state.direction = "long"
                new_state.entry_price = entry_price
                new_state.entry_bar = bar_index
                new_state.position_size = size
                journal.record(
                    bar_index, "open_long", filled_price,
                    size, 0.0, "signal_entry",
                )

        elif sig == "open_short":
            size = position_size_fn()
            if size and size > 0:
                filled_price = apply_slippage(entry_price, sig)
                new_state.has_position = True
                new_state.direction = "short"
                new_state.entry_price = entry_price
                new_state.entry_bar = bar_index
                new_state.position_size = size
                journal.record(
                    bar_index, "open_short", filled_price,
                    size, 0.0, "signal_entry",
                )

    return new_state, journal
