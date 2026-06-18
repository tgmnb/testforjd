"""Cost computation — slippage, trade P&L, and return series.

No tqsdk dependency.
"""

from __future__ import annotations

import pandas as pd

from config.strategy import SLIPPAGE


def apply_slippage(
    price: float,
    action: str,
    slippage_rate: float = SLIPPAGE,
) -> float:
    """Apply slippage to *price* based on *action* and *slippage_rate*.

    Buy operations (open_long, close_short): ``price * (1 + slippage_rate)``
    Sell operations (close_long, open_short): ``price * (1 - slippage_rate)``
    """
    if action in ("open_long", "close_short"):
        return price * (1.0 + slippage_rate)
    return price * (1.0 - slippage_rate)


def compute_trade_pl(
    entry_price: float,
    exit_price: float,
    direction: str,
    units: float,
) -> float:
    """Compute realised P&L for a closed trade.

    Long: ``(exit_price - entry_price) * units``
    Short: ``(entry_price - exit_price) * units``
    """
    if direction == "long":
        return (exit_price - entry_price) * units
    return (entry_price - exit_price) * units


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    """Compute percentage returns from an equity curve.

    The first value is NaN (no prior period).
    """
    if len(equity_curve) < 2:
        return pd.Series([], dtype=float)
    return equity_curve.pct_change().dropna()
