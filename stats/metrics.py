"""Performance metrics — return, Sharpe, drawdown, win rate, etc.

No tqsdk dependency.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def calculate_metrics(
    equity_curve: list[float],
    trade_journal: list[dict[str, Any]],
    risk_free_rate: float = 0.02,
    resolution_hours: int = 1,
) -> dict[str, Any]:
    """Compute performance metrics from an equity curve and trade journal.

    Args:
        equity_curve: List of equity values per bar.
        trade_journal: List of trade records (close actions with ``pnl``).
        risk_free_rate: Annual risk-free rate (default 0.02 = 2%).
        resolution_hours: Bar duration in hours (default 1 for hourly).

    Returns:
        Dict with keys: ``total_return``, ``annualized_return``,
        ``win_rate``, ``profit_factor``, ``sharpe_ratio``,
        ``max_drawdown``, ``max_drawdown_duration``, ``num_trades``,
        ``num_wins``.
    """
    eq = pd.Series(equity_curve)
    initial = equity_curve[0] if equity_curve else 0.0
    final = equity_curve[-1] if equity_curve else 0.0

    total_return = (final / initial - 1.0) if initial > 0 else 0.0

    # Annualized return
    n_bars = len(equity_curve)
    bars_per_year = 252 * 24 // resolution_hours if resolution_hours > 0 else 1
    if n_bars > 0 and total_return > -1.0:
        annualized_return = (1.0 + total_return) ** (bars_per_year / n_bars) - 1.0
    else:
        annualized_return = 0.0

    # Win rate
    close_trades = [t for t in trade_journal if "close" in t.get("action", "")]
    num_trades = len(close_trades)
    num_wins = sum(1 for t in close_trades if t.get("pnl", 0) > 0)
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in close_trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in close_trades if t.get("pnl", 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # Sharpe ratio (using daily returns approximation)
    returns = eq.pct_change().dropna()
    if len(returns) > 1:
        daily_returns = returns.resample("D").apply(lambda x: (1 + x).prod() - 1) if hasattr(returns.index, "freq") else returns
        excess = daily_returns - risk_free_rate / 252  # daily risk-free rate
        if excess.std() > 0:
            sharpe_ratio = (excess.mean() / excess.std()) * math.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    peak = eq.expanding().max()
    drawdown = (eq - peak) / peak
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Max drawdown duration
    is_drawdown = drawdown < 0
    if is_drawdown.any():
        # Count longest streak of drawdown
        streaks = _streak_lengths(is_drawdown)
        max_drawdown_duration = max(streaks) if streaks else 0
    else:
        max_drawdown_duration = 0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "num_trades": num_trades,
        "num_wins": num_wins,
    }


def _streak_lengths(series: pd.Series) -> list[int]:
    """Compute lengths of consecutive True runs in a boolean Series."""
    lengths: list[int] = []
    current = 0
    for v in series:
        if v:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths
