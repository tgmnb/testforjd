"""Equity curve and drawdown plotting.

Uses ``matplotlib`` with ``Agg`` backend for headless environments.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_equity_curve(
    equity_curve: list[float],
    title: str = "Equity Curve",
    save_path: str = "equity.png",
) -> str:
    """Plot and save the equity curve with drawdown shading.

    Args:
        equity_curve: List of equity values.
        title: Chart title.
        save_path: Path to save the PNG file.

    Returns:
        The *save_path* where the plot was saved.
    """
    eq = pd.Series(equity_curve)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Equity curve line
    ax.plot(eq.index, eq.values, color="navy", linewidth=1.5, label="Equity")

    peak = eq.expanding().max()
    ax.fill_between(
        eq.index, eq.values, peak.values,
        where=eq.values < peak.values,
        color="red", alpha=0.15, label="Drawdown",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    _save_or_warn(fig, save_path)
    plt.close(fig)
    return save_path


def plot_drawdown(
    equity_curve: list[float],
    save_path: str = "drawdown.png",
) -> str:
    """Plot the drawdown percentage over time.

    Args:
        equity_curve: List of equity values.
        save_path: Path to save the PNG file.

    Returns:
        The *save_path* where the plot was saved.
    """
    eq = pd.Series(equity_curve)
    peak = eq.expanding().max()
    drawdown_pct = (eq - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(
        drawdown_pct.index, drawdown_pct.values, 0,
        where=drawdown_pct.values < 0,
        color="red", alpha=0.4,
    )
    ax.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    _save_or_warn(fig, save_path)
    plt.close(fig)
    return save_path


def _save_or_warn(fig: matplotlib.figure.Figure, path: str) -> None:
    """Save figure, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
