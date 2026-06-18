"""Indicator computation and utilities — used by the portfolio engine.

Core functions:
- add_indicators: MA20, MA_long, zones, signals
- detect_session_type: night-session → MA_long mapping
- ohlc_fill_price: OHLC range check for exits
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.strategy import SLIPPAGE_RATE


# ---------------------------------------------------------------------------
# Fill helpers
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
# Signal helpers
# ---------------------------------------------------------------------------

def detect_session_type(df: pd.DataFrame) -> int:
    """Detect session type from hourly data, return appropriate MA_long.
    
    Checks the hour values to determine if nighttime trading exists.
    day_only → MA_long=75
    night_2100 (till 23:00) → MA_long=115
    night_0100 → MA_long=150
    night_0230 → MA_long=185
    """
    hours = set(df.index.hour.tolist())
    if 2 in hours:
        return 185
    if 0 in hours or 1 in hours:
        return 150
    if 22 in hours or 23 in hours or 21 in hours:
        return 115
    return 75


def add_indicators(df: pd.DataFrame, short_period: int, long_period: int) -> pd.DataFrame:
    """Compute MA20, MA_long, zones and entry signals.

    Returns same DataFrame with added columns:
    [ma20, ma20_slope, ma_long, upper, lower, zone, prev_zone]
    """
    out = df.copy().sort_index()
    out["ma20"] = out["close"].rolling(short_period, min_periods=short_period).mean()
    out["ma20_slope"] = out["ma20"] - out["ma20"].shift(1)
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
