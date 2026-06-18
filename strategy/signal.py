"""Signal generation — moving averages, tri-zone logic, entry signals.

All functions are pure: they take DataFrames and return DataFrames/Series
with no side effects and no tqsdk dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mas(
    df: pd.DataFrame,
    short_period: int = 20,
    long_period: int = 75,
) -> pd.DataFrame:
    """Compute short and long moving averages.

    Args:
        df: DataFrame with a ``'close'`` column.
        short_period: Period for the short MA (default 20).
        long_period: Period for the long MA (default 75).

    Returns:
        DataFrame with columns ``ma_short`` and ``ma_long``.
        Rows with insufficient data have NaN.
    """
    out = pd.DataFrame(index=df.index)
    out["ma_short"] = df["close"].rolling(window=short_period, min_periods=short_period).mean()
    out["ma_long"] = df["close"].rolling(window=long_period, min_periods=long_period).mean()
    return out


def determine_zones(
    mas: pd.DataFrame,
    close: pd.Series,
) -> pd.Series:
    """Classify each bar into a price zone relative to the two MAs.

    Zone rules:
        - ``"bullish"`` if ``close > upper_bound``
        - ``"bearish"`` if ``close < lower_bound``
        - ``"middle"`` otherwise

    Where:
        - ``upper_bound = max(ma_short, ma_long)``
        - ``lower_bound = min(ma_short, ma_long)``

    Returns:
        A Series of zone labels aligned to the input index.
    """
    upper = mas[["ma_short", "ma_long"]].max(axis=1)
    lower = mas[["ma_short", "ma_long"]].min(axis=1)

    zones = pd.Series("middle", index=mas.index)
    zones[close > upper] = "bullish"
    zones[close < lower] = "bearish"

    # NaN for rows where any MA is NaN
    zones[upper.isna() | lower.isna()] = np.nan

    return zones


def compute_signal(
    df: pd.DataFrame,
    short_period: int = 20,
    long_period: int = 75,
) -> pd.DataFrame:
    """Compute entry signals for each bar.

    Signal logic:
        1. Compute MA20 and MA_long.
        2. Determine zones.
        3. MA20 direction: +1 if rising, -1 if falling.
        4. Entry signal when the price crosses INTO a zone AND MA20
           direction matches:
            - ``"open_long"``: previous zone != bullish → current bullish,
              MA20 rising.
            - ``"open_short"``: previous zone != bearish → current bearish,
              MA20 falling.
            - ``"hold"``: otherwise.

    Returns:
        DataFrame with columns ``ma_short``, ``ma_long``, ``upper_bound``,
        ``lower_bound``, ``zone``, ``signal``, ``entry_price``.
    """
    mas = compute_mas(df, short_period, long_period)
    zones = determine_zones(mas, df["close"])

    # MA20 direction: +1 rising, -1 falling
    ma20_direction = np.sign(mas["ma_short"].diff()).fillna(0)

    # Zone boundaries
    upper_bound = mas[["ma_short", "ma_long"]].max(axis=1)
    lower_bound = mas[["ma_short", "ma_long"]].min(axis=1)

    # Previous-bar zone
    prev_zone = zones.shift(1)

    # Signal logic
    signal = pd.Series("hold", index=df.index)
    entry_price = pd.Series(np.nan, index=df.index)

    # Open long: previous NOT bullish, current IS bullish, MA20 up
    long_condition = (
        (prev_zone != "bullish")
        & (zones == "bullish")
        & (ma20_direction == 1)
    )
    signal[long_condition] = "open_long"
    entry_price[long_condition] = upper_bound[long_condition]

    # Open short: previous NOT bearish, current IS bearish, MA20 down
    short_condition = (
        (prev_zone != "bearish")
        & (zones == "bearish")
        & (ma20_direction == -1)
    )
    signal[short_condition] = "open_short"
    entry_price[short_condition] = lower_bound[short_condition]

    # NaN signals where zones are NaN (insufficient data)
    signal[zones.isna()] = "hold"

    out = pd.DataFrame(index=df.index)
    out["ma_short"] = mas["ma_short"]
    out["ma_long"] = mas["ma_long"]
    out["upper_bound"] = upper_bound
    out["lower_bound"] = lower_bound
    out["zone"] = zones
    out["signal"] = signal
    out["entry_price"] = entry_price

    return out
