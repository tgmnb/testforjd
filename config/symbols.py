"""Symbol configuration and night-session to long-MA mapping.

NIGHT_MA_MAP maps the decimal hour of a night session's end time
to the corresponding long moving average period:
    - 0 (no night session)    → 75
    - 23 (ends at 23:00)      → 115
    - 1 (ends at 01:00)       → 150
    - 2.5 (ends at 02:30)     → 185
"""

from __future__ import annotations

from typing import Any

NIGHT_MA_MAP: dict[float, int] = {
    0: 75,
    23: 115,
    1: 150,
    2.5: 185,
}

# v1 whitelist: JD (Eggs) only
WHITELIST: tuple[str, ...] = ("KQ.m@DCE.jd",)

# Example illiquid varieties (optional stub)
BLACKLIST: tuple[str, ...] = (
    "KQ.m@CZC.ZC",
    "KQ.m@CZC.JR",
)

SYMBOL_INFO: dict[str, dict[str, str | float | None]] = {
    "KQ.m@DCE.jd": {
        "name": "JD (Eggs)",
        "exchange": "DCE",
        "night_end_hour": None,  # no night session
    },
}


def get_long_ma_period(quote: dict[str, Any]) -> int:
    """Determine the long MA period for a symbol based on its night session.

    Args:
        quote: A tqsdk quote dict, expected to have a ``trading_time`` key
               with a ``night`` sub-key (list of [start, end] strings).

    Returns:
        The long MA period from ``NIGHT_MA_MAP``.
    """
    night_sessions = quote.get("trading_time", {}).get("night", [])
    if not night_sessions:
        return NIGHT_MA_MAP[0]

    try:
        end_time_str = night_sessions[0][1]
    except (IndexError, TypeError):
        return NIGHT_MA_MAP[0]

    if not isinstance(end_time_str, str) or ":" not in end_time_str:
        return NIGHT_MA_MAP[0]

    parts = end_time_str.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0

    if hour >= 24:
        # Trading-day time format: hour >= 24 means next calendar day
        # but same trading day. All such sessions map to 185.
        return NIGHT_MA_MAP[2.5]

    decimal_hour = hour + minute / 60.0
    return NIGHT_MA_MAP.get(decimal_hour, NIGHT_MA_MAP[23])
