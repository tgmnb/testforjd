"""Strategy parameter configuration.

All values are shared across the backtesting framework.
"""

from __future__ import annotations

# K-line resolution in seconds (3600 = 1 hour)
RESOLUTION: int = 3600

# Short moving average period
MA_SHORT: int = 20

# Slippage rate per fill (0.004 = 0.4%)
# Applied as: buy → price * (1 + SLIPPAGE), sell → price * (1 - SLIPPAGE)
SLIPPAGE: float = 0.004

# Fraction of equity used as margin budget per position
MARGIN_RATIO: float = 0.10

# Initial capital for backtesting
INITIAL_CAPITAL: float = 100_000.0

# Display label for the timeframe
TIMEFRAME_DISPLAY: str = "1h"
