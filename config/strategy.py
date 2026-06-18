"""Strategy parameter configuration.

All values are shared across the backtesting framework.
"""

from __future__ import annotations

# K-line resolution in seconds (3600 = 1 hour)
RESOLUTION: int = 3600

# Short moving average period
MA_SHORT: int = 20

# Slippage rate per fill (0.002 = 0.2%)
SLIPPAGE: float = 0.002
SLIPPAGE_RATE: float = 0.002  # GPT-style alias

# Commission per contract per side (yuan/手)
# JD (鸡蛋) at DCE: ~7.0 元/手
COMMISSION_PER_CONTRACT: float = 7.0

# Fraction of equity used as margin budget per position
MARGIN_RATIO: float = 0.10
MARGIN_FRACTION: float = 0.10  # GPT-style alias

# Contract multiplier (JD: 10 吨/手)
CONTRACT_MULTIPLIER: int = 10

# Leverage
LEVERAGE: int = 10

# Stop-loss: fixed percentage from entry price
STOP_LOSS_PCT: float = 0.02  # 2%

# Initial capital for backtesting
INITIAL_CAPITAL: float = 100_000.0

# Display label for the timeframe
TIMEFRAME_DISPLAY: str = "1h"
