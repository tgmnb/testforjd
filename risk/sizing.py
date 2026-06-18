"""Position sizing — pure functions for computing trade sizes and margin checks.

No tqsdk dependency.
"""

from __future__ import annotations

import math


def compute_position_size(
    equity: float,
    margin_per_contract: float,
    margin_ratio: float = 0.10,
) -> int:
    """Compute how many contracts can be opened given current equity.

    Args:
        equity: Current account equity.
        margin_per_contract: Real-time margin per contract from the quote.
        margin_ratio: Fraction of equity to use as margin budget (default 0.10).

    Returns:
        Number of contracts (integer). 0 if insufficient margin.
    """
    if margin_per_contract <= 0:
        return 0
    margin_budget = equity * margin_ratio
    if margin_budget < margin_per_contract:
        return 0
    return int(math.floor(margin_budget / margin_per_contract))


def compute_required_margin(
    position_value: float,
    margin_ratio: float = 0.10,
) -> float:
    """Compute required margin for a position of *position_value*."""
    return position_value * margin_ratio


def check_margin_sufficiency(equity: float, margin_required: float) -> bool:
    """Check whether *equity* is sufficient to cover *margin_required*."""
    return equity >= margin_required
