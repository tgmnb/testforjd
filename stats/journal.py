"""Trade journal — records, CSV output, and summary statistics.

No tqsdk dependency.
"""

from __future__ import annotations

import csv
from typing import Any

import pandas as pd


class TradeJournal:
    """Accumulates trade records and provides export methods.

    Each record is a dict with keys: ``bar_index``, ``action``, ``price``,
    ``size``, ``pnl``, ``reason``.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

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

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a pandas DataFrame."""
        return pd.DataFrame(self.records)

    def to_csv(self, path: str) -> None:
        """Write all records to a CSV file at *path*."""
        if not self.records:
            # Write header only
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["bar_index", "action", "price", "size", "pnl", "reason"]
                )
                writer.writeheader()
            return

        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def summary(self) -> dict[str, Any]:
        """Compute aggregate trade statistics.

        Returns:
            Dict with keys: ``total_trades``, ``long_trades``,
            ``short_trades``, ``gross_profit``, ``gross_loss``.
        """
        long_actions = [r for r in self.records if r["action"] in ("open_long", "close_long")]
        short_actions = [r for r in self.records if r["action"] in ("open_short", "close_short")]
        close_actions = [r for r in self.records if "close" in r["action"]]

        gross_profit = sum(r["pnl"] for r in close_actions if r["pnl"] > 0)
        gross_loss = sum(r["pnl"] for r in close_actions if r["pnl"] < 0)

        return {
            "total_trades": len(self.records),
            "long_trades": len(long_actions),
            "short_trades": len(short_actions),
            "gross_profit": gross_profit,
            "gross_loss": abs(gross_loss),
        }
