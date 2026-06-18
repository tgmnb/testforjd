"""Main contract discovery — OI ratio method.

Detects contract roll dates from OI discontinuities in KQ.m@ data,
then infers which contract is active in each segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta

import pandas as pd
from tqsdk import TqApi, TqAuth, TqBacktest, BacktestFinished

from .credentials import get_credentials


@dataclass
class ContractPeriod:
    start: datetime
    end: datetime
    contract: str


def discover_main_contracts(
    variety: str = "DCE.jd",
    start_year: int = 2022,
    end_year: int = 2026,
    oi_ratio_threshold: float = 1.4,
    verbose: bool = True,
) -> list[ContractPeriod]:
    """Discover main contract periods from KQ.m@ OI ratio changes.

    1. Load KQ.m@ daily data (one API call)
    2. Detect roll dates where OI ratio > threshold (contract switch)
    3. Segment data between rolls
    4. Infer contract name for each segment from the date

    Returns:
        Chronological list of ContractPeriod records.
    """
    creds = get_credentials()
    if verbose:
        print(f"[DISC] Loading KQ.m@{variety} daily data...")

    api = TqApi(auth=TqAuth(creds["username"], creds["password"]))
    try:
        klines = api.get_kline_serial(
            f"KQ.m@{variety}", 86400, data_length=2000
        )
    finally:
        api.close()

    if klines is None or len(klines) == 0:
        return []

    df = klines.copy()
    df["datetime"] = (
        pd.to_datetime(df["datetime"], unit="ns", utc=True)
        .dt.tz_convert("Asia/Shanghai")
    )
    df = df.sort_values("datetime").set_index("datetime")

    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

    if verbose:
        print(
            f"[DISC] {len(df)} daily bars from "
            f"{df.index[0].date()} to {df.index[-1].date()}"
        )

    # Detect roll dates from OI ratio
    df["oi_ratio"] = df["open_oi"] / df["open_oi"].shift(1)
    roll_mask = (df["oi_ratio"] > oi_ratio_threshold) | (
        df["oi_ratio"] < 1.0 / oi_ratio_threshold
    )
    roll_dates = df.index[roll_mask].tolist()

    if verbose:
        print(f"[DISC] {len(roll_dates)} roll dates detected")

    # Build periods
    all_boundaries = [df.index[0]] + roll_dates
    segments = []

    for i in range(len(all_boundaries)):
        seg_start = all_boundaries[i]
        if i + 1 < len(all_boundaries):
            seg_end = all_boundaries[i + 1]
        else:
            seg_end = df.index[-1]

        if seg_end <= seg_start:
            continue

        contract = _infer_contract(seg_start, variety.split(".")[-1])

        segments.append(
            ContractPeriod(
                start=seg_start,
                end=seg_end,
                contract=contract,
            )
        )

    segments = _merge_same_contracts(segments)
    segments = [s for s in segments if (s.end - s.start).days >= 5]

    if verbose:
        print(f"[DISC] {len(segments)} contract periods after filtering:")
        for s in segments:
            print(
                f"  {s.contract}: {s.start.date()} → {s.end.date()} "
                f"({(s.end - s.start).days} days)"
            )

    return segments


def _infer_contract(dt: datetime, suffix: str = "jd") -> str:
    """Infer contract symbol from a date.

    The main contract at date *dt* is typically one whose delivery month
    is 2-3 months after *dt*.
    """
    mm = dt.month + 3
    yy = dt.year
    while mm > 12:
        mm -= 12
        yy += 1
    return f"DCE.{suffix}{(yy % 100):02d}{mm:02d}"


def _merge_same_contracts(
    periods: list[ContractPeriod],
) -> list[ContractPeriod]:
    if not periods:
        return []

    merged = [periods[0]]
    for p in periods[1:]:
        if p.contract == merged[-1].contract:
            merged[-1].end = p.end
        else:
            merged.append(p)

    return merged
