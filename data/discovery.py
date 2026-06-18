"""Main contract discovery — daily volume-based selection.

Matches GPT framework approach: for each trade day, scan all available
contracts and pick the one with highest daily volume as the "main contract".
Then use THAT contract's hourly data for MA calculation.

Caches daily volume data in data/vol_cache/ for fast re-runs.

USAGE:
  1st run:  discover_main_contracts() — downloads + caches daily data (slow)
  2nd+ run: discover_main_contracts() — reads cache only (fast)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from tqsdk import TqApi, TqAuth

from config.credentials import get_credentials
from data.naming import contract_names as _generate_names


@dataclass
class ContractPeriod:
    start: datetime
    end: datetime
    contract: str


VOL_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "vol_cache"


def generate_contract_names(variety: str, start_year: int, end_year: int) -> list[str]:
    """Generate all possible contract names, exchange-aware."""
    return _generate_names(variety, start_year, end_year)


def _cached_path(contract: str) -> Path:
    safe = contract.replace(".", "_")
    return VOL_CACHE_DIR / f"{safe}_daily.parquet"


def _download_and_cache(contract: str, api: TqApi) -> pd.DataFrame | None:
    """Download daily data from tqsdk, cache, and return. Returns None on failure."""
    try:
        klines = api.get_kline_serial(contract, 86400, data_length=2000)
    except Exception:
        return None
    if klines is None or len(klines) == 0:
        return None

    df = klines.copy()
    df = df[df["volume"] > 0]
    if len(df) == 0:
        return None

    df["datetime"] = (
        pd.to_datetime(df["datetime"], unit="ns", utc=True)
        .dt.tz_convert("Asia/Shanghai")
    )
    df = df.sort_values("datetime")
    df["date"] = df["datetime"].dt.normalize()
    daily = (
        df.groupby("date", as_index=False)
        .agg(volume=("volume", "sum"), open_oi=("open_oi", "last"))
    )
    if len(daily) == 0:
        return None

    VOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(_cached_path(contract))
    return daily


def discover_main_contracts(
    variety: str = "DCE.jd",
    start_year: int = 2022,
    end_year: int = 2026,
    verbose: bool = True,
) -> list[ContractPeriod]:
    """Discover main contract periods by scanning all contracts' daily volume.

    Returns chronological list of ContractPeriod records.
    """
    contracts = generate_contract_names(variety, start_year, end_year)

    # Step 1: read cached data for all contracts
    cached_count = 0
    all_daily: list[pd.DataFrame] = []

    for c in contracts:
        cp = _cached_path(c)
        if not cp.exists():
            continue
        try:
            df = pd.read_parquet(cp)
            if df is not None and not df.empty:
                df = df.copy()
                df["contract"] = c
                yearly = df[df["date"].dt.year.between(start_year - 1, end_year)]
                if len(yearly) > 0:
                    all_daily.append(yearly)
                    cached_count += 1
        except Exception:
            pass

    if verbose:
        print(f"[DISC] Read {cached_count} cached contracts for {variety}")
        print(f"[DISC] {len(all_daily)} have data in window")

    # If not enough cached, download missing (first run)
    need_download = len(contracts) - cached_count
    if cached_count > 10:  # have some data already, use cache only
        if verbose and need_download > 0:
            print(f"[DISC] {cached_count}/{len(contracts)} contracts cached, {need_download} missing (download skipped)")
    else:  # very few cached, need full download
        if verbose:
            print(f"[DISC] Only {cached_count}/{len(contracts)} cached, downloading missing...")
        creds = get_credentials()
        api = TqApi(auth=TqAuth(creds["username"], creds["password"]))
        try:
            for i, c in enumerate(contracts):
                if _cached_path(c).exists():
                    continue
                daily = _download_and_cache(c, api)
                if daily is None:
                    continue
                yearly = daily[daily["date"].dt.year.between(start_year - 1, end_year)]
                if len(yearly) == 0:
                    continue
                yearly = yearly.copy()
                yearly["contract"] = c
                all_daily.append(yearly)
                if verbose:
                    print(f"  [{i+1}/{len(contracts)}] {c}: {yearly['date'].iloc[0].date()} -> {yearly['date'].iloc[-1].date()}, vol {yearly['volume'].sum():.0f}")
        finally:
            api.close()

    if not all_daily:
        return []

    # Step 2: pick main contract per day
    combined = pd.concat(all_daily, ignore_index=True)
    combined = combined[combined["date"].dt.year.between(start_year, end_year)].copy()
    if combined.empty:
        return []

    combined = combined.sort_values(["date", "volume"], ascending=[True, False])
    main_daily = combined.groupby("date", as_index=False).first()
    main_daily = main_daily.sort_values("date")

    if verbose:
        print(f"[DISC] {len(main_daily)} trading days, {main_daily['contract'].nunique()} unique contracts as main")

    # Step 3: build periods from consecutive same-contract days
    periods: list[ContractPeriod] = []
    current_contract = None
    period_start = None

    for _, row in main_daily.iterrows():
        dt = row["date"]
        dt_start = datetime(dt.year, dt.month, dt.day, 9, 0, 0, tzinfo=timezone(timedelta(hours=8)))
        c = str(row["contract"])

        if c != current_contract:
            if current_contract is not None and period_start is not None:
                periods.append(ContractPeriod(start=period_start, end=dt_start, contract=current_contract))
            current_contract = c
            period_start = dt_start

    if current_contract is not None and period_start is not None:
        last_dt = main_daily["date"].iloc[-1]
        period_end = datetime(last_dt.year, last_dt.month, last_dt.day, 15, 0, 0, tzinfo=timezone(timedelta(hours=8)))
        periods.append(ContractPeriod(start=period_start, end=period_end, contract=current_contract))

    if verbose:
        print(f"[DISC] {len(periods)} contract periods:")
        for p in periods:
            print(f"  {p.contract}: {p.start.date()} → {p.end.date()} ({(p.end - p.start).days} days)")

    return periods
