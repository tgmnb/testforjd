"""Contract data downloader — fetches individual contract hourly klines.

Each contract's data is cropped at the last viable trading day for
individual investors (last trading day of month before delivery month).
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from tqsdk import TqApi, TqAuth

from data.discovery import ContractPeriod
from config.credentials import get_credentials


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "contract_cache"


def _delivery_month_last_day(contract: str) -> date:
    """Last calendar day of the month BEFORE delivery month.

    Individual investors cannot hold into delivery month.
    For a DCE contract DCE.jdYYMM or DCE.pYYMM:
    last viable trading day = last day of (MM-1).
    """
    # Extract YYMM from contract code: standard=4digits(YYMM), CZCE=3digits(YMM)
    import re
    digits = re.search(r'(\d{3,4})$', contract)
    if not digits:
        return date(2099, 12, 31)
    d = digits.group(1)
    if len(d) == 4:
        yy = int(d[:2]) + 2000
        mm = int(d[2:])
    else:  # CZCE: 3 digits, first = year digit, last 2 = month
        yy = 2020 + int(d[0]) if int(d[0]) <= 9 else 2000 + int(d[:2])
        mm = int(d[1:])

    if mm == 1:
        return date(yy - 1, 12, 31)
    else:
        last_day = calendar.monthrange(yy, mm - 1)[1]
        return date(yy, mm - 1, last_day)


def download_contract_data(
    contract: str,
    resolution: int = 3600,
) -> pd.DataFrame | None:
    """Download hourly kline data for one contract, cropped at delivery cutoff.

    Returns DataFrame indexed by datetime (Beijing time) with columns:
    [open, high, low, close, volume, open_oi]
    """
    safe_name = contract.replace(".", "_")
    cache_path = CACHE_DIR / f"{safe_name}_{resolution}.parquet"

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    creds = get_credentials()
    cutoff = _delivery_month_last_day(contract)

    api = TqApi(auth=TqAuth(creds["username"], creds["password"]))
    try:
        klines = api.get_kline_serial(contract, resolution, data_length=5000)
    finally:
        api.close()

    if klines is None or len(klines) == 0:
        return None

    df = klines.copy()
    df["datetime"] = (
        pd.to_datetime(df["datetime"], unit="ns", utc=True)
        .dt.tz_convert("Asia/Shanghai")
    )
    df = df.sort_values("datetime").set_index("datetime")

    # Remove placeholders (bars with zero volume before actual listing)
    df = df[df["volume"] > 0]

    if len(df) == 0:
        return None

    # Keep standard columns
    cols = ["open", "high", "low", "close", "volume", "open_oi"]
    df = df[[c for c in cols if c in df.columns]]

    # Crop at delivery cutoff
    cutoff_dt = datetime(
        cutoff.year, cutoff.month, cutoff.day, 23, 59, 59,
        tzinfo=timezone(timedelta(hours=8)),
    )
    df = df[df.index <= cutoff_dt]

    if len(df) == 0:
        return None

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    return df


def download_all_contracts(
    periods: list[ContractPeriod],
    resolution: int = 3600,
) -> dict[str, pd.DataFrame]:
    """Download data for all unique contracts."""
    unique_contracts = sorted(set(p.contract for p in periods))
    result: dict[str, pd.DataFrame] = {}

    for c in unique_contracts:
        df = download_contract_data(c, resolution)
        if df is not None:
            result[c] = df
            print(
                f"[DATA] {c}: {len(df)} bars, "
                f"{df.index[0].date()} -> {df.index[-1].date()}, "
                f"price {df.iloc[0]['close']:.0f} -> {df.iloc[-1]['close']:.0f}"
            )
        else:
            print(f"[DATA] {c}: NO DATA")

    return result


def clear_cache() -> None:
    for f in CACHE_DIR.glob("*.parquet"):
        f.unlink(missing_ok=True)
