#!/usr/bin/env python3
"""Multi-contract backtest runner — the entry point.

Usage:
    # Set tqsdk credentials first:
    export TQ_USERNAME="your_username"
    export TQ_PASSWORD="your_password"

    # Full run:
    python run_multi.py --variety DCE.jd --start 2022-01-01 --end 2026-06-18

    # Step-by-step:
    python run_multi.py --only-discover --variety DCE.jd
    python run_multi.py --only-download
    python run_multi.py --only-backtest
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config.strategy import (
    INITIAL_CAPITAL,
    MA_SHORT,
    RESOLUTION,
    MARGIN_RATIO,
)
from data.discovery import discover_main_contracts, ContractPeriod
from data.contract_data import download_all_contracts
from engine.multi import run_backtest, BacktestResult
from stats.metrics import calculate_metrics
from stats.plot import plot_equity_curve


RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Schedule persistence
# ---------------------------------------------------------------------------

SCHEDULE_DIR = Path(__file__).resolve().parent / "data"


def _schedule_path(variety: str) -> Path:
    safe = variety.replace(".", "_").replace("@", "_")
    return SCHEDULE_DIR / f"{safe}_schedule.csv"


def save_schedule_csv(
    periods: list[ContractPeriod], variety: str = "DCE.jd"
) -> None:
    """Save contract periods to CSV for reuse."""
    path = _schedule_path(variety)
    rows = []
    for p in periods:
        rows.append(
            {
                "start": p.start.isoformat(),
                "end": p.end.isoformat(),
                "contract": p.contract,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[OUT] Schedule -> {path}")


def load_schedule_csv(variety: str = "DCE.jd") -> list[ContractPeriod] | None:
    """Load contract periods from previously saved CSV."""
    path = _schedule_path(variety)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    periods = []
    for _, row in df.iterrows():
        periods.append(
            ContractPeriod(
                start=datetime.fromisoformat(row["start"]),
                end=datetime.fromisoformat(row["end"]),
                contract=row["contract"],
            )
        )
    return periods


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_complete_backtest(
    variety: str = "DCE.jd",
    start_date: str = "2022-01-01",
    end_date: str = "2026-06-18",
    short_ma: int = MA_SHORT,
    long_ma: int = 75,
    initial_capital: float = INITIAL_CAPITAL,
    margin_ratio: float = MARGIN_RATIO,
) -> None:
    """Discover -> download -> backtest -> output."""
    print("=" * 60)
    print(f"Multi-Contract Backtest: {variety}")
    print(f"Period: {start_date} to {end_date}")
    print(f"MA{short_ma}/MA{long_ma}, capital={initial_capital:.0f}")
    print("=" * 60)

    # Step 1: Discover
    print("\n[1/4] Discovering contract schedule...")
    sd = datetime.strptime(start_date, "%Y-%m-%d").date()
    ed = datetime.strptime(end_date, "%Y-%m-%d").date()
    periods = discover_main_contracts(variety, sd.year, ed.year)
    save_schedule_csv(periods, variety)

    # Step 2: Download
    print(f"\n[2/4] Downloading data for {len(periods)} periods...")
    contract_data = download_all_contracts(periods, RESOLUTION)
    print(f"[2/4] Got data for {len(contract_data)} contracts")

    if len(contract_data) == 0:
        print("[ERROR] No contract data downloaded. Check credentials.")
        return

    # Step 3: Backtest
    print(f"\n[3/4] Running backtest...")
    result = run_backtest(
        periods=periods,
        contract_data=contract_data,
        initial_capital=initial_capital,
        short_ma=short_ma,
        long_ma=long_ma,
        margin_ratio=margin_ratio,
    )

    # Step 4: Output
    print(f"\n[4/4] Writing outputs...")
    _write_outputs(result, variety, start_date, end_date)

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)


def _write_outputs(
    result: BacktestResult,
    variety: str,
    period_start: str,
    period_end: str,
) -> None:
    """Write CSV trade journal, TXT metrics, and equity curve PNG."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = variety.replace(".", "_").replace("@", "_")

    # CSV
    csv_path = RESULTS_DIR / f"{safe}_trades.csv"
    result.trade_journal.to_csv(str(csv_path))
    print(f"[OUT] Trades -> {csv_path}")

    # Metrics
    metrics = calculate_metrics(
        result.equity_curve, result.trade_journal.records
    )

    metrics_lines = [
        f"====== Backtest Results: {variety} ======",
        f"Period: {period_start} to {period_end}",
        f"Initial Capital: {INITIAL_CAPITAL:,.2f}",
        f"Final Equity: {result.final_equity:,.2f}",
        f"Total Return: {metrics['total_return'] * 100:+.2f}%",
        f"Annualized Return: "
        f"{metrics.get('annualized_return', 0) * 100:+.2f}%",
        f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
        f"Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%",
        f"Win Rate: {metrics.get('win_rate', 0) * 100:.1f}%  "
        f"({metrics.get('num_wins', 0)}/{metrics.get('num_trades', 0)})",
        f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
        f"Number of Trades: {metrics.get('num_trades', 0)}",
    ]

    metrics_txt = "\n".join(metrics_lines)
    print("\n" + metrics_txt)

    txt_path = RESULTS_DIR / f"{safe}_metrics.txt"
    with open(txt_path, "w") as f:
        f.write(metrics_txt + "\n")
    print(f"[OUT] Metrics -> {txt_path}")

    # Equity curve
    try:
        png_path = RESULTS_DIR / f"{safe}_equity.png"
        plot_equity_curve(
            result.equity_curve,
            f"Equity Curve — {variety} (individual contract MA)",
            str(png_path),
        )
        print(f"[OUT] Equity -> {png_path}")
    except Exception as e:
        print(f"[WARN] Plot error: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-contract futures backtest — no KQ.m@ splicing",
    )
    parser.add_argument(
        "--variety",
        default="DCE.jd",
        help="Variety (default: DCE.jd)",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Start date YYYY-MM-DD (default: 2022-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2026-06-18",
        help="End date YYYY-MM-DD (default: 2026-06-18)",
    )
    parser.add_argument(
        "--ma-short",
        type=int,
        default=MA_SHORT,
        help=f"Short MA period (default: {MA_SHORT})",
    )
    parser.add_argument(
        "--ma-long",
        type=int,
        default=75,
        help="Long MA period (default: 75)",
    )
    parser.add_argument(
        "--only-discover",
        action="store_true",
        help="Only discover contract schedule, then exit",
    )
    parser.add_argument(
        "--only-download",
        action="store_true",
        help="Only download contract data (requires previous discovery)",
    )
    parser.add_argument(
        "--only-backtest",
        action="store_true",
        help="Only run backtest (requires previous discovery + download)",
    )
    args = parser.parse_args()

    if args.only_discover:
        sd = datetime.strptime(args.start, "%Y-%m-%d").date()
        ed = datetime.strptime(args.end, "%Y-%m-%d").date()
        periods = discover_main_contracts(
            args.variety, sd.year, ed.year
        )
        save_schedule_csv(periods, args.variety)
        return

    if args.only_download:
        periods = load_schedule_csv(args.variety)
        if periods is None:
            print(
                "[ERROR] No schedule found. Run --only-discover first."
            )
            sys.exit(1)
        contract_data = download_all_contracts(periods, RESOLUTION)
        print(f"Downloaded {len(contract_data)} contracts")
        return

    if args.only_backtest:
        periods = load_schedule_csv(args.variety)
        if periods is None:
            print(
                "[ERROR] No schedule found. Run --only-discover first."
            )
            sys.exit(1)

        # Load cached contract data
        from pathlib import Path as _P

        cache_dir = (
            _P(__file__).resolve().parent / "data" / "contract_cache"
        )
        contract_data = {}
        for p in periods:
            safe = p.contract.replace(".", "_")
            path = cache_dir / f"{safe}_{RESOLUTION}.parquet"
            if path.exists():
                contract_data[p.contract] = pd.read_parquet(path)

        if len(contract_data) == 0:
            print(
                "[ERROR] No cached data found. Run --only-download first."
            )
            sys.exit(1)

        result = run_backtest(
            periods=periods,
            contract_data=contract_data,
            initial_capital=INITIAL_CAPITAL,
            short_ma=args.ma_short,
            long_ma=args.ma_long,
        )
        _write_outputs(result, args.variety, args.start, args.end)
        return

    # Default: full pipeline
    run_complete_backtest(
        variety=args.variety,
        start_date=args.start,
        end_date=args.end,
        short_ma=args.ma_short,
        long_ma=args.ma_long,
    )


if __name__ == "__main__":
    main()
