#!/usr/bin/env python3
"""Run multi-product portfolio backtest.

Usage: python3 run_portfolio.py [product1,product2,...]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.strategy import INITIAL_CAPITAL
from data.discovery import discover_main_contracts
from data.contract_data import download_all_contracts
from engine.portfolio import build_product_series, run_portfolio
from engine.multi_v3 import detect_session_type
from gen_report import compute_advanced_metrics, generate_equity_chart, generate_html_report


def main():
    if len(sys.argv) > 1:
        products = [p.strip() for p in sys.argv[1].split(",")]
    else:
        products = ["DCE.jd", "DCE.p", "DCE.m", "SHFE.rb", "CZCE.MA"]

    print(f"\n{'='*60}")
    print(f" Portfolio Backtest: {len(products)} products")
    print(f"{'='*60}\n")

    # Step 1: for each product, discover + download + build series
    product_series: dict[str, pd.DataFrame] = {}
    for i, prod in enumerate(products):
        print(f"[{i+1}/{len(products)}] {prod}")
        print(f"  Discovering main contracts...")
        periods = discover_main_contracts(prod, 2022, 2026, verbose=False)
        if not periods:
            print(f"  WARNING: No periods found, skipping")
            continue

        print(f"  {len(periods)} periods, downloading data...")
        contract_data = download_all_contracts(periods)
        if not contract_data:
            print(f"  WARNING: No contract data, skipping")
            continue

        print(f"  Building signal series...")
        series = build_product_series(prod, periods, contract_data)
        if series is None or len(series) == 0:
            print(f"  WARNING: Empty series, skipping")
            continue

        ma_long = detect_session_type(contract_data[list(contract_data.keys())[0]])
        print(f"  {len(series)} bars, MA_long={ma_long}")
        product_series[prod] = series

    if not product_series:
        print("ERROR: No valid product series")
        return

    print(f"\n{'='*60}")
    print(f" Running portfolio with {len(product_series)} products")
    print(f"{'='*60}\n")

    result = run_portfolio(
        product_series=product_series,
        product_order=list(product_series.keys()),
        initial_capital=INITIAL_CAPITAL,
    )

    # Collect all trade records for reporting
    all_trades = []
    for prod, journal in result.journals.items():
        for rec in journal.records:
            rec["product"] = prod
            all_trades.append(rec)

    print(f"\n=== Portfolio Results ===")
    print(f"Final Equity: ¥{result.final_equity:,.0f}")
    print(f"Total Return: {result.total_return*100:+.2f}%")
    print(f"Total Trades: {len(all_trades)}")
    print(f"Max Concurrent Positions: {max(result.positions_held) if result.positions_held else 0}")

    # Generate report
    metrics = compute_advanced_metrics(
        result.equity_curve, result.equity_dates,
        [t for t in all_trades if "close" in t.get("action", "")],
        INITIAL_CAPITAL,
    )
    chart = generate_equity_chart(
        result.equity_curve, result.equity_dates,
        metrics["max_drawdown_start_idx"],
        metrics["max_drawdown_end_idx"],
    )

    # Portfolio HTML
    html = portfolio_html_report(metrics, all_trades, result, chart)

    out = Path(__file__).resolve().parent / "results" / "portfolio_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nReport: {out}")

    # Save per-product trade CSVs
    out_dir = Path(__file__).resolve().parent / "results" / "portfolio_trades"
    out_dir.mkdir(parents=True, exist_ok=True)
    for prod, journal in result.journals.items():
        if journal.records:
            jf = out_dir / f"{prod.replace('.','_')}_trades.csv"
            journal.to_csv(str(jf))

    # Portfolio equity CSV
    eq_df = pd.DataFrame({
        "datetime": result.equity_dates,
        "equity": result.equity_curve,
        "positions": result.positions_held,
    })
    eq_df.to_csv(out_dir.parent / "portfolio_equity.csv", index=False)
    print(f"Equity CSV: {out_dir.parent / 'portfolio_equity.csv'}")


def portfolio_html_report(
    metrics: dict,
    trades: list[dict],
    result,
    chart_b64: str,
) -> str:
    """Generate portfolio HTML report."""
    from gen_report import _fmt, _fmt_pct

    def card(label, *values):
        val = " ".join(str(v) for v in values)
        return f'<div class="mc"><div class="mv">{val}</div><div class="mt">{label}</div></div>'

    eq_curve = result.equity_curve
    initial = eq_curve[0] if eq_curve else 1
    final = eq_curve[-1] if eq_curve else 1
    total_ret = (final / initial) - 1

    # Trade breakdown by product
    by_product: dict[str, list] = {}
    for t in trades:
        p = t.get("product", "?")
        by_product.setdefault(p, []).append(t)

    prod_rows = ""
    for prod, ptrades in sorted(by_product.items()):
        close_t = [t for t in ptrades if "close" in t.get("action", "")]
        pnls = [t.get("pnl", 0) for t in close_t]
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        prod_rows += f"<tr><td>{prod}</td><td>{len(close_t)}</td><td>{wins}</td>"
        prod_rows += f"<td class=\"{'pos' if total_pnl>0 else 'neg'}\">{_fmt(total_pnl,0)}</td>"
        win_rate = wins / len(close_t) * 100 if close_t else 0
        prod_rows += f"<td>{win_rate:.0f}%</td></tr>"

    # Trade detail rows
    close_trades = [t for t in trades if "close" in t.get("action", "")]
    trade_rows = ""
    for i, t in enumerate(close_trades):
        cls = "pos" if t.get("pnl", 0) > 0 else "neg"
        trade_rows += (
            f"<tr><td>{i+1}</td><td>{t.get('product','')}</td>"
            f"<td>{t.get('bar_index','')}</td>"
            f"<td>{t.get('action','')}</td>"
            f"<td>{_fmt(t.get('price',0),1)}</td>"
            f"<td>{_fmt(t.get('size',0),0)}</td>"
            f"<td class=\"{cls}\">{_fmt(t.get('pnl',0),0)}</td>"
            f"<td>{t.get('reason','')}</td></tr>"
        )

    t1 = card("总收益", _fmt_pct(total_ret))
    t1 += card("最终权益", f"¥{_fmt(final,0)}")
    t1 += card("总交易", str(len(close_trades)))
    t1 += card("品种数", str(len(by_product)))

    t2 = card("年化收益", _fmt_pct(metrics.get("annualized_return_pct", 0)))
    t2 += card("夏普比率", _fmt(metrics.get("sharpe_ratio", 0), 2))
    t2 += card("最大回撤", _fmt_pct(-metrics.get("max_drawdown_pct", 0)))
    t2 += card("盈利因子", _fmt(metrics.get("profit_factor", 0), 2))

    avg_pos = sum(result.positions_held) / len(result.positions_held) if result.positions_held else 0
    t3 = card("平均持仓品种", _fmt(avg_pos, 1))
    t3 += card("最多持仓品种", str(max(result.positions_held) if result.positions_held else 0))
    t3 += card("回撤天数", str(metrics.get("max_drawdown_duration_days", 0)))

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Backtest Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;background:#0f0f23;color:#e0e0e0}}
.hdr{{background:linear-gradient(135deg,#1a1a3e,#16213e);padding:30px 40px;border-bottom:1px solid #2a2a5e}}
.hdr h1{{font-size:24px;color:#fff}}
.hdr .sub{{font-size:13px;color:#8888aa;margin-top:6px}}
.ct{{max-width:1280px;margin:0 auto;padding:24px 20px}}
.mg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:24px}}
.mc{{background:#1a1a3e;border:1px solid #2a2a5e;border-radius:10px;padding:16px;text-align:center}}
.mv{{font-size:20px;font-weight:700;color:#fff}}
.mt{{font-size:12px;color:#8888aa;text-transform:uppercase;letter-spacing:.5px}}
.sec{{background:#1a1a3e;border:1px solid #2a2a5e;border-radius:10px;padding:20px;margin-bottom:24px}}
.sec h2{{font-size:16px;color:#fff;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #2a2a5e}}
.cc img{{max-width:100%;border-radius:6px}}
.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl th{{background:#252550;color:#aaaacc;padding:8px 10px;text-align:left;font-weight:500;position:sticky;top:0}}
.tbl td{{padding:6px 10px;border-bottom:1px solid #252550;font-family:"SF Mono",monospace}}
.pos{{color:#4ecdc4;font-weight:600}}
.neg{{color:#ff6b6b;font-weight:600}}
.scroll{{max-height:480px;overflow-y:auto}}
.ft{{padding:20px;text-align:center;font-size:12px;color:#555577}}
</style></head><body>
<div class="hdr"><h1>Portfolio Multi-Product Backtest</h1>
<div class="sub">{len(by_product)} products · 10% margin/product · CAP 100% total · 2022-01 → 2026-06</div></div>
<div class="ct">
<div class="mg">{t1}</div>
<div class="sec"><h2>风险调整收益</h2><div class="mg">{t2}</div></div>
<div class="sec"><h2>持仓统计</h2><div class="mg">{t3}</div></div>
<div class="sec"><h2>权益曲线</h2><div class="cc"><img src="data:image/png;base64,{chart_b64}"></div></div>
<div class="sec"><h2>品种表现</h2>
<div class="scroll"><table class="tbl"><thead><tr><th>品种</th><th>交易</th><th>胜</th><th>净盈亏</th><th>胜率</th></tr></thead><tbody>{prod_rows}</tbody></table></div></div>
<div class="sec"><h2>交易明细</h2>
<div class="scroll"><table class="tbl"><thead><tr><th>#</th><th>品种</th><th>Bar</th><th>操作</th><th>价格</th><th>手数</th><th>盈亏</th><th>原因</th></tr></thead><tbody>{trade_rows}</tbody></table></div></div>
</div>
<div class="ft">Generated by testforjd portfolio · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""


if __name__ == "__main__":
    main()
