#!/usr/bin/env python3
"""Generate a comprehensive HTML backtest report.

Reads the multi-contract backtest results and produces a self-contained
HTML page with equity curve, trade journal, and full metric table.
"""

from __future__ import annotations

import base64
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.strategy import INITIAL_CAPITAL
from data.discovery import discover_main_contracts
from data.contract_data import download_all_contracts
from engine.multi import run_backtest
from stats.metrics import calculate_metrics


# ---------------------------------------------------------------------------
# Advanced metrics
# ---------------------------------------------------------------------------


def compute_advanced_metrics(
    equity_curve: list[float],
    trade_records: list[dict],
) -> dict:
    """Compute a comprehensive set of performance metrics."""
    eq = pd.Series(equity_curve)
    initial = eq.iloc[0] if len(eq) > 0 else 1.0
    final = eq.iloc[-1] if len(eq) > 0 else 1.0
    total_return = (final / initial) - 1.0
    n_bars = len(eq)

    # Basic metrics via existing module
    basic = calculate_metrics(equity_curve, trade_records)

    # Annualized metrics
    # JD: ~7 bars/day * ~252 trading days = ~1764 bars/year (roughly)
    bars_per_year = 1764
    if n_bars > 0 and total_return > -1.0:
        ann_return = (1.0 + total_return) ** (bars_per_year / n_bars) - 1.0
    else:
        ann_return = 0.0

    # Return series (bar-based, no datetime index)
    returns = eq.pct_change().dropna()

    # Approximate daily returns by grouping every ~7 bars (JD has ~7 bars/day)
    daily_returns = returns.groupby(returns.index // 7).apply(
        lambda x: (1 + x).prod() - 1
    )

    # Sortino ratio (downside deviation)
    rf_daily = 0.02 / 365
    excess_daily = daily_returns - rf_daily
    downside = excess_daily[excess_daily < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-10
    sortino = (excess_daily.mean() / downside_std) * math.sqrt(365) if downside_std > 0 else 0.0

    # Max drawdown analysis
    peak = eq.expanding().max()
    drawdown = (eq - peak) / peak
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Max drawdown period (start and end)
    dd_series = pd.Series(drawdown.values, index=range(len(drawdown)))
    max_dd_idx = dd_series.idxmin() if len(dd_series) > 0 else 0
    # Find peak before max_dd
    peak_before = eq.iloc[:max_dd_idx + 1].max()
    peak_before_idx = eq.iloc[:max_dd_idx + 1].idxmax()
    # Find recovery (when equity returns to peak)
    after = eq.iloc[max_dd_idx:]
    recovery_idx = after[after >= peak_before].index
    dd_end_idx = recovery_idx[0] if len(recovery_idx) > 0 else len(eq) - 1
    dd_duration = dd_end_idx - peak_before_idx

    # Calmar ratio
    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    # Trade analysis
    close_trades = [t for t in trade_records if "close" in t.get("action", "")]
    pnls = [t.get("pnl", 0) for t in close_trades]
    num_trades = len(pnls)

    avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0.0
    avg_loss = abs(np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0.0
    largest_win = max(pnls) if pnls else 0.0
    largest_loss = min(pnls) if pnls else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # Consecutive wins/losses
    wins = [1 if p > 0 else 0 for p in pnls]
    max_consec_wins = 0
    max_consec_losses = 0
    cur = 0
    for w in wins:
        if w == 1:
            cur += 1
            max_consec_wins = max(max_consec_wins, cur)
        else:
            cur = 0
    cur = 0
    for w in wins:
        if w == 0:
            cur += 1
            max_consec_losses = max(max_consec_losses, cur)
        else:
            cur = 0

    # Average trade P&L
    avg_trade_pnl = np.mean(pnls) if pnls else 0.0

    # Recovery factor (total P&L / max drawdown $)
    total_pnl = sum(p for p in pnls)
    initial_eq = eq.iloc[0] if len(eq) > 0 else 1.0
    dd_dollars = initial_eq * max_dd if max_dd > 0 else 1.0
    recovery_factor = abs(total_pnl / dd_dollars) if dd_dollars > 0 else 0.0

    # Gross profit / loss
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    return {
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "annualized_return": ann_return,
        "annualized_return_pct": ann_return * 100,
        "sharpe_ratio": basic.get("sharpe_ratio", 0),
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100,
        "max_drawdown_start_idx": peak_before_idx,
        "max_drawdown_end_idx": dd_end_idx,
        "max_drawdown_duration_bars": int(dd_duration),
        "win_rate": basic.get("win_rate", 0),
        "win_rate_pct": basic.get("win_rate", 0) * 100,
        "num_trades": num_trades,
        "num_wins": basic.get("num_wins", 0),
        "profit_factor": basic.get("profit_factor", 0),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "win_loss_ratio": win_loss_ratio,
        "avg_trade_pnl": avg_trade_pnl,
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "recovery_factor": recovery_factor,
        "initial_capital": float(initial),
        "final_equity": float(final),
        "total_pnl": total_pnl,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _fmt(v, decimals=2, prefix="", suffix=""):
    """Format a number for display."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if isinstance(v, float):
        return f"{prefix}{v:,.{decimals}f}{suffix}"
    return f"{prefix}{v}{suffix}"


def _fmt_pct(v):
    """Format as percentage."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.2f}%"


def generate_equity_chart(
    equity_curve: list[float],
    drawdown_start: int,
    drawdown_end: int,
) -> str:
    """Generate an enhanced equity curve chart with max drawdown highlighted.
    Returns base64-encoded PNG.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    eq = pd.Series(equity_curve)
    eq_pct = (eq / eq.iloc[0] - 1) * 100

    # Upper: equity curve
    ax1.fill_between(range(len(eq)), eq_pct, alpha=0.15, color="#00d2ff")
    ax1.plot(range(len(eq)), eq_pct, color="#00d2ff", linewidth=1.2, label="Equity")

    # Highlight max drawdown
    if drawdown_start >= 0 and drawdown_end > drawdown_start:
        ax1.axvspan(drawdown_start, drawdown_end, alpha=0.2, color="#ff4444")
        mid = (drawdown_start + drawdown_end) // 2
        ax1.annotate("Max DD", (mid, eq_pct.iloc[drawdown_end] + 1),
                     fontsize=9, color="#ff4444", ha="center",
                     arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1))

    # Peak line
    peak = eq_pct.expanding().max()
    ax1.plot(range(len(peak)), peak, color="#ffd700", linewidth=0.8,
             linestyle="--", alpha=0.6, label="Peak")

    ax1.set_facecolor("#16213e")
    ax1.grid(True, alpha=0.1)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylabel("Return (%)", color="white")
    ax1.tick_params(colors="white")
    ax1.set_title("Equity Curve — JD Multi-Contract MA20/MA75",
                  color="white", fontsize=13, fontweight="bold")

    # Lower: drawdown
    drawdown_pct = (eq / peak - 1) * 100
    ax2.fill_between(range(len(drawdown_pct)), 0, drawdown_pct,
                     color="#ff6b6b", alpha=0.7)
    ax2.set_facecolor("#16213e")
    ax2.grid(True, alpha=0.1)
    ax2.set_ylabel("Drawdown (%)", color="white")
    ax2.tick_params(colors="white")
    ax2.set_xlabel("Bar", color="white")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_html_report(
    metrics: dict,
    trades: list[dict],
    equity_chart_b64: str,
) -> str:
    """Generate a self-contained HTML report page."""
    # Trade table rows
    trade_rows = ""
    for i, t in enumerate(trades):
        action = t.get("action", "")
        price = t.get("price", 0)
        size = t.get("size", 0)
        pnl = t.get("pnl", 0)
        reason = t.get("reason", "")

        pnl_class = "positive" if pnl > 0 else ("negative" if pnl < 0 else "")

        trade_rows += f"""<tr>
            <td>{i + 1}</td>
            <td>{action}</td>
            <td>{_fmt(price)}</td>
            <td>{_fmt(size, 0)}</td>
            <td class="{pnl_class}">{_fmt(pnl)}</td>
            <td>{reason}</td>
        </tr>\n"""

    # Key metrics columns
    def metric_card(title, value, subtitle="", color=""):
        color_style = f"color: {color};" if color else ""
        return f"""
        <div class="metric-card">
            <div class="metric-value" style="{color_style}">{value}</div>
            <div class="metric-title">{title}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>"""

    # Tier 1: Core returns
    tier1 = ""
    tier1 += metric_card("总收益", _fmt_pct(metrics["total_return_pct"]),
                         f"年化 {_fmt_pct(metrics['annualized_return_pct'])}",
                         color="#4ecdc4" if metrics["total_return"] > 0 else "#ff6b6b")
    tier1 += metric_card("最终权益", _fmt(metrics["final_equity"], 0, prefix="¥"),
                         f"初始 ¥{_fmt(metrics['initial_capital'], 0)}")
    tier1 += metric_card("交易总盈亏", _fmt(metrics["total_pnl"], 0, prefix="¥"),
                         f"{metrics['num_trades']} 笔交易")
    tier1 += metric_card("胜率", _fmt_pct(metrics["win_rate_pct"]),
                         f"{metrics['num_wins']} 胜 / {metrics['num_trades']} 总")

    # Tier 2: Risk
    tier2 = ""
    tier2 += metric_card("夏普比率", _fmt(metrics["sharpe_ratio"], 2),
                         color="#4ecdc4" if metrics["sharpe_ratio"] > 0 else "#ff6b6b")
    tier2 += metric_card("索提诺比率", _fmt(metrics["sortino_ratio"], 2))
    tier2 += metric_card("卡尔玛比率", _fmt(metrics["calmar_ratio"], 2))
    tier2 += metric_card("盈亏比", _fmt(metrics["win_loss_ratio"], 2),
                         f"平均赢 ¥{_fmt(metrics['avg_win'])} / 输 ¥{_fmt(metrics['avg_loss'])}")

    # Tier 3: Drawdown
    tier3 = ""
    tier3 += metric_card("最大回撤", _fmt_pct(metrics["max_drawdown_pct"]))
    tier3 += metric_card("最大回撤时长",
                         f"{metrics['max_drawdown_duration_bars']} 根K线")
    tier3 += metric_card("恢复因子", _fmt(metrics["recovery_factor"], 2))
    tier3 += metric_card("最大连续亏损",
                         f"{metrics['max_consecutive_losses']} 笔",
                         f"最大连续盈利 {metrics['max_consecutive_wins']} 笔")

    # Tier 4: Profit
    tier4 = ""
    tier4 += metric_card("毛利", _fmt(metrics["gross_profit"], 0, prefix="¥"))
    tier4 += metric_card("毛损", _fmt(metrics["gross_loss"], 0, prefix="¥"))
    tier4 += metric_card("盈利因子", _fmt(metrics["profit_factor"], 2))
    tier4 += metric_card("最大单笔盈利",
                         _fmt(metrics["largest_win"], 0, prefix="¥"),
                         f"最大单笔亏损 ¥{_fmt(abs(metrics['largest_loss']), 0)}")

    # Profit distribution bar
    pnls = [t.get("pnl", 0) for t in trades if "close" in t.get("action", "")]
    hist_data = _histogram_data(pnls)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JD Multi-Contract Backtest Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei",
                 sans-serif;
    background: #0f0f23;
    color: #e0e0e0;
    min-height: 100vh;
}}
.header {{
    background: linear-gradient(135deg, #1a1a3e 0%, #16213e 100%);
    padding: 30px 40px;
    border-bottom: 1px solid #2a2a5e;
}}
.header h1 {{
    font-size: 24px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 6px;
}}
.header .subtitle {{
    font-size: 14px;
    color: #8888aa;
}}
.content {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 24px 20px;
}}
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 28px;
}}
.metric-card {{
    background: #1a1a3e;
    border: 1px solid #2a2a5e;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    transition: border-color 0.2s;
}}
.metric-card:hover {{ border-color: #4ecdc4; }}
.metric-value {{
    font-size: 20px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
}}
.metric-title {{
    font-size: 12px;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.metric-subtitle {{
    font-size: 11px;
    color: #666688;
    margin-top: 4px;
}}
.section {{
    background: #1a1a3e;
    border: 1px solid #2a2a5e;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
}}
.section h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a5e;
}}
.chart-container {{
    text-align: center;
}}
.chart-container img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
}}
.trade-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
.trade-table th {{
    background: #252550;
    color: #aaaacc;
    padding: 8px 10px;
    text-align: left;
    font-weight: 500;
    position: sticky;
    top: 0;
}}
.trade-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid #252550;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
}}
.trade-table tr:hover {{ background: #252550; }}
.positive {{ color: #4ecdc4; font-weight: 600; }}
.negative {{ color: #ff6b6b; font-weight: 600; }}
.trade-scroll {{
    max-height: 480px;
    overflow-y: auto;
}}
.trade-scroll::-webkit-scrollbar {{
    width: 6px;
}}
.trade-scroll::-webkit-scrollbar-track {{
    background: #1a1a3e;
}}
.trade-scroll::-webkit-scrollbar-thumb {{
    background: #3a3a6e;
    border-radius: 3px;
}}
.section-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.badge {{
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 12px;
    background: #252550;
    color: #8888aa;
}}
.footer {{
    text-align: center;
    padding: 20px;
    color: #555577;
    font-size: 12px;
}}
@media (max-width: 600px) {{
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .header {{ padding: 20px; }}
    .header h1 {{ font-size: 18px; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>📊 JD Multi-Contract Backtest Report</h1>
    <div class="subtitle">
        MA20/MA75 三区策略 · 25个独立JD合约 · 无KQ.m@拼接 ·
        2022-01 至 2026-06 · 5385小时K线
    </div>
</div>

<div class="content">

    <!-- Summary -->
    <div class="metrics-grid">
        {tier1}
    </div>

    <!-- Risk Metrics -->
    <div class="section">
        <h2>📈 风险调整收益</h2>
        <div class="metrics-grid">
            {tier2}
        </div>
    </div>

    <!-- Drawdown -->
    <div class="section">
        <h2>⚠️ 回撤分析</h2>
        <div class="metrics-grid">
            {tier3}
        </div>
    </div>

    <!-- Profit Analysis -->
    <div class="section">
        <h2>💰 盈亏分析</h2>
        <div class="metrics-grid">
            {tier4}
        </div>
    </div>

    <!-- Equity Curve -->
    <div class="section">
        <div class="section-title-row">
            <h2>📉 权益曲线</h2>
            <span class="badge">红色阴影 = 最大回撤区间</span>
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{equity_chart_b64}" alt="Equity Curve">
        </div>
    </div>

    <!-- Profit Histogram -->
    <div class="section">
        <h2>📊 盈亏分布</h2>
        <div class="chart-container" id="histogram-chart">
            <div style="height:200px;display:flex;align-items:flex-end;gap:4px;padding:20px 10px;">
                {hist_data}
            </div>
        </div>
    </div>

    <!-- Trade Journal -->
    <div class="section">
        <div class="section-title-row">
            <h2>📋 交易明细</h2>
            <span class="badge">{len(trades)} 条记录</span>
        </div>
        <div class="trade-scroll">
            <table class="trade-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>操作</th>
                        <th>价格</th>
                        <th>手数</th>
                        <th>盈亏 (¥)</th>
                        <th>原因</th>
                    </tr>
                </thead>
                <tbody>
                    {trade_rows}
                </tbody>
            </table>
        </div>
    </div>

</div>

<div class="footer">
    Generated by testforjd · {datetime.now().strftime("%Y-%m-%d %H:%M")} · Public repo: github.com/tgmnb/testforjd
</div>

</body>
</html>"""

    return html


def _histogram_data(pnls: list[float], bins: int = 20) -> str:
    """Generate a simple bar chart for P&L distribution using pure CSS."""
    if not pnls:
        return "<div style='color:#666;text-align:center'>No trade data</div>"

    pnl_arr = np.array(pnls)
    counts, edges = np.histogram(pnl_arr, bins=bins)
    max_count = max(counts) if len(counts) > 0 else 1

    bars = ""
    for i in range(len(counts)):
        pct = counts[i] / max_count * 100
        left = edges[i]
        right = edges[i + 1]
        mid = (left + right) / 2
        color = "#4ecdc4" if mid >= 0 else "#ff6b6b"
        bar_label = f"{counts[i]}" if counts[i] > 0 else ""
        bars += f"""<div style="flex:1;display:flex;flex-direction:column;align-items:center;">
            <div style="width:90%;height:{max(pct, 2)}%;background:{color};border-radius:3px 3px 0 0;opacity:0.8;" title="{left:.0f} ~ {right:.0f}: {counts[i]}笔"></div>
            <span style="font-size:9px;color:#666;margin-top:2px;">{bar_label}</span>
        </div>"""

    return bars


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full pipeline and generate HTML report."""
    print("[REPORT] Running backtest pipeline...")

    # Re-run backtest (cached data will be reused from parquet)
    periods = discover_main_contracts("DCE.jd", 2022, 2026)
    contract_data = download_all_contracts(periods)
    result = run_backtest(
        periods=periods,
        contract_data=contract_data,
        initial_capital=INITIAL_CAPITAL,
        short_ma=20,
        long_ma=75,
    )

    print("[REPORT] Computing advanced metrics...")
    trade_records = result.trade_journal.records
    metrics = compute_advanced_metrics(result.equity_curve, trade_records)

    print("[REPORT] Generating equity chart...")
    chart_b64 = generate_equity_chart(
        result.equity_curve,
        metrics["max_drawdown_start_idx"],
        metrics["max_drawdown_end_idx"],
    )

    print("[REPORT] Generating HTML report...")
    html = generate_html_report(metrics, trade_records, chart_b64)

    output_path = Path(__file__).resolve().parent / "results" / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"[REPORT] Report saved -> {output_path}")
    print(f"[REPORT] {len(trade_records)} trades, "
          f"{metrics['total_return_pct']:+.2f}% return, "
          f"{metrics['num_trades']} trades")


if __name__ == "__main__":
    main()
