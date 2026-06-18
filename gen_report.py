#!/usr/bin/env python3
"""Generate comprehensive HTML backtest report — v3 engine support.

Fixed IndexError on date-indexed drawdown computation.
"""

from __future__ import annotations

import base64
import math
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.strategy import INITIAL_CAPITAL
from stats.metrics import calculate_metrics


def _to_series(equity_curve: list[float], equity_dates: list | None):
    """Build a Series with date index if dates provided, else bar index."""
    s = pd.Series(equity_curve, dtype=float)
    if equity_dates and len(equity_dates) == len(s):
        s.index = pd.DatetimeIndex(equity_dates)
    return s


def compute_advanced_metrics(
    equity_curve: list[float],
    equity_dates: list[datetime] | None,
    trade_records: list[dict],
    initial_capital: float,
) -> dict:
    """Compute comprehensive metrics with optional date index."""
    eq = _to_series(equity_curve, equity_dates)
    n = len(eq)
    initial = eq.iloc[0] if n > 0 else 1.0
    final = eq.iloc[-1] if n > 0 else 1.0
    total_return = (final / initial) - 1.0

    # Daily returns
    if hasattr(eq.index, "date") and isinstance(eq.index, pd.DatetimeIndex):
        daily_eq = eq.resample("D").last().dropna()
    else:
        # Approximate: every ~7 bars = 1 day
        daily_eq = eq.groupby(eq.index // 7).last()

    daily_returns = daily_eq.pct_change().dropna()

    if len(daily_returns) > 0:
        ann_return = (
            (1 + total_return) ** (365.0 / len(daily_returns)) - 1.0
        ) if total_return > -1.0 else 0.0
    else:
        ann_return = 0.0

    # Sharpe
    rf_daily = 0.02 / 365
    excess_daily = daily_returns - rf_daily
    sharpe = 0.0
    if len(excess_daily) > 1 and excess_daily.std() > 0:
        sharpe = (excess_daily.mean() / excess_daily.std()) * math.sqrt(365)

    # Sortino
    downside = excess_daily[excess_daily < 0]
    sortino = 0.0
    if len(downside) > 1 and downside.std() > 0:
        sortino = (excess_daily.mean() / downside.std()) * math.sqrt(365)

    # Max drawdown (integer-indexed to avoid DatetimeIndex issues)
    idx_peak = eq.expanding().max()
    idx_dd_series = (eq - idx_peak) / idx_peak
    max_dd = abs(idx_dd_series.min()) if len(idx_dd_series) > 0 else 0.0

    # Drawdown period (by integer position)
    dd_vals = idx_dd_series.values
    max_dd_pos = int(np.argmin(dd_vals)) if len(dd_vals) > 0 else 0
    peak_before_pos = int(np.argmax(eq.values[:max_dd_pos + 1])) if max_dd_pos > 0 else 0
    peak_val = eq.values[peak_before_pos]
    after = eq.values[max_dd_pos:]
    recovery_positions = np.where(after >= peak_val)[0]
    dd_end_pos = max_dd_pos + recovery_positions[0] if len(recovery_positions) > 0 else n - 1

    dd_duration_bars = dd_end_pos - peak_before_pos
    dd_duration_days = int(dd_duration_bars / 7)  # ~7 bars/day

    # Calmar
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

    wins = [1 if p > 0 else 0 for p in pnls]
    max_cw = max_cl = cur = 0
    for w in wins:
        if w == 1:
            cur += 1; max_cw = max(max_cw, cur)
        else:
            cur = 0
    cur = 0
    for w in wins:
        if w == 0:
            cur += 1; max_cl = max(max_cl, cur)
        else:
            cur = 0

    avg_trade_pnl = np.mean(pnls) if pnls else 0.0
    total_pnl = sum(p for p in pnls)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    basic = calculate_metrics(equity_curve, trade_records)
    dd_dollars = initial * max_dd if max_dd > 0 else 1.0
    recovery_factor = abs(total_pnl / dd_dollars) if dd_dollars > 0 else 0.0

    return {
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "annualized_return": ann_return,
        "annualized_return_pct": ann_return * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100,
        "max_drawdown_start_idx": peak_before_pos,
        "max_drawdown_end_idx": dd_end_pos,
        "max_drawdown_duration_bars": int(dd_duration_bars),
        "max_drawdown_duration_days": dd_duration_days,
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
        "max_consecutive_wins": max_cw,
        "max_consecutive_losses": max_cl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "recovery_factor": recovery_factor,
        "initial_capital": float(initial),
        "final_equity": float(final),
        "total_pnl": total_pnl,
        "n_bars": n,
    }


# ---- charts & HTML -------------------------------------------------------

def _fmt(v, decimals=2, prefix="", suffix=""):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "\u2014"
    try:
        return f"{prefix}{float(v):,.{decimals}f}{suffix}"
    except Exception:
        return str(v)


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "\u2014"
    return f"{v:+.2f}%"


def generate_equity_chart(
    equity_curve: list[float],
    equity_dates: list | None,
    dd_start: int,
    dd_end: int,
) -> str:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#1a1a2e")

    eq = pd.Series(equity_curve)
    has_dates = equity_dates and len(equity_dates) == len(eq)
    if has_dates:
        eq.index = pd.DatetimeIndex(equity_dates)

    x = eq.index if has_dates else range(len(eq))
    eq_pct = (eq / eq.iloc[0] - 1) * 100
    peak_line = eq_pct.expanding().max()

    ax1.fill_between(x, eq_pct.values, alpha=0.15, color="#00d2ff")
    ax1.plot(x, eq_pct.values, color="#00d2ff", linewidth=1.2)
    ax1.plot(x, peak_line.values, color="#ffd700", linewidth=0.8,
             linestyle="--", alpha=0.6)

    if 0 <= dd_start < dd_end < len(eq):
        sx = x[dd_start]
        ex = x[min(dd_end, len(x) - 1)]
        ax1.axvspan(sx, ex, alpha=0.2, color="#ff4444")
        mid = x[(dd_start + dd_end) // 2]
        ax1.annotate("Max DD", (mid, eq_pct.iloc[dd_end] + 1),
                     fontsize=9, color="#ff4444", ha="center",
                     arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1))

    ax1.set_facecolor("#16213e")
    ax1.grid(True, alpha=0.1)
    ax1.legend(["Equity", "Peak"], loc="upper left", fontsize=9)
    ax1.set_ylabel("Return (%)", color="white")
    ax1.tick_params(colors="white")
    ax1.set_title("JD Multi-Contract MA Backtest (v3)",
                  color="white", fontsize=13, fontweight="bold")

    dd_chg = (eq_pct / peak_line - 1) * 100
    ax2.fill_between(x, 0, dd_chg.values, color="#ff6b6b", alpha=0.7)
    ax2.set_facecolor("#16213e")
    ax2.grid(True, alpha=0.1)
    ax2.set_ylabel("Drawdown (%)", color="white")
    ax2.tick_params(colors="white")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _histogram_data(pnls, bins=20):
    if not pnls:
        return "<div>No trade data</div>"
    counts, edges = np.histogram(pnls, bins=bins)
    mx = max(counts) if len(counts) > 0 else 1
    bars = ""
    for i, c in enumerate(counts):
        pct = c / mx * 100
        mid = (edges[i] + edges[i + 1]) / 2
        color = "#4ecdc4" if mid >= 0 else "#ff6b6b"
        bars += (
            f'<div style="flex:1;display:flex;flex-direction:column;'
            f'align-items:center;">'
            f'<div style="width:90%;height:{max(pct,2)}%;'
            f'background:{color};border-radius:3px 3px 0 0;opacity:.8" '
            f'title="{edges[i]:.0f}~{edges[i+1]:.0f}:{c}笔"></div>'
            f'<span style="font-size:9px;color:#666;margin-top:2px">'
            f'{c if c > 0 else ""}</span></div>'
        )
    return bars


def generate_html_report(metrics, trades, chart_b64, version="v3"):
    trade_rows = ""
    for i, t in enumerate(trades):
        act = t.get("action", "")
        pnl = t.get("pnl", 0)
        cls = "pos" if pnl > 0 else ("neg" if pnl < 0 else "")
        ts = t.get("datetime", "")
        dt_s = pd.Timestamp(ts).strftime("%m-%d %H:%M") if ts else ""
        trade_rows += f"<tr><td>{i+1}</td><td style='font-size:11px'>{dt_s}</td><td>{act}</td><td>{_fmt(t.get('price'))}</td><td>{_fmt(t.get('size'),0)}</td><td class='{cls}'>{_fmt(pnl)}</td><td>{t.get('reason','')}</td></tr>\n"

    def card(title, value, sub="", color=""):
        sc = f"color:{color};" if color else ""
        s = f'<div class="mt-s">{sub}</div>' if sub else ""
        return f'<div class="mc"><div class="mv" style="{sc}">{value}</div><div class="mt">{title}</div>{s}</div>'

    t1 = card("总收益", _fmt_pct(metrics["total_return_pct"]), f"年化 {_fmt_pct(metrics['annualized_return_pct'])}",
              color="#4ecdc4" if metrics["total_return"] > 0 else "#ff6b6b") + \
        card("最终权益", _fmt(metrics["final_equity"], 0, prefix="¥\n"), f"初始 ¥{_fmt(metrics['initial_capital'],0)}") + \
        card("总盈亏", _fmt(metrics["total_pnl"], 0, prefix="¥\n"), f"{metrics['num_trades']}笔") + \
        card("胜率", _fmt_pct(metrics["win_rate_pct"]), f"{metrics['num_wins']}胜/{metrics['num_trades']}总")
    t2 = card("夏普", _fmt(metrics["sharpe_ratio"], 2)) + \
        card("索提诺", _fmt(metrics["sortino_ratio"], 2)) + \
        card("卡尔玛", _fmt(metrics["calmar_ratio"], 2)) + \
        card("盈亏比", _fmt(metrics["win_loss_ratio"],2), f"均赢¥{_fmt(metrics['avg_win'])}/均亏¥{_fmt(metrics['avg_loss'])}")
    t3 = card("最大回撤", _fmt_pct(metrics["max_drawdown_pct"])) + \
        card("回撤时长", f"{metrics['max_drawdown_duration_days']}天") + \
        card("恢复因子", _fmt(metrics["recovery_factor"],2)) + \
        card("最大连亏", f"{metrics['max_consecutive_losses']}笔", f"连赢{metrics['max_consecutive_wins']}笔")
    t4 = card("毛利", _fmt(metrics["gross_profit"],0,prefix="¥\n")) + \
        card("毛损", _fmt(metrics["gross_loss"],0,prefix="¥\n")) + \
        card("盈利因子", _fmt(metrics["profit_factor"],2)) + \
        card("最大单笔", f"赢¥{_fmt(metrics['largest_win'],0)}", f"亏¥{_fmt(abs(metrics['largest_loss']),0)}")

    hist = _histogram_data([t.get("pnl",0) for t in trades if "close" in t.get("action","")])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JD Backtest Report {version}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:#0f0f23;color:#e0e0e0}}
.hdr{{background:linear-gradient(135deg,#1a1a3e,#16213e);padding:30px 40px;border-bottom:1px solid #2a2a5e}}
.hdr h1{{font-size:24px;color:#fff;margin-bottom:6px}}
.hdr .sub{{font-size:13px;color:#8888aa}}
.ct{{max-width:1280px;margin:0 auto;padding:24px 20px}}
.mg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:24px}}
.mc{{background:#1a1a3e;border:1px solid #2a2a5e;border-radius:10px;padding:16px;text-align:center}}
.mc:hover{{border-color:#4ecdc4}}
.mv{{font-size:20px;font-weight:700;color:#fff;margin-bottom:4px}}
.mt{{font-size:12px;color:#8888aa;text-transform:uppercase;letter-spacing:.5px}}
.mt-s{{font-size:11px;color:#666688;margin-top:4px}}
.sec{{background:#1a1a3e;border:1px solid #2a2a5e;border-radius:10px;padding:20px;margin-bottom:24px}}
.sec h2{{font-size:16px;color:#fff;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #2a2a5e}}
.cc img{{max-width:100%;border-radius:6px}}
.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl th{{background:#252550;color:#aaaacc;padding:8px 10px;text-align:left;font-weight:500;position:sticky;top:0}}
.tbl td{{padding:6px 10px;border-bottom:1px solid #252550;font-family:"SF Mono","Consolas",monospace}}
.tbl tr:hover{{background:#252550}}
.pos{{color:#4ecdc4;font-weight:600}}
.neg{{color:#ff6b6b;font-weight:600}}
.scroll{{max-height:480px;overflow-y:auto}}
.scroll::-webkit-scrollbar{{width:6px}}
.scroll::-webkit-scrollbar-thumb{{background:#3a3a6e;border-radius:3px}}
.sr{{display:flex;justify-content:space-between;align-items:center}}
.badge{{font-size:11px;padding:3px 10px;border-radius:12px;background:#252550;color:#8888aa}}
.ft{{padding:20px;text-align:center;font-size:12px;color:#555577}}
</style></head><body>
<div class="hdr"><h1>JD Multi-Contract Backtest — {version}</h1>
<div class="sub">GPT策略定义 + tqsdk数据 · 日成交量选主力 · MA20方向滤波 · 无保护止损 · 滑点0.2% · {len(trades)}条记录</div></div>
<div class="ct">
<div class="mg">{t1}</div>
<div class="sec"><h2>风险调整收益</h2><div class="mg">{t2}</div></div>
<div class="sec"><h2>回撤分析</h2><div class="mg">{t3}</div></div>
<div class="sec"><h2>盈亏分析</h2><div class="mg">{t4}</div></div>
<div class="sec"><div class="sr"><h2>权益曲线</h2><span class="badge">红色=最大回撤</span></div><div class="cc"><img src="data:image/png;base64,{chart_b64}"></div></div>
<div class="sec"><h2>盈亏分布</h2><div style="height:200px;display:flex;align-items:flex-end;gap:4px;padding:20px 10px">{hist}</div></div>
<div class="sec"><div class="sr"><h2>交易明细</h2><span class="badge">{len(trades)}条</span></div><div class="scroll"><table class="tbl"><thead><tr><th>#</th><th>时间</th><th>操作</th><th>价格</th><th>手数</th><th>盈亏</th><th>原因</th></tr></thead><tbody>{trade_rows}</tbody></table></div></div>
</div>
<div class="ft">Generated by testforjd {version} · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""
    return html
