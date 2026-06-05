"""
Rich terminal reporting — pretty-prints backtest results and equity curve.
"""

from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_report(stats: dict, params: dict, symbol: str, period: str) -> None:
    """Prints a formatted backtest summary to the terminal."""

    if stats.get("n_trades", 0) == 0:
        console.print(f"[yellow]No trades generated for {symbol}.[/yellow]")
        return

    # ── Header ──────────────────────────────────────────────────────────────
    win_rate_col = "bright_green" if stats["win_rate"] >= 60 else "yellow" if stats["win_rate"] >= 45 else "bright_red"
    pf_col       = "bright_green" if stats["profit_factor"] >= 1.5 else "yellow" if stats["profit_factor"] >= 1.0 else "bright_red"
    dd_col       = "bright_green" if stats["max_drawdown"] >= -20 else "yellow" if stats["max_drawdown"] >= -30 else "bright_red"
    pnl_col      = "bright_green" if stats["net_pnl"] >= 0 else "bright_red"

    title = f"[bold white]Confluence Scalper v1  ·  {symbol}  ·  {period}[/bold white]"
    console.print(Panel(title, border_style="dim white"))

    # ── Key metrics ─────────────────────────────────────────────────────────
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim white")
    t.add_column("Metric",       style="dim white",  min_width=22)
    t.add_column("Value",        justify="right",    min_width=14)
    t.add_column("Target",       justify="right",    min_width=14, style="dim")

    rows = [
        ("Net P&L",          f"${stats['net_pnl']:+.4f}  ({stats['net_pnl_pct']:+.2f}%)", pnl_col, "positive"),
        ("Final Equity",     f"${stats['final_equity']:.4f}",  pnl_col, ""),
        ("Win Rate",         f"{stats['win_rate']:.1f}%",       win_rate_col, "≥ 60%"),
        ("Profit Factor",    f"{stats['profit_factor']:.3f}",   pf_col,       "≥ 1.50"),
        ("Max Drawdown",     f"{stats['max_drawdown']:.2f}%",   dd_col,       "≥ -20%"),
        ("Total Trades",     f"{stats['n_trades']}  (L:{stats['n_longs']} / S:{stats['n_shorts']})", "white", ""),
        ("TP / SL / Time",   f"{stats['tp_exits']} / {stats['sl_exits']} / {stats['time_exits']}", "white", ""),
        ("Avg Bars/Trade",   f"{stats['avg_bars']:.1f}",        "white",      ""),
        ("Avg Win",          f"${stats['avg_win']:+.4f}",        "bright_green", ""),
        ("Avg Loss",         f"${stats['avg_loss']:+.4f}",       "bright_red",   ""),
        ("Best Trade",       f"{stats['best_trade']:+.2f}%",     "bright_green", ""),
        ("Worst Trade",      f"{stats['worst_trade']:+.2f}%",    "bright_red",   ""),
    ]

    for label, value, color, target in rows:
        t.add_row(label, f"[{color}]{value}[/{color}]", target)

    console.print(t)

    # ── Config summary ───────────────────────────────────────────────────────
    cfg = (
        f"TP {params['tp_mult']}×ATR  |  SL {params['sl_mult']}×ATR  |  "
        f"EMA {params['ema_fast']}/{params['ema_slow']}  |  "
        f"ST({params['st_atr_len']},{params['st_factor']})  |  "
        f"RSI L:{params['rsi_long_lo']}-{params['rsi_long_hi']}  "
        f"S:{params['rsi_short_lo']}-{params['rsi_short_hi']}  |  "
        f"Vol×{params['vol_mult']}  |  "
        f"{params['leverage']}× lev  |  "
        f"Maker {params['maker_fee']*100:.2f}%  Taker {params['taker_fee']*100:.3f}%"
    )
    console.print(f"[dim]{cfg}[/dim]\n")


def print_optimization_results(results: list[dict], top_n: int = 10) -> None:
    """Prints the top N parameter combinations from an optimization run."""
    if not results:
        console.print("[red]No results to display.[/red]")
        return

    # Sort by composite score: win_rate weight 40%, pf 30%, dd 30%
    scored = sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:top_n]

    t = Table(
        title=f"[bold]Top {top_n} Parameter Combinations[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    t.add_column("#",          justify="right",  style="dim", width=3)
    t.add_column("Symbol",     justify="left",   width=14)
    t.add_column("TP×",        justify="right",  width=5)
    t.add_column("SL×",        justify="right",  width=5)
    t.add_column("RSI Lo/Hi",  justify="right",  width=10)
    t.add_column("Vol×",       justify="right",  width=5)
    t.add_column("WinRate",    justify="right",  width=9)
    t.add_column("PF",         justify="right",  width=7)
    t.add_column("MaxDD",      justify="right",  width=8)
    t.add_column("NetPnL%",    justify="right",  width=9)
    t.add_column("Trades",     justify="right",  width=7)
    t.add_column("Score",      justify="right",  width=7)

    for rank, r in enumerate(scored, 1):
        wr_col  = "bright_green" if r["win_rate"] >= 60 else "yellow" if r["win_rate"] >= 45 else "red"
        pf_col  = "bright_green" if r["profit_factor"] >= 1.5 else "yellow" if r["profit_factor"] >= 1.0 else "red"
        dd_col  = "bright_green" if r["max_drawdown"] >= -20 else "yellow"
        pnl_col = "bright_green" if r["net_pnl_pct"] >= 0 else "red"

        t.add_row(
            str(rank),
            r.get("symbol", ""),
            f"{r['tp_mult']:.1f}",
            f"{r['sl_mult']:.1f}",
            f"{r['rsi_long_lo']}/{r['rsi_long_hi']}",
            f"{r['vol_mult']:.1f}",
            f"[{wr_col}]{r['win_rate']:.1f}%[/{wr_col}]",
            f"[{pf_col}]{r['profit_factor']:.2f}[/{pf_col}]",
            f"[{dd_col}]{r['max_drawdown']:.1f}%[/{dd_col}]",
            f"[{pnl_col}]{r['net_pnl_pct']:+.1f}%[/{pnl_col}]",
            str(r["n_trades"]),
            f"{r['score']:.3f}",
        )

    console.print(t)
