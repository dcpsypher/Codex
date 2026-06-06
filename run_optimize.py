#!/usr/bin/env python3
"""
Parameter optimizer — grid searches TP/SL/RSI/Volume across BTC+ETH+SOL,
splits data 70/30 (train/test), reports the top combinations by out-of-sample score.

Usage:
    python run_optimize.py
    python run_optimize.py --symbol BTC/USDT:USDT   # single pair
    python run_optimize.py --months 6               # more data = more reliable
    python run_optimize.py --top 15                 # show top 15 results
"""

import argparse

from rich.console import Console

from backtest.data import fetch_ohlcv
from backtest.optimizer import run_grid_search
from backtest.report import print_optimization_results
from run_backtest import DEFAULT_PARAMS, DEFAULT_SYMBOLS

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Confluence Scalper Optimizer")
    parser.add_argument("--symbol", type=str,  default=None)
    parser.add_argument("--months", type=int,  default=3)
    parser.add_argument("--top",    type=int,  default=10)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else DEFAULT_SYMBOLS

    console.print(
        "\n[bold cyan]═══ Confluence Scalper — Parameter Optimizer ═══[/bold cyan]"
    )
    console.print(
        "[dim]Train/test split: 70% in-sample / 30% out-of-sample  "
        "(results shown are OOS only)[/dim]\n"
    )

    all_results: list[dict] = []

    for sym in symbols:
        console.print(f"[bold white]▶ {sym}[/bold white]")
        try:
            raw = fetch_ohlcv(sym, "5m", args.months)
        except Exception as e:
            console.print(f"  [red]Data fetch failed: {e}[/red]")
            continue

        results = run_grid_search(raw, DEFAULT_PARAMS, symbol=sym)
        console.print(
            f"  → [green]{len(results)}[/green] combinations passed thresholds\n"
        )
        all_results.extend(results)

    if not all_results:
        console.print("[red]No combinations met the minimum thresholds.[/red]")
        console.print(
            "[dim]Try --months 6 for more data, or relax thresholds in optimizer.py.[/dim]"
        )
        return

    all_results.sort(key=lambda r: r["score"], reverse=True)
    print_optimization_results(all_results, top_n=args.top)

    # ── Best config summary ───────────────────────────────────────────────
    best = all_results[0]
    console.print("\n[bold green]✓ Best config to plug into Pine Script / DEFAULT_PARAMS:[/bold green]")
    console.print(
        f"  TP mult       = [cyan]{best['tp_mult']}[/cyan]\n"
        f"  SL mult       = [cyan]{best['sl_mult']}[/cyan]\n"
        f"  Breakout len  = [cyan]{best['breakout_len']}[/cyan]\n"
        f"  ADX min       = [cyan]{best['adx_min']}[/cyan]\n"
        f"  Vol mult      = [cyan]{best['vol_mult']}[/cyan]\n"
        f"  (All other params unchanged from defaults)\n"
    )


if __name__ == "__main__":
    main()
