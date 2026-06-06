"""
Grid-search parameter optimizer.

Trains on the first 70% of data (in-sample), validates on the remaining 30%
(out-of-sample). Only combinations that meet minimum thresholds on BOTH
splits are considered good.

Scoring formula (composite, 0–1):
    score = 0.35 × norm(win_rate)
          + 0.30 × norm(profit_factor)
          + 0.20 × norm(-max_drawdown)   ← lower drawdown = better
          + 0.15 × norm(net_pnl_pct)
"""

import itertools
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

from backtest.indicators import add_indicators
from backtest.strategy import generate_signals
from backtest.engine import run_backtest, compute_stats

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid — these are the axes we sweep
# (keep total combinations manageable so it runs in a few minutes)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRID = {
    "tp_mult":       [1.5, 2.0, 2.5],
    "sl_mult":       [0.8, 1.0, 1.2],
    "breakout_len":  [10, 20, 30],
    "adx_min":       [20, 25, 30],
    "vol_mult":      [1.0, 1.2, 1.5],
}

# Minimum thresholds — both in-sample AND out-of-sample must pass
MIN_TRADES    = 20
MIN_WIN_RATE  = 50.0
MIN_PF        = 1.1
MAX_DRAWDOWN  = -35.0


def _score(stats: dict) -> float:
    """Composite score in [0, 1]."""
    wr  = min(stats["win_rate"] / 80.0, 1.0)               # normalise toward 80%
    pf  = min((stats["profit_factor"] - 1.0) / 2.0, 1.0)   # PF=3 → 1.0
    dd  = min((-stats["max_drawdown"]) / 50.0, 1.0)         # lower DD = higher score
    dd  = 1.0 - dd                                          # invert: 0 DD → score 1.0
    pnl = min(max(stats["net_pnl_pct"] / 50.0, 0.0), 1.0)  # normalise toward +50%

    return 0.35 * wr + 0.30 * pf + 0.20 * dd + 0.15 * pnl


def _passes_thresholds(stats: dict) -> bool:
    return (
        stats["n_trades"] >= MIN_TRADES
        and stats["win_rate"] >= MIN_WIN_RATE
        and stats["profit_factor"] >= MIN_PF
        and stats["max_drawdown"] >= MAX_DRAWDOWN
    )


def run_grid_search(
    raw_df: pd.DataFrame,
    base_params: dict,
    symbol: str = "",
    verbose: bool = True,
) -> list[dict]:
    """
    Runs a full grid search on raw_df.
    Returns a list of result dicts, sorted by OOS score descending.
    """
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    n      = len(combos)

    # Split: 70% in-sample, 30% out-of-sample
    split_idx = int(len(raw_df) * 0.70)
    is_df  = raw_df.iloc[:split_idx].copy()
    oos_df = raw_df.iloc[split_idx:].copy()

    if verbose:
        console.print(
            f"  Grid search: [cyan]{n}[/cyan] combos  |  "
            f"IS {len(is_df):,} bars  /  OOS {len(oos_df):,} bars"
        )

    results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Optimising {symbol}", total=n)

        for combo in combos:
            override = dict(zip(keys, combo))
            params   = {**base_params, **override}

            # ── In-sample ────────────────────────────────────────────────
            try:
                is_ind = add_indicators(is_df, params)
                is_sig = generate_signals(is_ind, params)
                is_trades, is_eq = run_backtest(is_sig, params)
                is_stats = compute_stats(is_trades, is_eq, params["initial_capital"])
            except Exception:
                progress.advance(task)
                continue

            if not _passes_thresholds(is_stats):
                progress.advance(task)
                continue

            # ── Out-of-sample ─────────────────────────────────────────────
            try:
                oos_ind = add_indicators(oos_df, params)
                oos_sig = generate_signals(oos_ind, params)
                oos_trades, oos_eq = run_backtest(oos_sig, params)
                oos_stats = compute_stats(oos_trades, oos_eq, params["initial_capital"])
            except Exception:
                progress.advance(task)
                continue

            if oos_stats.get("n_trades", 0) == 0:
                progress.advance(task)
                continue

            oos_score = _score(oos_stats)

            results.append({
                "symbol":         symbol,
                **override,
                # OOS (what matters for real trading)
                "win_rate":       oos_stats["win_rate"],
                "profit_factor":  oos_stats["profit_factor"],
                "max_drawdown":   oos_stats["max_drawdown"],
                "net_pnl_pct":    oos_stats["net_pnl_pct"],
                "n_trades":       oos_stats["n_trades"],
                "score":          oos_score,
                # IS for reference
                "is_win_rate":    is_stats["win_rate"],
                "is_pf":          is_stats["profit_factor"],
            })

            progress.advance(task)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
