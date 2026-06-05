#!/usr/bin/env python3
"""
Run a single backtest for one or more symbols — 5m VWAP-Bounce strategy.

Usage:
    python run_backtest.py                                # defaults: BTC+ETH+SOL, 3 months
    python run_backtest.py --symbol BTC/USDT:USDT
    python run_backtest.py --months 6
    python run_backtest.py --tp 1.5 --sl 1.0            # custom TP/SL multipliers
    python run_backtest.py --refresh                     # force re-download data
"""

import argparse
import sys

from rich.console import Console

from backtest.data import fetch_ohlcv
from backtest.engine import compute_stats, run_backtest
from backtest.indicators import add_indicators
from backtest.report import print_report
from backtest.strategy import generate_signals

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Default parameters  (mirror Pine Script v6 inputs exactly)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # ── Trend ──────────────────────────────────────────────────────────────
    "ema_fast":        20,
    "ema_slow":        50,
    "st_factor":       3.0,
    "st_atr_len":      7,
    # ── Momentum ───────────────────────────────────────────────────────────
    "rsi_len":         14,
    "rsi_long_lo":     40,
    "rsi_long_hi":     65,
    "rsi_short_lo":    35,
    "rsi_short_hi":    60,
    "macd_fast":       12,
    "macd_slow":       26,
    "macd_signal":     9,
    # ── Volume ─────────────────────────────────────────────────────────────
    "vol_len":         20,
    "vol_mult":        1.2,
    # ── Market structure (still computed for debugging, not used in signals) ─
    "swing_len":       3,
    # ── VWAP bounce entry trigger ──────────────────────────────────────────
    # Low must be within this % above VWAP for a long bounce to count.
    # High must be within this % below VWAP for a short rejection to count.
    "vwap_touch_pct":  0.003,   # 0.3 % proximity band
    # ── Risk management ────────────────────────────────────────────────────
    "atr_len":         14,
    "tp_mult":         1.5,     # TP at 1.5× ATR — 5m moves are larger; good R:R
    "sl_mult":         1.0,     # SL at 1× ATR  — tight since entry is at VWAP support
    "limit_offset_pct": 0.02,
    "max_bars":        12,      # 12 × 5m = 60-minute time stop
    # ── Exchange / account ─────────────────────────────────────────────────
    "leverage":           10,
    "initial_capital":    20.0,
    "risk_per_trade_pct": 1.5,  # risk 1.5% of equity per trade
    "maker_fee":          0.0,  # 0% on Binance USDC/USDT limit orders
    "taker_fee":          0.0004,  # 0.04% taker (SL / time stop exits)
}

DEFAULT_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Confluence Scalper Backtester")
    parser.add_argument("--symbol",   type=str,   default=None,
                        help="e.g. BTC/USDT:USDT (default: BTC+ETH+SOL)")
    parser.add_argument("--months",   type=int,   default=3,
                        help="Months of history (default: 3)")
    parser.add_argument("--tp",       type=float, default=None, help="TP ATR multiplier")
    parser.add_argument("--sl",       type=float, default=None, help="SL ATR multiplier")
    parser.add_argument("--leverage", type=int,   default=None, help="Leverage display")
    parser.add_argument("--refresh",  action="store_true",
                        help="Force re-download data (ignore cache)")
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.tp:
        params["tp_mult"] = args.tp
    if args.sl:
        params["sl_mult"] = args.sl
    if args.leverage:
        params["leverage"] = args.leverage

    symbols = [args.symbol] if args.symbol else DEFAULT_SYMBOLS

    console.print("\n[bold cyan]═══ Confluence Scalper — Python Backtester ═══[/bold cyan]\n")

    for sym in symbols:
        console.print(f"[bold]{sym}[/bold]")
        try:
            raw = fetch_ohlcv(sym, "5m", args.months, force_refresh=args.refresh)
        except Exception as e:
            console.print(f"  [red]Data fetch failed: {e}[/red]")
            continue

        console.print("  Computing indicators…")
        df = add_indicators(raw, params)

        console.print("  Generating signals…")
        df = generate_signals(df, params)

        n_long  = int(df["long_cond"].sum())
        n_short = int(df["short_cond"].sum())
        console.print(f"  Signals → Long: [green]{n_long}[/green]  Short: [red]{n_short}[/red]")

        console.print("  Running simulation…")
        trades, equity = run_backtest(df, params)
        stats = compute_stats(trades, equity, params["initial_capital"])

        period = (
            f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
        )
        print_report(stats, params, sym, period)


if __name__ == "__main__":
    main()
