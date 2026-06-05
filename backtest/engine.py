"""
Trade simulation engine — bar-by-bar, no lookahead.

Order execution model (matches Pine Script with calc_on_order_fills=True):
  Entry  : Limit order placed at close of signal bar
           → fills next bar if price touches the limit level
           → cancelled if conditions no longer hold before fill
  TP exit: Limit order (maker, 0% fee on Binance USDC/USDT pairs)
  SL exit: Stop-market order (taker, 0.04% fee)
  Time   : Market close after max_bars (taker fee)

Within a single bar where both TP and SL levels are inside the range:
  → Assume SL hit first (conservative / realistic for volatile 1m crypto).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_bar:   int
    exit_bar:    int
    side:        str          # "long" | "short"
    entry_price: float
    exit_price:  float
    exit_type:   str          # "tp" | "sl" | "time"
    pnl:         float        # net dollar P&L (after fees)
    pnl_pct:     float        # net % of equity at entry
    equity_after: float


@dataclass
class EngineState:
    state:        str   = "flat"          # flat | pending_long | pending_short | long | short
    limit_price:  float = 0.0
    entry_price:  float = 0.0
    tp_price:     float = 0.0
    sl_price:     float = 0.0
    entry_bar:    int   = 0
    bars_in:      int   = 0
    pending_signal_bar: int = -1          # bar that generated the pending order


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, params: dict) -> tuple[list[Trade], pd.Series]:
    """
    Runs a bar-by-bar simulation.

    Returns:
        trades     : list of Trade objects
        equity_curve: pd.Series aligned to df.index
    """
    capital   = float(params["initial_capital"])
    leverage  = float(params["leverage"])
    maker_fee = float(params["maker_fee"])   # 0.0 for limit orders
    taker_fee = float(params["taker_fee"])   # 0.0004 for market/stop orders
    lim_off   = float(params["limit_offset_pct"]) / 100.0
    tp_mult   = float(params["tp_mult"])
    sl_mult   = float(params["sl_mult"])
    max_bars  = int(params["max_bars"])

    equity = capital
    s = EngineState()
    trades: list[Trade] = []
    n = len(df)

    equity_vals = np.full(n, np.nan)
    equity_vals[0] = equity

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atrs   = df["atr"].values
    long_c = df["long_cond"].values
    short_c = df["short_cond"].values

    for i in range(1, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # ── Step 1: try to fill a pending limit order ──────────────────────
        if s.state == "pending_long":
            # Cancel if the condition that triggered the order is gone
            if not long_c[i - 1]:
                s.state = "flat"
            elif l <= s.limit_price:
                # Filled — set TP/SL based on ATR at fill bar
                atr_val = atrs[i]
                s.entry_price = s.limit_price
                s.tp_price    = s.entry_price + tp_mult * atr_val
                s.sl_price    = s.entry_price - sl_mult * atr_val
                s.state       = "long"
                s.entry_bar   = i
                s.bars_in     = 0

        elif s.state == "pending_short":
            if not short_c[i - 1]:
                s.state = "flat"
            elif h >= s.limit_price:
                atr_val = atrs[i]
                s.entry_price = s.limit_price
                s.tp_price    = s.entry_price - tp_mult * atr_val
                s.sl_price    = s.entry_price + sl_mult * atr_val
                s.state       = "short"
                s.entry_bar   = i
                s.bars_in     = 0

        # ── Step 2: manage open position ──────────────────────────────────
        if s.state == "long":
            s.bars_in += 1

            # SL and TP both touched this bar → SL first (conservative)
            sl_hit = l <= s.sl_price
            tp_hit = h >= s.tp_price

            if sl_hit and tp_hit:
                # Assume SL hit first
                exit_px   = min(s.sl_price, o)   # gap-down protection
                exit_type = "sl"
            elif tp_hit:
                exit_px   = s.tp_price
                exit_type = "tp"
            elif sl_hit:
                exit_px   = min(s.sl_price, o)
                exit_type = "sl"
            elif s.bars_in >= max_bars:
                exit_px   = c
                exit_type = "time"
            else:
                exit_px   = None
                exit_type = None

            if exit_px is not None:
                fee = maker_fee if exit_type == "tp" else taker_fee
                raw_pct = (exit_px - s.entry_price) / s.entry_price
                # Entry was a limit order → 0% entry fee always
                net_pct = raw_pct - fee
                pnl     = equity * leverage * net_pct
                equity  = max(equity + pnl, 0.0)   # floor at 0 (liquidated)

                trades.append(Trade(
                    entry_bar    = s.entry_bar,
                    exit_bar     = i,
                    side         = "long",
                    entry_price  = s.entry_price,
                    exit_price   = exit_px,
                    exit_type    = exit_type,
                    pnl          = pnl,
                    pnl_pct      = net_pct * leverage * 100,
                    equity_after = equity,
                ))
                s = EngineState()

        elif s.state == "short":
            s.bars_in += 1

            sl_hit = h >= s.sl_price
            tp_hit = l <= s.tp_price

            if sl_hit and tp_hit:
                exit_px   = max(s.sl_price, o)
                exit_type = "sl"
            elif tp_hit:
                exit_px   = s.tp_price
                exit_type = "tp"
            elif sl_hit:
                exit_px   = max(s.sl_price, o)
                exit_type = "sl"
            elif s.bars_in >= max_bars:
                exit_px   = c
                exit_type = "time"
            else:
                exit_px   = None
                exit_type = None

            if exit_px is not None:
                fee = maker_fee if exit_type == "tp" else taker_fee
                raw_pct = (s.entry_price - exit_px) / s.entry_price
                net_pct = raw_pct - fee
                pnl     = equity * leverage * net_pct
                equity  = max(equity + pnl, 0.0)

                trades.append(Trade(
                    entry_bar    = s.entry_bar,
                    exit_bar     = i,
                    side         = "short",
                    entry_price  = s.entry_price,
                    exit_price   = exit_px,
                    exit_type    = exit_type,
                    pnl          = pnl,
                    pnl_pct      = net_pct * leverage * 100,
                    equity_after = equity,
                ))
                s = EngineState()

        # ── Step 3: place new pending order if flat ────────────────────────
        if s.state == "flat":
            if long_c[i]:
                s.limit_price = closes[i] * (1 - lim_off)
                s.state       = "pending_long"
            elif short_c[i]:
                s.limit_price = closes[i] * (1 + lim_off)
                s.state       = "pending_short"

        equity_vals[i] = equity

    # Forward-fill equity during pending/in-trade periods
    equity_series = pd.Series(equity_vals, index=df.index)
    equity_series = equity_series.ffill()
    return trades, equity_series


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(trades: list[Trade], equity: pd.Series, initial_capital: float) -> dict:
    if not trades:
        return {
            "n_trades": 0, "n_longs": 0, "n_shorts": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "net_pnl": 0.0, "net_pnl_pct": 0.0,
            "final_equity": initial_capital,
            "max_drawdown": 0.0, "avg_bars": 0.0,
            "tp_exits": 0, "sl_exits": 0, "time_exits": 0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
        }

    pnls      = [t.pnl for t in trades]
    pnl_pcts  = [t.pnl_pct for t in trades]
    winners   = [t for t in trades if t.pnl > 0]
    losers    = [t for t in trades if t.pnl <= 0]

    gross_profit = sum(t.pnl for t in winners)
    gross_loss   = abs(sum(t.pnl for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_rate = len(winners) / len(trades) * 100

    # Max drawdown on equity curve
    eq = equity.ffill().values
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100
    max_dd = float(dd.min())

    # Average trade duration in bars
    avg_bars = np.mean([t.exit_bar - t.entry_bar for t in trades])

    # Long / short split
    longs  = [t for t in trades if t.side == "long"]
    shorts = [t for t in trades if t.side == "short"]

    tp_exits   = sum(1 for t in trades if t.exit_type == "tp")
    sl_exits   = sum(1 for t in trades if t.exit_type == "sl")
    time_exits = sum(1 for t in trades if t.exit_type == "time")

    final_equity = trades[-1].equity_after
    net_pnl      = final_equity - initial_capital
    net_pnl_pct  = net_pnl / initial_capital * 100

    return {
        "n_trades":       len(trades),
        "n_longs":        len(longs),
        "n_shorts":       len(shorts),
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "gross_profit":   gross_profit,
        "gross_loss":     gross_loss,
        "net_pnl":        net_pnl,
        "net_pnl_pct":    net_pnl_pct,
        "final_equity":   final_equity,
        "max_drawdown":   max_dd,
        "avg_bars":       avg_bars,
        "tp_exits":       tp_exits,
        "sl_exits":       sl_exits,
        "time_exits":     time_exits,
        "avg_win":        np.mean([t.pnl for t in winners]) if winners else 0,
        "avg_loss":       np.mean([t.pnl for t in losers]) if losers else 0,
        "best_trade":     max(pnl_pcts),
        "worst_trade":    min(pnl_pcts),
    }
