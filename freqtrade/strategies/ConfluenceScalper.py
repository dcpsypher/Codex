"""
Confluence Scalper v1 — Binance USDC Futures (Freqtrade Strategy)
=================================================================
Mirrors the Pine Script v6 / Python backtester logic exactly.

All indicators are imported from backtest/indicators.py — the same functions
used in the Python backtester — so live signals are guaranteed to match the
backtest signals 1-for-1. No pandas_ta dependency; no indicator drift.

Order model:
  Entry  — limit order (maker = 0% fee on Binance USDC pairs)
  TP     — limit order (maker = 0% fee), via custom_exit
  SL     — stop-market on exchange (taker = 0.04%)
  Time   — market close after max_trade_bars candles, via custom_exit

Run backtesting (after `freqtrade download-data`):
  freqtrade backtesting -c freqtrade/config.json --strategy ConfluenceScalper

Run hyperopt (tune TP/SL/RSI):
  freqtrade hyperopt -c freqtrade/config.json --strategy ConfluenceScalper \
    --hyperopt-loss SharpeHyperOptLoss --pairs BTC/USDC:USDC --epochs 200
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame

# ── Import our exact indicator implementations ────────────────────────────────
# Adds the repo root to sys.path so `backtest` package is importable regardless
# of where freqtrade is invoked from.
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backtest.indicators import (  # noqa: E402
    atr,
    ema,
    macd,
    rsi,
    supertrend,
    swing_levels,
    vwap_daily,
)


class ConfluenceScalper(IStrategy):

    # ── Metadata ───────────────────────────────────────────────────────────────
    INTERFACE_VERSION = 3
    timeframe         = "1m"
    can_short         = True        # Binance Futures — long AND short

    use_exit_signal            = True
    exit_profit_only           = False
    ignore_roi_if_entry_signal = False

    # ── ROI disabled — exits handled by custom_exit (ATR TP) and custom_stoploss
    minimal_roi = {"0": 100}

    # Fallback hard-stop (safety net if custom_stoploss returns None)
    stoploss      = -0.10
    trailing_stop = False

    # ── Order types — limit everywhere, stop-market for SL ───────────────────
    order_types = {
        "entry":                 "limit",
        "exit":                  "limit",
        "stoploss":              "market",       # taker fee 0.04%
        "stoploss_on_exchange":  True,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ── Strategy parameters (update with optimizer results) ───────────────────
    ema_fast_len   = 20
    ema_slow_len   = 50
    st_atr_len     = 7
    st_factor      = 3.0
    rsi_len        = 14
    rsi_long_lo    = 40
    rsi_long_hi    = 65
    rsi_short_lo   = 35
    rsi_short_hi   = 60
    macd_fast      = 12
    macd_slow      = 26
    macd_sig       = 9
    vol_sma_len    = 20
    vol_mult       = 1.2
    swing_len      = 3
    atr_len        = 14
    tp_mult        = 1.0    # TP at 1× ATR  (closer = higher win rate)
    sl_mult        = 1.5    # SL at 1.5× ATR (wider = more breathing room)
    limit_offset   = 0.0002  # 0.02% inside close for maker fill probability
    max_trade_bars = 30      # time stop: flatten if trade lives longer than this

    # ─────────────────────────────────────────────────────────────────────────
    # Indicators
    # ─────────────────────────────────────────────────────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Computes all indicators using the same functions as the Python backtester.
        Freqtrade's dataframe has integer index; we temporarily set DatetimeIndex
        for VWAP and swing pivot calculations, then restore.
        """
        # Freqtrade provides a 'date' column; set as index for our functions
        df = dataframe.set_index("date")

        # ── Trend ─────────────────────────────────────────────────────────────
        dataframe["ema_fast"] = ema(df["close"], self.ema_fast_len).values
        dataframe["ema_slow"] = ema(df["close"], self.ema_slow_len).values

        st_line, st_dir = supertrend(
            df["high"], df["low"], df["close"],
            period=self.st_atr_len, factor=self.st_factor,
        )
        dataframe["st_line"]  = st_line.values
        dataframe["st_dir"]   = st_dir.values   # -1 = bullish, +1 = bearish

        # ── VWAP ──────────────────────────────────────────────────────────────
        dataframe["vwap"] = vwap_daily(df).values

        # ── Market structure ───────────────────────────────────────────────────
        sh1, sh2, sl1, sl2 = swing_levels(df["high"], df["low"], self.swing_len)
        dataframe["sh1"] = sh1.values
        dataframe["sh2"] = sh2.values
        dataframe["sl1"] = sl1.values
        dataframe["sl2"] = sl2.values

        # ── Momentum ───────────────────────────────────────────────────────────
        dataframe["rsi"] = rsi(df["close"], self.rsi_len).values

        macd_line, signal_line, histogram = macd(
            df["close"], self.macd_fast, self.macd_slow, self.macd_sig
        )
        dataframe["macd_line"] = macd_line.values
        dataframe["macd_sig"]  = signal_line.values
        dataframe["macd_hist"] = histogram.values

        # ── Volume ─────────────────────────────────────────────────────────────
        dataframe["vol_sma"] = df["volume"].rolling(self.vol_sma_len).mean().values

        # ── ATR ────────────────────────────────────────────────────────────────
        dataframe["atr"] = atr(df["high"], df["low"], df["close"], self.atr_len).values

        return dataframe

    # ─────────────────────────────────────────────────────────────────────────
    # Entry signals
    # ─────────────────────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # ── MACD momentum helpers ──────────────────────────────────────────────
        macd_xover  = (df["macd_line"] > df["macd_sig"]) & (df["macd_line"].shift(1) <= df["macd_sig"].shift(1))
        macd_xunder = (df["macd_line"] < df["macd_sig"]) & (df["macd_line"].shift(1) >= df["macd_sig"].shift(1))

        macd_bull = macd_xover | (
            (df["macd_line"] > df["macd_sig"]) &
            (df["macd_hist"] > df["macd_hist"].shift(1)) &
            (df["macd_hist"] > 0)
        )
        macd_bear = macd_xunder | (
            (df["macd_line"] < df["macd_sig"]) &
            (df["macd_hist"] < df["macd_hist"].shift(1)) &
            (df["macd_hist"] < 0)
        )

        # ── Market structure helpers ───────────────────────────────────────────
        struct_valid = df["sh1"].notna() & df["sh2"].notna() & df["sl1"].notna() & df["sl2"].notna()
        bull_ms = struct_valid & (df["sh1"] > df["sh2"]) & (df["sl1"] > df["sl2"])
        bear_ms = struct_valid & (df["sh1"] < df["sh2"]) & (df["sl1"] < df["sl2"])
        ema_accel_bull = df["ema_fast"] > df["ema_fast"].shift(5)
        ema_accel_bear = df["ema_fast"] < df["ema_fast"].shift(5)

        vol_ok = df["volume"] > self.vol_mult * df["vol_sma"]

        # ── Long ───────────────────────────────────────────────────────────────
        long_cond = (
            (df["ema_fast"] > df["ema_slow"])       &   # EMA: bullish structure
            (df["st_dir"]   < 0)                    &   # Supertrend: bullish
            (df["close"]    > df["vwap"])            &   # VWAP: above anchor
            (bull_ms | ema_accel_bull)               &   # Market structure
            (df["rsi"] > self.rsi_long_lo)           &   # RSI: not oversold
            (df["rsi"] < self.rsi_long_hi)           &   # RSI: not overbought
            macd_bull                                &   # MACD: momentum up
            vol_ok                                   &   # Volume: real move
            (df["volume"] > 0)
        )
        dataframe.loc[long_cond,  "enter_long"]  = 1
        dataframe.loc[long_cond,  "enter_tag"]   = "Long"

        # ── Short ──────────────────────────────────────────────────────────────
        short_cond = (
            (df["ema_fast"] < df["ema_slow"])       &
            (df["st_dir"]   > 0)                    &
            (df["close"]    < df["vwap"])            &
            (bear_ms | ema_accel_bear)               &
            (df["rsi"] > self.rsi_short_lo)          &
            (df["rsi"] < self.rsi_short_hi)          &
            macd_bear                                &
            vol_ok                                   &
            (df["volume"] > 0)
        )
        dataframe.loc[short_cond, "enter_short"] = 1
        dataframe.loc[short_cond, "enter_tag"]   = "Short"

        return dataframe

    # ─────────────────────────────────────────────────────────────────────────
    # Exit signals — handled via custom_exit and custom_stoploss
    # ─────────────────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"]  = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ─────────────────────────────────────────────────────────────────────────
    # Custom entry price — limit inside spread for maker fill
    # ─────────────────────────────────────────────────────────────────────────

    def custom_entry_price(
        self, pair: str, trade, current_time: datetime,
        proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs,
    ) -> float:
        if side == "long":
            return proposed_rate * (1.0 - self.limit_offset)
        return proposed_rate * (1.0 + self.limit_offset)

    # ─────────────────────────────────────────────────────────────────────────
    # Custom stoploss — ATR-based with breakeven trail
    # ─────────────────────────────────────────────────────────────────────────

    def custom_stoploss(
        self, pair: str, trade, current_time: datetime,
        current_rate: float, current_profit: float, after_fill: bool, **kwargs,
    ) -> Optional[float]:
        """
        Returns an ATR-based stoploss.
        Once price moves 1× SL-ATR in our favour, trails to breakeven.
        Uses stoploss_from_absolute() which correctly handles long/short sign.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        atr_val = dataframe.iloc[-1].get("atr", None)
        if atr_val is None or np.isnan(float(atr_val)):
            return None
        atr_val = float(atr_val)

        entry  = trade.open_rate
        sl_dist = self.sl_mult * atr_val

        if trade.is_short:
            sl_price = entry + sl_dist
            # Breakeven: price has moved 1× SL distance in our favour
            if current_rate <= entry - sl_dist:
                sl_price = min(sl_price, entry)
        else:
            sl_price = entry - sl_dist
            if current_rate >= entry + sl_dist:
                sl_price = max(sl_price, entry)

        return stoploss_from_absolute(sl_price, current_rate, is_short=trade.is_short)

    # ─────────────────────────────────────────────────────────────────────────
    # Custom exit — take-profit limit + time stop
    # ─────────────────────────────────────────────────────────────────────────

    def custom_exit(
        self, pair: str, trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs,
    ) -> Optional[str]:
        """
        Closes at 1× ATR take-profit (limit order, 0% fee) or
        flattens after max_trade_bars (market, 0.04% fee).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        atr_val = dataframe.iloc[-1].get("atr", None)
        if atr_val is None or np.isnan(float(atr_val)):
            return None
        atr_val = float(atr_val)

        entry   = trade.open_rate
        tp_dist = self.tp_mult * atr_val

        if trade.is_short:
            if current_rate <= entry - tp_dist:
                return "TP"
        else:
            if current_rate >= entry + tp_dist:
                return "TP"

        # Time stop — flatten if trade has been open too long
        bars_open = (current_time - trade.open_date_utc).total_seconds() / 60
        if bars_open >= self.max_trade_bars:
            return "Time Stop"

        return None
