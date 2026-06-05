"""
Confluence Scalper v1 — Binance USDC Futures
=============================================
Mirrors the Pine Script v6 logic for Python backtesting via Freqtrade.

Indicators used:
  Trend       : EMA 20, EMA 50, Supertrend (7, 3.0)
  VWAP        : intra-day VWAP reset
  Momentum    : RSI (14), MACD (12, 26, 9)
  Volume      : Volume vs 20-bar SMA
  Structure   : Pivot highs/lows (swing 3 bars each side)
  Risk sizing : ATR (14)

Order model:
  Entry  — limit order (maker = 0% fee on Binance USDC pairs)
  TP     — limit order (maker = 0% fee)
  SL     — stop-market (taker = 0.04%)

Run backtesting:
  freqtrade backtesting --strategy ConfluenceScalper \
    --pairs BTC/USDC:USDC ETH/USDC:USDC SOL/USDC:USDC \
    --timeframe 1m --timerange 20240101-20241231

Run hyperopt (tune parameters):
  freqtrade hyperopt --strategy ConfluenceScalper \
    --hyperopt-loss SharpeHyperOptLoss \
    --pairs BTC/USDC:USDC --timeframe 1m --timerange 20240601-20241231 \
    --epochs 200
"""

from datetime import datetime
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    merge_informative_pair,
)
from pandas import DataFrame


class ConfluenceScalper(IStrategy):

    # ── Strategy metadata ──────────────────────────────────────────────────────
    INTERFACE_VERSION = 3
    timeframe = "1m"
    can_short = True                   # Binance Futures — long AND short
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ── Risk / reward ──────────────────────────────────────────────────────────
    # ROI is disabled — exits are handled by custom_stoploss (ATR-based)
    minimal_roi = {"0": 100}           # effectively never exit by ROI alone

    stoploss = -0.05                   # max 5% fallback hard-stop (safety net)
    trailing_stop = False              # we handle breakeven manually

    # ── Order settings (limit orders everywhere except SL) ────────────────────
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",          # stop-market = taker fee (0.04%)
        "stoploss_on_exchange": True,  # place SL directly on Binance
    }
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC",
    }

    # ── Optimisable hyperparameters ────────────────────────────────────────────
    # Uncomment and run `freqtrade hyperopt` to search for better values.

    # buy_ema_fast   = IntParameter(10, 30,   default=20,  space="buy",  load=True)
    # buy_ema_slow   = IntParameter(30, 100,  default=50,  space="buy",  load=True)
    # buy_rsi_lo     = IntParameter(30, 50,   default=40,  space="buy",  load=True)
    # buy_rsi_hi     = IntParameter(55, 75,   default=65,  space="buy",  load=True)
    # sell_rsi_lo    = IntParameter(25, 45,   default=35,  space="sell", load=True)
    # sell_rsi_hi    = IntParameter(50, 65,   default=60,  space="sell", load=True)
    # buy_vol_mult   = DecimalParameter(1.0, 2.5, default=1.2, space="buy", load=True)
    # buy_tp_mult    = DecimalParameter(1.5, 4.0, default=2.0, space="buy", load=True)
    # buy_sl_mult    = DecimalParameter(0.5, 2.0, default=1.0, space="buy", load=True)

    # Fixed parameters (change here or enable hyperopt above)
    ema_fast_len = 20
    ema_slow_len = 50
    st_factor    = 3.0
    st_atr_len   = 7
    rsi_len      = 14
    rsi_long_lo  = 40
    rsi_long_hi  = 65
    rsi_short_lo = 35
    rsi_short_hi = 60
    macd_fast    = 12
    macd_slow    = 26
    macd_sig     = 9
    vol_sma_len  = 20
    vol_mult     = 1.2
    atr_len      = 14
    tp_mult      = 2.0
    sl_mult      = 1.0
    limit_offset = 0.0002              # 0.02% — entry limit inside close for maker fill
    max_trade_bars = 30                # time stop

    # ── Custom stoploss (ATR-based with breakeven trail) ───────────────────────

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """
        Returns a dynamic stoploss relative to current_rate.
        Implements:
          1. ATR-based initial SL at entry
          2. Breakeven trail once 1× SL-ATR profit is secured
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        last = dataframe.iloc[-1]
        atr  = last.get("atr", None)
        if atr is None or np.isnan(atr):
            return None

        entry_rate = trade.open_rate
        sl_distance = self.sl_mult * atr      # price distance for SL

        if trade.is_short:
            sl_price = entry_rate + sl_distance
            # Breakeven once 1× SL distance gained
            if current_rate <= entry_rate - sl_distance:
                sl_price = min(sl_price, entry_rate)
            return (sl_price - current_rate) / current_rate  # negative value
        else:
            sl_price = entry_rate - sl_distance
            if current_rate >= entry_rate + sl_distance:
                sl_price = max(sl_price, entry_rate)
            return (sl_price - current_rate) / current_rate

    # ── Custom exit (TP + time stop) ───────────────────────────────────────────

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """TP limit (2× ATR) and time-based exit."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        last  = dataframe.iloc[-1]
        atr   = last.get("atr", None)
        if atr is None or np.isnan(atr):
            return None

        entry_rate  = trade.open_rate
        tp_distance = self.tp_mult * atr

        # Take-profit hit
        if trade.is_short:
            if current_rate <= entry_rate - tp_distance:
                return "TP"
        else:
            if current_rate >= entry_rate + tp_distance:
                return "TP"

        # Time stop — flatten after max_trade_bars candles
        bars_open = (current_time - trade.open_date_utc).total_seconds() / 60
        if bars_open >= self.max_trade_bars:
            return "Time Stop"

        return None

    # ── Custom entry price (limit offset for maker fill) ──────────────────────

    def custom_entry_price(
        self,
        pair: str,
        trade,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Place limit slightly inside the spread to improve maker probability."""
        if side == "long":
            return proposed_rate * (1 - self.limit_offset)
        else:
            return proposed_rate * (1 + self.limit_offset)

    # ── Populate indicators ────────────────────────────────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend — EMA
        dataframe["ema_fast"] = pta.ema(dataframe["close"], length=self.ema_fast_len)
        dataframe["ema_slow"] = pta.ema(dataframe["close"], length=self.ema_slow_len)

        # Trend — Supertrend
        st = pta.supertrend(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            length=self.st_atr_len,
            multiplier=self.st_factor,
        )
        # pandas_ta Supertrend returns columns named SUPERTd_<len>_<mult>
        st_dir_col = [c for c in st.columns if c.startswith("SUPERTd")][0]
        st_val_col = [c for c in st.columns if c.startswith("SUPERT_")][0]
        dataframe["st_direction"] = st[st_dir_col]  # 1 = bearish, -1 = bullish
        dataframe["st_value"]     = st[st_val_col]

        # Trend — VWAP (session-anchored)
        dataframe["vwap"] = pta.vwap(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            dataframe["volume"],
        )

        # Momentum — RSI
        dataframe["rsi"] = pta.rsi(dataframe["close"], length=self.rsi_len)

        # Momentum — MACD
        macd = pta.macd(dataframe["close"],
                        fast=self.macd_fast,
                        slow=self.macd_slow,
                        signal=self.macd_sig)
        dataframe["macd"]        = macd[f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_sig}"]
        dataframe["macd_signal"] = macd[f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_sig}"]
        dataframe["macd_hist"]   = macd[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_sig}"]

        # Volume filter
        dataframe["vol_sma"] = pta.sma(dataframe["volume"], length=self.vol_sma_len)
        dataframe["vol_ok"]  = dataframe["volume"] > self.vol_mult * dataframe["vol_sma"]

        # ATR for dynamic SL/TP
        dataframe["atr"] = pta.atr(
            dataframe["high"], dataframe["low"], dataframe["close"],
            length=self.atr_len
        )

        # Market Structure — rolling swing highs/lows (3-bar pivots)
        n = 3
        dataframe["swing_high"] = dataframe["high"].rolling(2 * n + 1, center=True).max()
        dataframe["swing_low"]  = dataframe["low"].rolling(2 * n + 1, center=True).min()
        dataframe["is_swing_high"] = dataframe["high"] == dataframe["swing_high"]
        dataframe["is_swing_low"]  = dataframe["low"]  == dataframe["swing_low"]

        # Track last two confirmed swing highs/lows
        sh_vals = dataframe.loc[dataframe["is_swing_high"], "high"].reindex(dataframe.index).ffill()
        sh_prev = sh_vals.shift(1)
        sl_vals = dataframe.loc[dataframe["is_swing_low"], "low"].reindex(dataframe.index).ffill()
        sl_prev = sl_vals.shift(1)

        dataframe["bull_ms"] = (sh_vals > sh_prev) & (sl_vals > sl_prev)
        dataframe["bear_ms"] = (sh_vals < sh_prev) & (sl_vals < sl_prev)

        return dataframe

    # ── Entry signals ──────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # MACD momentum helpers
        macd_bull = (
            (dataframe["macd"] > dataframe["macd_signal"]) &
            (dataframe["macd_hist"] > dataframe["macd_hist"].shift(1)) &
            (dataframe["macd_hist"] > 0)
        ) | (
            (dataframe["macd"] > dataframe["macd_signal"]) &
            (dataframe["macd"].shift(1) <= dataframe["macd_signal"].shift(1))  # crossover
        )

        macd_bear = (
            (dataframe["macd"] < dataframe["macd_signal"]) &
            (dataframe["macd_hist"] < dataframe["macd_hist"].shift(1)) &
            (dataframe["macd_hist"] < 0)
        ) | (
            (dataframe["macd"] < dataframe["macd_signal"]) &
            (dataframe["macd"].shift(1) >= dataframe["macd_signal"].shift(1))  # crossunder
        )

        ema_bull_accel = dataframe["ema_fast"] > dataframe["ema_fast"].shift(5)
        ema_bear_accel = dataframe["ema_fast"] < dataframe["ema_fast"].shift(5)

        # ── Long conditions ──
        dataframe.loc[
            (dataframe["ema_fast"] > dataframe["ema_slow"])         &  # EMA trend up
            (dataframe["st_direction"] == -1)                        &  # Supertrend bullish
            (dataframe["close"] > dataframe["vwap"])                 &  # Above VWAP
            (dataframe["bull_ms"] | ema_bull_accel)                  &  # Structure / momentum
            (dataframe["rsi"] > self.rsi_long_lo)                    &
            (dataframe["rsi"] < self.rsi_long_hi)                    &
            macd_bull                                                 &  # MACD confirms
            dataframe["vol_ok"]                                       &  # Volume confirms
            (dataframe["volume"] > 0),
            "enter_long",
        ] = 1

        dataframe.loc[
            (dataframe["ema_fast"] > dataframe["ema_slow"])         &
            (dataframe["st_direction"] == -1)                        &
            (dataframe["close"] > dataframe["vwap"])                 &
            (dataframe["bull_ms"] | ema_bull_accel)                  &
            (dataframe["rsi"] > self.rsi_long_lo)                    &
            (dataframe["rsi"] < self.rsi_long_hi)                    &
            macd_bull                                                 &
            dataframe["vol_ok"],
            "enter_tag",
        ] = "Long"

        # ── Short conditions ──
        dataframe.loc[
            (dataframe["ema_fast"] < dataframe["ema_slow"])         &  # EMA trend down
            (dataframe["st_direction"] == 1)                         &  # Supertrend bearish
            (dataframe["close"] < dataframe["vwap"])                 &  # Below VWAP
            (dataframe["bear_ms"] | ema_bear_accel)                  &  # Structure / momentum
            (dataframe["rsi"] > self.rsi_short_lo)                   &
            (dataframe["rsi"] < self.rsi_short_hi)                   &
            macd_bear                                                 &
            dataframe["vol_ok"]                                       &
            (dataframe["volume"] > 0),
            "enter_short",
        ] = 1

        dataframe.loc[
            (dataframe["ema_fast"] < dataframe["ema_slow"])         &
            (dataframe["st_direction"] == 1)                         &
            (dataframe["close"] < dataframe["vwap"])                 &
            (dataframe["bear_ms"] | ema_bear_accel)                  &
            (dataframe["rsi"] > self.rsi_short_lo)                   &
            (dataframe["rsi"] < self.rsi_short_hi)                   &
            macd_bear                                                 &
            dataframe["vol_ok"],
            "enter_tag",
        ] = "Short"

        return dataframe

    # ── Exit signals (main exit via custom_stoploss / custom_exit) ────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # No indicator-based exits — handled in custom_exit and custom_stoploss
        dataframe["exit_long"]  = 0
        dataframe["exit_short"] = 0
        return dataframe
