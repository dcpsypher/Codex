"""
Signal generation — 5m VWAP-bounce entry strategy.

Long setup  (all must hold):
  1. EMA 20 > EMA 50              — uptrend structure
  2. Supertrend bullish           — intermediate trend confirmation
  3. VWAP bounce                  — low touched near VWAP, candle closes above VWAP as a
                                    bullish (close > open) bar — pullback-to-anchor entry
  4. RSI in healthy range         — not extreme / overbought
  5. MACD line above signal       — momentum net positive
  6. Volume above SMA             — real participation on the bounce bar

Short setup mirrors the above with bearish conditions.

Returns the same DataFrame with two new boolean columns:
  long_cond  — all long conditions are True (position check handled by engine)
  short_cond — all short conditions are True
"""

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()

    vwap_touch_pct = float(params.get("vwap_touch_pct", 0.003))

    # ── Trend layer ───────────────────────────────────────────────────────────
    ema_bull = df["ema_fast"] > df["ema_slow"]
    ema_bear = df["ema_fast"] < df["ema_slow"]
    st_bull  = df["st_dir"] < 0   # -1 = bullish (price above Supertrend line)
    st_bear  = df["st_dir"] > 0

    # ── VWAP bounce / rejection ───────────────────────────────────────────────
    # Long:  wick dipped to within vwap_touch_pct above VWAP, then closed above VWAP
    #        as a bullish candle — classic pullback-to-anchor with demand rejection
    vwap_touch_long  = df["low"]  <= df["vwap"] * (1.0 + vwap_touch_pct)
    vwap_bounce_long = (
        vwap_touch_long
        & (df["close"] > df["vwap"])
        & (df["close"] > df["open"])
    )

    # Short: wick rallied to within vwap_touch_pct below VWAP, then closed below
    #        as a bearish candle — supply rejection at the VWAP ceiling
    vwap_touch_short  = df["high"] >= df["vwap"] * (1.0 - vwap_touch_pct)
    vwap_reject_short = (
        vwap_touch_short
        & (df["close"] < df["vwap"])
        & (df["close"] < df["open"])
    )

    # ── Momentum — RSI ────────────────────────────────────────────────────────
    rsi_long_ok  = (df["rsi"] >= params["rsi_long_lo"])  & (df["rsi"] <= params["rsi_long_hi"])
    rsi_short_ok = (df["rsi"] >= params["rsi_short_lo"]) & (df["rsi"] <= params["rsi_short_hi"])

    # ── Momentum — MACD direction ─────────────────────────────────────────────
    macd_bull = df["macd_line"] > df["macd_sig"]
    macd_bear = df["macd_line"] < df["macd_sig"]

    # ── Volume layer ──────────────────────────────────────────────────────────
    vol_ok = df["volume"] > params["vol_mult"] * df["vol_sma"]

    # ── Combined signals (position check handled by engine) ───────────────────
    df["long_cond"] = (
        ema_bull & st_bull & vwap_bounce_long & rsi_long_ok & macd_bull & vol_ok
    )
    df["short_cond"] = (
        ema_bear & st_bear & vwap_reject_short & rsi_short_ok & macd_bear & vol_ok
    )

    # Fill NaN from indicator warm-up as False
    df["long_cond"]  = df["long_cond"].fillna(False)
    df["short_cond"] = df["short_cond"].fillna(False)

    return df
