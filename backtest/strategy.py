"""
Signal generation — v3 Donchian-breakout entry.

The 5m VWAP-bounce (v2) selected end-of-trend reversals: buying pullbacks in a
confirmed uptrend is anti-predictive during the ranging markets that dominate
crypto. v3 flips to BREAKOUT entries — enter when price breaks the prior N-bar
range, i.e. when momentum is accelerating, not exhausting. Breakouts trade well
in both trending and range-expansion regimes.

Long setup  (all must hold):
  1. EMA fast > EMA slow            — fast-EMA trend alignment (9/21 by default)
  2. Close > highest high of prior  — Donchian breakout above the range
     `breakout_len` bars
  3. RSI below overbought cap       — don't chase a vertical move
  4. Volume > mult × volume SMA     — breakout backed by real participation

Short setup mirrors the above (close < prior-range low, EMA fast < slow, etc.).

Returns the same DataFrame with two new boolean columns:
  long_cond  — all long conditions are True (position check handled by engine)
  short_cond — all short conditions are True
"""

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()

    breakout_len = int(params.get("breakout_len", 20))

    # ── Trend layer (fast EMA alignment) ──────────────────────────────────────
    ema_bull = df["ema_fast"] > df["ema_slow"]
    ema_bear = df["ema_fast"] < df["ema_slow"]

    # ── Donchian breakout trigger ─────────────────────────────────────────────
    # Prior-bar range only (shift(1)) so the breakout bar itself isn't included —
    # no lookahead. Close must clear the highest high / lowest low of the window.
    prior_high = df["high"].shift(1).rolling(breakout_len).max()
    prior_low  = df["low"].shift(1).rolling(breakout_len).min()
    breakout_long  = df["close"] > prior_high
    breakout_short = df["close"] < prior_low

    # ── Momentum — RSI guard (don't chase exhaustion) ─────────────────────────
    rsi_long_ok  = df["rsi"] < params["rsi_long_hi"]
    rsi_short_ok = df["rsi"] > params["rsi_short_lo"]

    # ── Volume confirmation ───────────────────────────────────────────────────
    vol_ok = df["volume"] > params["vol_mult"] * df["vol_sma"]

    # ── Regime filter — only trade breakouts in a strong trend ────────────────
    # Low ADX = chop, where breakouts whipsaw (the false-breakout trap that sank
    # the unfiltered v3). Requiring ADX above a floor skips ranging conditions.
    adx_ok = df["adx"] > params.get("adx_min", 25)

    # ── Combined signals (position check handled by engine) ───────────────────
    df["long_cond"] = (
        ema_bull & breakout_long & rsi_long_ok & vol_ok & adx_ok
    )
    df["short_cond"] = (
        ema_bear & breakout_short & rsi_short_ok & vol_ok & adx_ok
    )

    # Fill NaN from indicator warm-up as False
    df["long_cond"]  = df["long_cond"].fillna(False)
    df["short_cond"] = df["short_cond"].fillna(False)

    return df
