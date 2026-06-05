"""
Signal generation — mirrors the Pine Script v6 Confluence Scalper entry logic
exactly, operating only on already-computed indicator columns.

Returns the same DataFrame with two new boolean columns:
  long_cond  — all long entry conditions are True (position check handled by engine)
  short_cond — all short entry conditions are True
"""

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()

    # ── Trend layer ────────────────────────────────────────────────────────
    ema_bull = df["ema_fast"] > df["ema_slow"]
    ema_bear = df["ema_fast"] < df["ema_slow"]

    st_bull = df["st_dir"] < 0   # -1 = bullish in our Supertrend convention
    st_bear = df["st_dir"] > 0

    # ── VWAP layer ─────────────────────────────────────────────────────────
    above_vwap = df["close"] > df["vwap"]
    below_vwap = df["close"] < df["vwap"]

    # ── Market structure layer ─────────────────────────────────────────────
    # HH+HL = bullish structure; LL+LH = bearish structure
    # Falls back to EMA fast acceleration when pivots haven't confirmed yet
    struct_valid = (
        df["sh1"].notna() & df["sh2"].notna() &
        df["sl1"].notna() & df["sl2"].notna()
    )
    bull_ms = struct_valid & (df["sh1"] > df["sh2"]) & (df["sl1"] > df["sl2"])
    bear_ms = struct_valid & (df["sh1"] < df["sh2"]) & (df["sl1"] < df["sl2"])

    ema_accel_bull = df["ema_fast"] > df["ema_fast"].shift(5)
    ema_accel_bear = df["ema_fast"] < df["ema_fast"].shift(5)

    struct_bull = bull_ms | ema_accel_bull
    struct_bear = bear_ms | ema_accel_bear

    # ── Momentum layer — RSI ───────────────────────────────────────────────
    rsi_long_ok  = (df["rsi"] > params["rsi_long_lo"])  & (df["rsi"] < params["rsi_long_hi"])
    rsi_short_ok = (df["rsi"] > params["rsi_short_lo"]) & (df["rsi"] < params["rsi_short_hi"])

    # ── Momentum layer — MACD ─────────────────────────────────────────────
    # Fresh crossover OR histogram accelerating in the right direction
    macd_xover = (df["macd_line"] > df["macd_sig"]) & (df["macd_line"].shift(1) <= df["macd_sig"].shift(1))
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

    # ── Volume layer ───────────────────────────────────────────────────────
    vol_ok = df["volume"] > params["vol_mult"] * df["vol_sma"]

    # ── Combined signals (no position check — handled by engine) ──────────
    df["long_cond"] = (
        ema_bull & st_bull & above_vwap & struct_bull &
        rsi_long_ok & macd_bull & vol_ok
    )
    df["short_cond"] = (
        ema_bear & st_bear & below_vwap & struct_bear &
        rsi_short_ok & macd_bear & vol_ok
    )

    # Fill NaN (from indicator warm-up) as False
    df["long_cond"]  = df["long_cond"].fillna(False)
    df["short_cond"] = df["short_cond"].fillna(False)

    return df
