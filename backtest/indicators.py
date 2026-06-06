"""
All indicator calculations — implemented from scratch to match Pine Script v6
exactly. No external TA libraries needed, giving full control and zero
discrepancy with the TradingView strategy.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — used inside ATR and RSI."""
    return series.ewm(com=period - 1, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    return rma(true_range(high, low, close), period)


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    avg_gain = rma(delta.clip(lower=0), period)
    avg_loss = rma((-delta).clip(lower=0), period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average Directional Index (Wilder's) — trend-strength gauge in [0, 100].
    Matches ta.adx(period) in Pine Script v6. High ADX (>25) = strong trend;
    low ADX (<20) = chop/range. Used as a regime filter for breakout entries.
    """
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(plus_dm,  index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr  = true_range(high, low, close)
    atr_ = rma(tr, period)

    plus_di  = 100.0 * rma(plus_dm,  period) / atr_.replace(0, np.nan)
    minus_di = 100.0 * rma(minus_dm, period) / atr_.replace(0, np.nan)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return rma(dx, period)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend  (Pine Script v6 identical implementation)
# ─────────────────────────────────────────────────────────────────────────────

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 7,
    factor: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (st_line, direction).
    direction: -1 = bullish (price above ST line), +1 = bearish.
    Matches ta.supertrend(factor, period) in Pine Script v6.
    """
    hl2 = (high + low) / 2
    atr_vals = atr(high, low, close, period).values

    basic_upper = (hl2 + factor * atr_vals).values
    basic_lower = (hl2 - factor * atr_vals).values
    close_vals = close.values

    n = len(close_vals)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = np.ones(n, dtype=np.float64)    # 1 = bearish, -1 = bullish
    st_line = np.full(n, np.nan)

    # Seed first bar
    st_line[0] = final_upper[0]

    for i in range(1, n):
        # Update final upper band
        if basic_upper[i] < final_upper[i - 1] or close_vals[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Update final lower band
        if basic_lower[i] > final_lower[i - 1] or close_vals[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction
        if st_line[i - 1] == final_upper[i - 1]:
            direction[i] = -1.0 if close_vals[i] > final_upper[i] else 1.0
        else:
            direction[i] = 1.0 if close_vals[i] < final_lower[i] else -1.0

        st_line[i] = final_lower[i] if direction[i] == -1.0 else final_upper[i]

    return (
        pd.Series(st_line, index=close.index),
        pd.Series(direction, index=close.index),
    )


# ─────────────────────────────────────────────────────────────────────────────
# VWAP — daily reset (matches TradingView default for 24/7 crypto)
# ─────────────────────────────────────────────────────────────────────────────

def vwap_daily(df: pd.DataFrame) -> pd.Series:
    """
    VWAP that resets at midnight UTC each calendar day.
    Uses HLC3 as the typical price, matching ta.vwap(hlc3) in Pine Script.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = tp * df["volume"]
    date_key = df.index.normalize()  # midnight UTC per bar

    cum_tp_vol = tp_vol.groupby(date_key).cumsum()
    cum_vol = df["volume"].groupby(date_key).cumsum()
    return cum_tp_vol / cum_vol


# ─────────────────────────────────────────────────────────────────────────────
# Market structure — swing pivot detection (no lookahead)
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_high_confirmed(high: pd.Series, left: int, right: int) -> pd.Series:
    """
    Pivot high confirmed at bar i+right (matching Pine Script's ta.pivothigh).
    A bar is a swing high if it is strictly the highest in [i-left … i+right].
    Returns a Series with NaN except at bars where a pivot was confirmed.
    """
    vals = high.values
    n = len(vals)
    result = np.full(n, np.nan)

    for i in range(left, n - right):
        candidate = vals[i]
        left_win = vals[i - left : i]
        right_win = vals[i + 1 : i + right + 1]
        if (
            len(left_win) == left
            and len(right_win) == right
            and candidate > left_win.max()
            and candidate >= right_win.max()
        ):
            result[i + right] = candidate   # confirmed at i+right, no lookahead

    return pd.Series(result, index=high.index)


def _pivot_low_confirmed(low: pd.Series, left: int, right: int) -> pd.Series:
    vals = low.values
    n = len(vals)
    result = np.full(n, np.nan)

    for i in range(left, n - right):
        candidate = vals[i]
        left_win = vals[i - left : i]
        right_win = vals[i + 1 : i + right + 1]
        if (
            len(left_win) == left
            and len(right_win) == right
            and candidate < left_win.min()
            and candidate <= right_win.min()
        ):
            result[i + right] = candidate

    return pd.Series(result, index=low.index)


def swing_levels(
    high: pd.Series, low: pd.Series, swing_len: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns (sh1, sh2, sl1, sl2) — the most recent and second-most-recent
    confirmed swing highs and lows at each bar.
    """
    ph = _pivot_high_confirmed(high, swing_len, swing_len)
    pl = _pivot_low_confirmed(low,  swing_len, swing_len)

    ph_vals = ph.values
    pl_vals = pl.values
    n = len(high)

    sh1 = np.full(n, np.nan)
    sh2 = np.full(n, np.nan)
    sl1 = np.full(n, np.nan)
    sl2 = np.full(n, np.nan)

    last_sh = [np.nan, np.nan]
    last_sl = [np.nan, np.nan]

    for i in range(n):
        if not np.isnan(ph_vals[i]):
            last_sh[1] = last_sh[0]
            last_sh[0] = ph_vals[i]
        if not np.isnan(pl_vals[i]):
            last_sl[1] = last_sl[0]
            last_sl[0] = pl_vals[i]

        sh1[i] = last_sh[0]
        sh2[i] = last_sh[1]
        sl1[i] = last_sl[0]
        sl2[i] = last_sl[1]

    return (
        pd.Series(sh1, index=high.index),
        pd.Series(sh2, index=high.index),
        pd.Series(sl1, index=low.index),
        pd.Series(sl2, index=low.index),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Master function — adds all indicator columns to a OHLCV DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Adds all indicator columns in-place. Expects columns: open, high, low, close, volume.
    """
    df = df.copy()

    # ── Trend ──────────────────────────────────────────────────────────────
    df["ema_fast"] = ema(df["close"], params["ema_fast"])
    df["ema_slow"] = ema(df["close"], params["ema_slow"])

    df["st_line"], df["st_dir"] = supertrend(
        df["high"], df["low"], df["close"],
        params["st_atr_len"], params["st_factor"]
    )

    # ── VWAP ───────────────────────────────────────────────────────────────
    df["vwap"] = vwap_daily(df)

    # ── Market structure ───────────────────────────────────────────────────
    df["sh1"], df["sh2"], df["sl1"], df["sl2"] = swing_levels(
        df["high"], df["low"], params["swing_len"]
    )

    # ── Momentum ───────────────────────────────────────────────────────────
    df["rsi"] = rsi(df["close"], params["rsi_len"])

    df["macd_line"], df["macd_sig"], df["macd_hist"] = macd(
        df["close"],
        params["macd_fast"],
        params["macd_slow"],
        params["macd_signal"],
    )

    # ── Volume ─────────────────────────────────────────────────────────────
    df["vol_sma"] = df["volume"].rolling(params["vol_len"]).mean()

    # ── ATR (for SL/TP sizing) ─────────────────────────────────────────────
    df["atr"] = atr(df["high"], df["low"], df["close"], params["atr_len"])

    # ── ADX (trend-strength regime filter) ─────────────────────────────────
    df["adx"] = adx(df["high"], df["low"], df["close"], params.get("adx_len", 14))

    return df
