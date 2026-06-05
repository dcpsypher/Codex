"""
Synthetic BTC/ETH/SOL-like 1m OHLCV generator.

Uses GBM with:
  - Regime switching (trending / ranging / volatile)
  - Volatility clustering (simplified GARCH)
  - Fat-tailed noise (Student-t, df=4 typical for crypto)
  - Realistic volume correlated with absolute returns

Useful for local smoke tests and CI when Binance API isn't available.
This is NOT a substitute for real data — always backtest on real OHLCV
before trading with real money.
"""

import numpy as np
import pandas as pd


def generate(
    symbol: str = "BTC/USDT:USDT",
    start_price: float = 65000.0,
    n_bars: int = 131_040,   # ~3 months of 1m bars
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [open, high, low, close, volume]
    indexed by UTC timestamps, 1m frequency.
    """
    rng = np.random.default_rng(seed)
    n   = n_bars

    # ── Regime schedule ─────────────────────────────────────────────────────
    # Alternating blocks of trending, ranging, volatile regimes
    regime_len = 1440   # ~1 day per regime change
    n_regimes  = n // regime_len + 1
    regime_types = rng.choice(["bull", "bear", "range", "volatile"],
                               size=n_regimes,
                               p=[0.30, 0.25, 0.30, 0.15])

    mu_map    = {"bull": 0.0003, "bear": -0.0003, "range": 0.0,   "volatile": 0.0}
    sigma_map = {"bull": 0.0008, "bear": 0.0008,  "range": 0.0005, "volatile": 0.0020}

    mu    = np.array([mu_map[regime_types[i // regime_len]] for i in range(n)])
    sigma = np.array([sigma_map[regime_types[i // regime_len]] for i in range(n)])

    # ── Volatility clustering (GARCH-lite) ──────────────────────────────────
    vol = sigma.copy()
    alpha, beta = 0.10, 0.85
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = rng.standard_t(df=4) * vol[i - 1]
        vol[i]   = np.sqrt(sigma[i] ** 2 * (1 - alpha - beta)
                           + alpha * resid[i] ** 2
                           + beta  * vol[i - 1] ** 2)
    vol = np.clip(vol, sigma * 0.3, sigma * 5.0)

    # ── Log returns → price path ─────────────────────────────────────────────
    returns = mu + vol * rng.standard_t(df=4, size=n)
    log_price = np.log(start_price) + np.cumsum(returns)
    close = np.exp(log_price)

    # ── Build OHLC from close ────────────────────────────────────────────────
    bar_range = np.abs(rng.normal(0, vol * close * 1.5, n)) + 1.0   # in price units
    open_  = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1] + rng.normal(0, bar_range[:-1] * 0.1)   # tiny gap

    high = np.maximum(open_, close) + np.abs(rng.normal(0, bar_range * 0.5, n))
    low  = np.minimum(open_, close) - np.abs(rng.normal(0, bar_range * 0.5, n))
    low  = np.minimum(low, close * 0.995)   # sanity floor

    # ── Volume: spikes correlated with large candles ─────────────────────────
    base_vol  = rng.exponential(scale=500, size=n) + 200
    vol_spike = 1.0 + 3.0 * np.abs(returns) / (vol + 1e-12)
    volume    = base_vol * vol_spike

    # ── Assemble ─────────────────────────────────────────────────────────────
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    df  = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    # Sanity: high >= low, high >= close, low <= close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"]  = df[["open", "low",  "close"]].min(axis=1)

    return df.astype(np.float64)
