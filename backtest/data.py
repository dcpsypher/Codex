"""
Data fetcher — pulls OHLCV from Binance USDT-margined futures via ccxt.
Results are cached to disk as Parquet so you only download once.

We use USDT perpetuals (BTC/USDT:USDT etc.) because they have the deepest
history and highest liquidity. Price action is identical to USDC futures for
signal backtesting; fee differences are handled in the engine config.
"""

import time
from pathlib import Path

import ccxt
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Binance USDT-margined futures  (no API key needed for public OHLCV)
_exchange = None


def _get_exchange() -> ccxt.binanceusdm:
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binanceusdm({"enableRateLimit": True})
    return _exchange


def fetch_ohlcv(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1m",
    months: int = 3,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [open, high, low, close, volume]
    indexed by UTC timestamp.

    Args:
        symbol:        ccxt-style perpetual symbol, e.g. "BTC/USDT:USDT"
        timeframe:     "1m", "3m", "5m", etc.
        months:        how many months of history to fetch
        force_refresh: ignore cache and re-download
    """
    safe = symbol.replace("/", "_").replace(":", "_")
    cache_file = CACHE_DIR / f"{safe}_{timeframe}_{months}mo.parquet"

    if cache_file.exists() and not force_refresh:
        df = pd.read_parquet(cache_file)
        print(f"  [{symbol}] Loaded {len(df):,} bars from cache ({cache_file.name})")
        return df

    ex = _get_exchange()
    # Parse timeframe to minutes so progress estimate works for any resolution.
    tf_lower = timeframe.lower()
    if tf_lower.endswith("h"):
        tf_minutes = int(tf_lower[:-1]) * 60
    elif tf_lower.endswith("d"):
        tf_minutes = int(tf_lower[:-1]) * 1440
    else:
        tf_minutes = int(tf_lower.rstrip("m") or 1)
    bars_per_month = (30 * 24 * 60) // max(tf_minutes, 1)
    total_needed = months * bars_per_month
    since_ms = ex.milliseconds() - months * 30 * 24 * 60 * 60 * 1000

    print(f"  [{symbol}] Downloading ~{total_needed:,} bars of {timeframe} data…")
    all_candles: list[list] = []

    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        except Exception as e:
            print(f"  Warning: fetch error ({e}), retrying in 5 s…")
            time.sleep(5)
            continue

        if not batch:
            break

        all_candles.extend(batch)
        since_ms = batch[-1][0] + 1

        pct = min(len(all_candles) / total_needed * 100, 100)
        print(f"    {len(all_candles):,} / {total_needed:,} bars  ({pct:.0f}%)", end="\r")

        # Stop once we've passed now
        if batch[-1][0] >= ex.milliseconds() - 2 * 60 * 1000:
            break

        time.sleep(0.25)

    print()  # newline after \r progress

    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = (
        df.set_index("ts")
        .pipe(lambda d: d[~d.index.duplicated()])
        .sort_index()
        .astype(float)
    )

    df.to_parquet(cache_file)
    print(f"  [{symbol}] Saved {len(df):,} bars → {cache_file.name}")
    return df
