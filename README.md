# Confluence Scalper v1

A multi-factor crypto scalping strategy for Binance Futures (USDC pairs).

**Capital**: $20 starting | **Leverage**: 10–30x | **Pairs**: BTC/USDC, ETH/USDC, SOL/USDC  
**Timeframe**: 1m | **Style**: Long + Short scalps

---

## Strategy Logic

All six layers must agree before an entry fires:

| Layer | Long | Short |
|-------|------|-------|
| **EMA Trend** | EMA 20 > EMA 50 | EMA 20 < EMA 50 |
| **Supertrend** | Bullish (green) | Bearish (red) |
| **VWAP** | Price above VWAP | Price below VWAP |
| **Market Structure** | Higher highs + higher lows | Lower highs + lower lows |
| **Momentum (RSI + MACD)** | RSI 40–65, MACD histogram rising | RSI 35–60, MACD histogram falling |
| **Volume** | Volume > 1.2× SMA | Volume > 1.2× SMA |

**Orders:**
- Entry: **Limit** (0% maker fee on Binance USDC pairs)
- Take-profit: **Limit** (0% maker fee)
- Stop-loss: **Stop-market** (0.04% taker fee)
- Breakeven trail: SL moves to entry after 1× ATR profit locked in
- Time stop: Flat after 30 bars if TP/SL not reached

---

## Fee Advantage

| Order | Type | Fee |
|-------|------|-----|
| Entry | Limit | **0%** |
| TP hit | Limit | **0%** |
| SL hit | Market | **0.04%** |

With a 60%+ win rate: ~60% of trades cost **0% total fees**, only 40% incur the 0.04% taker fee.  
Average round-trip cost ≈ **0.016%** vs 0.08% with all market orders — **5× cheaper**.

---

## Phase 1 — TradingView (Pine Script v6)

### How to use

1. Open [TradingView](https://tradingview.com) → any USDC chart (e.g. BTCUSDC, 1m)
2. Click **Pine Script Editor** at the bottom
3. Paste the contents of `pine_scripts/confluence_scalper_v1.pine`
4. Click **Add to chart**
5. Open the **Strategy Tester** tab

### Recommended test setup

| Setting | Value |
|---------|-------|
| Symbol | BTCUSDC (Binance) |
| Timeframe | 1m |
| Initial Capital | $20 |
| Order Size | 100% of equity |
| Commission | 0.04% (per order) |
| Slippage | 1 tick |
| Test range | Last 3–6 months |

### What to look for

| Metric | Target |
|--------|--------|
| Win rate | ≥ 60% |
| Profit factor | ≥ 1.5 |
| Max drawdown | ≤ 20% of capital |

---

## Phase 2 — Python Backtesting (Freqtrade)

### Installation

```bash
# Install Freqtrade (Docker is easiest)
docker pull freqtradeorg/freqtrade:stable

# Or pip install
python -m venv .venv
source .venv/bin/activate
pip install freqtrade

# Install pandas_ta (indicators library)
pip install pandas_ta
```

### Download historical data

```bash
freqtrade download-data \
  --pairs BTC/USDC:USDC ETH/USDC:USDC SOL/USDC:USDC \
  --timeframe 1m \
  --timerange 20240101-20241231 \
  --exchange binance \
  --trading-mode futures
```

### Run backtesting

```bash
freqtrade backtesting \
  --strategy ConfluenceScalper \
  --config freqtrade/config.json \
  --pairs BTC/USDC:USDC ETH/USDC:USDC SOL/USDC:USDC \
  --timeframe 1m \
  --timerange 20240601-20241231
```

### Run hyperopt (parameter optimization)

```bash
freqtrade hyperopt \
  --strategy ConfluenceScalper \
  --config freqtrade/config.json \
  --hyperopt-loss SharpeHyperOptLoss \
  --pairs BTC/USDC:USDC \
  --timeframe 1m \
  --timerange 20240601-20241231 \
  --epochs 200
```

---

## Phase 3 — Live / Dry-Run

### Step 1: Create Binance API keys

1. Log in to Binance → Profile → API Management
2. Create a new API key (label: "ConfluenceScalper")
3. Permissions needed: **Read + Futures trading** (no withdrawals)
4. Restrict to your IP address for security
5. Copy key + secret

### Step 2: Configure

Edit `freqtrade/config_live.json`:
```json
"key": "YOUR_BINANCE_API_KEY",
"secret": "YOUR_BINANCE_API_SECRET"
```

For Telegram alerts (optional):
1. Message `@BotFather` on Telegram → `/newbot` → copy token
2. Message `@userinfobot` → copy your chat_id
3. Fill in `telegram.token` and `telegram.chat_id` in `config_live.json`

### Step 3: Run dry-run (paper trading — NO real money)

```bash
freqtrade trade \
  --strategy ConfluenceScalper \
  --config freqtrade/config_live.json \
  --dry-run
```

Run dry-run for **at least 1–2 weeks** and verify results match backtesting before going live.

### Step 4: Go live

Change `"dry_run": false` in `config_live.json`, then:

```bash
freqtrade trade \
  --strategy ConfluenceScalper \
  --config freqtrade/config_live.json
```

---

## Fee Calculator

Run to see exact P&L at any capital/leverage/TP/SL combination:

```bash
python analysis/fee_calculator.py
# or custom values:
python analysis/fee_calculator.py --capital 20 --leverage 20 --tp 0.5 --sl 0.25
```

---

## Risk Warnings

- **Liquidation**: At 20x leverage, a 5% adverse move without a stop = 100% loss. The ATR-based SL is mandatory.
- **Funding rates**: Binance Futures charges every 8 hours. The 30-bar time stop keeps positions short.
- **Limit fills**: In very fast markets, limit entries may not fill. Monitor unfilled orders.
- **Overfitting**: Always validate on out-of-sample data (test on dates not used during design).
- **Start small**: Keep leverage at 10x until you have 50+ live trades confirming the strategy.

---

## Project Structure

```
Codex/
├── pine_scripts/
│   └── confluence_scalper_v1.pine     # TradingView Pine Script v6
├── freqtrade/
│   ├── strategies/
│   │   └── ConfluenceScalper.py       # Python strategy (Freqtrade)
│   ├── config.json                    # Backtesting / dry-run config
│   └── config_live.json               # Live trading config
├── analysis/
│   └── fee_calculator.py              # Fee + leverage impact calculator
└── README.md
```
