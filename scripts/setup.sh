#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Confluence Scalper — one-command local setup
# Run from the repo root: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo ""
echo "═══ Confluence Scalper Setup ═══"
echo ""

# ── 1. Create virtual environment ────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  $PYTHON -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists."
fi

source "$VENV_DIR/bin/activate"

# ── 2. Upgrade pip quietly ────────────────────────────────────────────────────
pip install -q --upgrade pip

# ── 3. Install backtester dependencies ───────────────────────────────────────
echo "Installing backtester dependencies (ccxt, pandas, numpy, rich) ..."
pip install -q -r requirements.txt

# ── 4. Install Freqtrade ──────────────────────────────────────────────────────
echo "Installing Freqtrade ..."
pip install -q "freqtrade>=2024.1"

echo ""
echo "✓ Setup complete."
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Activate the virtual env:"
echo "       source $VENV_DIR/bin/activate"
echo ""
echo "  2. Run the Python backtester on real Binance data (no API key needed):"
echo "       python run_backtest.py --months 3"
echo ""
echo "  3. Run the parameter optimizer:"
echo "       python run_optimize.py --months 3"
echo ""
echo "  4. Download 1m data for Freqtrade backtesting:"
echo "       freqtrade download-data -c freqtrade/config.json --timeframe 1m --days 90"
echo ""
echo "  5. Run Freqtrade backtesting:"
echo "       freqtrade backtesting -c freqtrade/config.json --strategy ConfluenceScalper"
echo ""
echo "  6. Add your Binance API keys to freqtrade/config_live.json"
echo "     (open the file and replace YOUR_BINANCE_API_KEY / YOUR_BINANCE_API_SECRET)"
echo "     Required permissions: Enable Reading + Enable Futures (NO withdrawals)"
echo ""
echo "  7. Set up Telegram (optional but recommended for trade alerts):"
echo "     - Create a bot via @BotFather, get token"
echo "     - Get your chat_id from @userinfobot"
echo "     - Fill in freqtrade/config_live.json telegram section"
echo ""
echo "  8. Start dry-run (paper trading — no real money at risk):"
echo "       freqtrade trade -c freqtrade/config_live.json --strategy ConfluenceScalper"
echo ""
echo "  9. Monitor for 2+ weeks. Only set dry_run: false after consistent profits."
echo ""
echo "⚠  SECURITY: Never share API keys in chat, email, or commit them to git."
echo "   Restrict keys to Futures trading only — disable withdrawals."
echo ""
