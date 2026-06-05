"""
Fee & Leverage Impact Calculator
=================================
Run this before trading to understand the exact cost and minimum
profitable move for any capital / leverage / fee combination.

Usage:
    python analysis/fee_calculator.py
    python analysis/fee_calculator.py --capital 20 --leverage 20 --tp 0.5 --sl 0.25
"""

import argparse


MAKER_FEE = 0.0000   # Binance USDC pairs — limit orders
TAKER_FEE = 0.0004   # Binance USDC pairs — market orders (SL)


def analyse(capital: float, leverage: float, tp_pct: float, sl_pct: float) -> None:
    position   = capital * leverage
    entry_fee  = position * MAKER_FEE        # limit entry  → 0%
    tp_fee     = position * MAKER_FEE        # limit TP     → 0%
    sl_fee     = position * TAKER_FEE        # market SL    → 0.04%

    gross_win  = position * (tp_pct / 100)
    gross_loss = position * (sl_pct / 100)
    net_win    = gross_win  - entry_fee - tp_fee
    net_loss   = gross_loss + entry_fee + sl_fee

    rr         = net_win / net_loss
    breakeven  = 1 / (1 + rr)               # minimum win rate to break even
    min_move   = (entry_fee + tp_fee) / position * 100  # % price needed just to cover entry+TP fees

    print("\n" + "═" * 58)
    print(f"  CONFLUENCE SCALPER — Fee & Leverage Analysis")
    print("═" * 58)
    print(f"  Capital      : ${capital:>10.2f}")
    print(f"  Leverage     : {leverage:>10.0f}x")
    print(f"  Position     : ${position:>10.2f}")
    print(f"  Take-Profit  : {tp_pct:>10.2f}%  (price move)")
    print(f"  Stop-Loss    : {sl_pct:>10.2f}%  (price move)")
    print("─" * 58)
    print(f"  Maker fee (entry + TP) : ${entry_fee + tp_fee:.4f}  ({(entry_fee + tp_fee) / capital * 100:.4f}% of capital)")
    print(f"  Taker fee (SL only)    : ${sl_fee:.4f}  ({sl_fee / capital * 100:.4f}% of capital)")
    print("─" * 58)
    print(f"  Net WIN      : +${net_win:>8.2f}  (+{net_win/capital*100:.2f}% of capital)")
    print(f"  Net LOSS     : -${net_loss:>8.2f}  (-{net_loss/capital*100:.2f}% of capital)")
    print(f"  Risk:Reward  :  1 : {rr:.2f}")
    print(f"  Breakeven WR :  {breakeven*100:.1f}%  (need >{breakeven*100:.1f}% win rate to profit)")
    print(f"  Min move (fee cover) : {min_move:.4f}%  (entry + TP fees only)")
    print("═" * 58)

    # Scenario table
    print("\n  Monthly simulation (20 trading days, 5 trades/day = 100 trades)\n")
    print(f"  {'Win Rate':>10} | {'Net Profit':>12} | {'Return on $'+str(int(capital)):>15}")
    print("  " + "-" * 44)
    for wr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        wins    = int(100 * wr)
        losses  = 100 - wins
        total   = wins * net_win - losses * net_loss
        pct     = total / capital * 100
        flag    = "  << TARGET" if wr >= 0.60 else ""
        print(f"  {wr*100:>9.0f}% | ${total:>11.2f} | {pct:>14.1f}%{flag}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalper fee and leverage calculator")
    parser.add_argument("--capital",  type=float, default=20.0,  help="Starting capital in USDC")
    parser.add_argument("--leverage", type=float, default=10.0,  help="Leverage multiplier")
    parser.add_argument("--tp",       type=float, default=0.5,   help="Take-profit %% (price move)")
    parser.add_argument("--sl",       type=float, default=0.25,  help="Stop-loss %% (price move)")
    args = parser.parse_args()

    analyse(args.capital, args.leverage, args.tp, args.sl)

    # Also show 20x comparison
    if args.leverage == 10.0:
        print("  Comparison at 20x leverage (same TP/SL):")
        analyse(args.capital, 20.0, args.tp, args.sl)


if __name__ == "__main__":
    main()
