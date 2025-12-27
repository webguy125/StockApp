"""
CLI Interface for ML Trading System
Run manual scans and view results
"""

import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system.core.trading_system import TradingSystem


def main():
    parser = argparse.ArgumentParser(description='ML Trading System - Run Daily Scan')
    parser.add_argument('--max-stocks', type=int, default=50,
                       help='Maximum number of stocks to analyze (default: 50)')
    parser.add_argument('--no-crypto', action='store_true',
                       help='Exclude cryptocurrencies from scan')
    parser.add_argument('--stats', action='store_true',
                       help='Show performance statistics only')

    args = parser.parse_args()

    # Initialize system
    print("Initializing ML Trading System...")
    system = TradingSystem()

    if args.stats:
        # Show statistics only
        print("\n" + "=" * 60)
        print("SYSTEM PERFORMANCE STATISTICS")
        print("=" * 60 + "\n")

        stats = system.get_performance_stats()

        print(f"Total Trades:     {stats['total_trades']}")
        print(f"Wins:             {stats['wins']}")
        print(f"Losses:           {stats['losses']}")
        print(f"Win Rate:         {stats['win_rate']:.2f}%")
        print(f"Avg P/L:          ${stats['avg_profit_loss']:.2f}")
        print(f"Total P/L:        ${stats['total_profit_loss']:.2f}")
        print(f"Profit Factor:    {stats['profit_factor']:.2f}")
        print("\n" + "=" * 60 + "\n")

    else:
        # Run full scan
        signals = system.run_daily_scan(
            max_stocks=args.max_stocks,
            include_crypto=not args.no_crypto
        )

        # Show top 10 signals
        if signals:
            print("\n🏆 TOP 10 TRADING SIGNALS:\n")
            print(f"{'Rank':<6} {'Symbol':<10} {'Score':<8} {'Conf':<8} {'Direction':<10} {'ML':<10} {'Price':<10}")
            print("-" * 70)

            for signal in signals[:10]:
                print(f"{signal['rank']:<6} {signal['symbol']:<10} {signal['score']:.3f}    {signal['confidence']:.3f}    {signal['direction']:<10} {signal['ml_prediction']:<10} ${signal['price']:<10.2f}")

            print("\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Scan interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
