"""
Automated Scheduler
Runs the complete ML system automatically:
1. Daily scan (generates signals)
2. Automated learner (simulates trades, tracks outcomes, retrains)
"""

import schedule
import time
from datetime import datetime
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.trading_system import TradingSystem
from trading_system.automated_learner import AutomatedLearner


def run_daily_cycle():
    """Run complete daily cycle: scan → learn → retrain"""
    print("\n" + "=" * 80)
    print("AUTOMATED ML SYSTEM - DAILY CYCLE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    try:
        # Step 1: Run scan (ENTIRE S&P 500!)
        print("[SCAN] Step 1: Running market scan (ENTIRE S&P 500 + crypto)...")
        system = TradingSystem()
        signals = system.run_daily_scan(max_stocks=500, include_crypto=True)
        print(f"[OK] Generated {len(signals)} signals\n")

        # Step 2: Run automated learner (OPTIONS TRADING SETTINGS)
        print("[LEARN] Step 2: Running automated learner...")
        learner = AutomatedLearner(
            hold_period_days=14,        # 14 days = 2 weeks (good for monthly options)
            win_threshold_pct=10.0,     # 10% gain target (options move bigger!)
            loss_threshold_pct=-5.0,    # 5% max loss (tight stop for options)
            max_simulated_positions=30  # 30 positions for fast learning
        )
        learner.run_cycle()

        print("\n" + "=" * 80)
        print("DAILY CYCLE COMPLETE")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Error in daily cycle: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main scheduler loop"""
    import argparse

    parser = argparse.ArgumentParser(description='Automated ML Trading System Scheduler')
    parser.add_argument('--time', type=str, default='18:00',
                       help='Time to run daily (HH:MM format, default: 18:00 / 6 PM)')
    parser.add_argument('--now', action='store_true',
                       help='Run immediately instead of scheduling')

    args = parser.parse_args()

    if args.now:
        # Run immediately
        print("Running immediately...")
        run_daily_cycle()
    else:
        # Schedule daily run
        schedule.every().day.at(args.time).do(run_daily_cycle)

        print("\n" + "=" * 80)
        print("AUTOMATED ML SYSTEM - SCHEDULER STARTED")
        print("=" * 80)
        print(f"Daily run scheduled at: {args.time}")
        print(f"Next run: {schedule.next_run()}")
        print("\nWhat happens automatically:")
        print("  1. Market scan (generates 50+ signals)")
        print("  2. Automated learner simulates trades")
        print("  3. Checks outcomes of existing positions")
        print("  4. Marks wins/losses automatically")
        print("  5. Retrains model when enough data collected")
        print("\nPress Ctrl+C to stop the scheduler")
        print("=" * 80 + "\n")

        # Run scheduler loop
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Scheduler stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
