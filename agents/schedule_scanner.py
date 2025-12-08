"""
Scanner Automation Script
Schedules the comprehensive scanner to run nightly at midnight UTC

Usage:
    python schedule_scanner.py              # Run scheduler (keeps running)
    python schedule_scanner.py --now        # Run scan immediately
    python schedule_scanner.py --test       # Test scan with limited symbols
"""

import schedule
import time
import sys
import os
from datetime import datetime
from pathlib import Path
import argparse
import subprocess

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def run_comprehensive_scan():
    """
    Execute the comprehensive scanner
    """
    print(f"\n{'='*80}")
    print(f"🕐 Scheduled Scan Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*80}\n")

    try:
        # Import and run scanner
        from comprehensive_scanner import ComprehensiveScanner

        scanner = ComprehensiveScanner()
        scan_results = scanner.run_comprehensive_scan()
        scanner.integrate_with_learning_loop(scan_results)

        print(f"\n✅ Scheduled scan completed successfully")
        print(f"   Next run: {schedule.next_run()}")

        # Optionally trigger fusion agent
        try:
            print(f"\n🔄 Triggering fusion agent...")
            fusion_script = Path(__file__).parent / "fusion_agent.py"
            result = subprocess.run(
                [sys.executable, str(fusion_script)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print("   ✅ Fusion agent completed")
            else:
                print(f"   ⚠️  Fusion agent error: {result.stderr}")

        except Exception as e:
            print(f"   ⚠️  Could not trigger fusion agent: {e}")

        return True

    except Exception as e:
        print(f"\n❌ Scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_now():
    """
    Run scan immediately (for testing)
    """
    print("🚀 Running scan immediately...")
    return run_comprehensive_scan()


def run_scheduler():
    """
    Run the scheduler loop (keeps running)
    """
    print("="*80)
    print("📅 SCANNER AUTOMATION SCHEDULER")
    print("="*80)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scheduled time: Midnight UTC (00:00)")
    print("\nScheduling comprehensive scanner to run nightly...")
    print("Press Ctrl+C to stop\n")

    # Schedule the scan for midnight UTC
    schedule.every().day.at("00:00").do(run_comprehensive_scan)

    print(f"✅ Scheduler started")
    print(f"   Next run: {schedule.next_run()}")
    print(f"\n⏳ Waiting for scheduled time...\n")

    # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n\n⚠️  Scheduler stopped by user")
        sys.exit(0)


def main():
    """
    Main entry point with command-line argument handling
    """
    parser = argparse.ArgumentParser(
        description='Comprehensive Scanner Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python schedule_scanner.py              # Run scheduler (nightly at midnight UTC)
  python schedule_scanner.py --now        # Run scan immediately
  python schedule_scanner.py --test       # Test run with sample data

For Windows Task Scheduler:
  Create a task that runs daily at midnight:
  Program: C:\\StockApp\\venv\\Scripts\\python.exe
  Arguments: C:\\StockApp\\agents\\schedule_scanner.py --now
  Start in: C:\\StockApp\\agents

For Linux cron:
  Add to crontab:
  0 0 * * * cd /path/to/StockApp/agents && /path/to/python schedule_scanner.py --now
        """
    )

    parser.add_argument('--now', action='store_true',
                       help='Run scan immediately instead of scheduling')
    parser.add_argument('--test', action='store_true',
                       help='Run test scan with limited symbols')

    args = parser.parse_args()

    if args.test:
        print("🧪 Test mode - running limited scan...")
        # Set environment variable to limit symbols
        os.environ['SCANNER_TEST_MODE'] = '1'
        return run_now()

    elif args.now:
        return run_now()

    else:
        run_scheduler()


if __name__ == "__main__":
    main()
