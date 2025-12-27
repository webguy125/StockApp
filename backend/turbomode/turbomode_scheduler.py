"""
TurboMode Scheduler
Runs S&P 500 scan at 11 PM nightly (separate from ML automation)
"""

import os
import sys
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from turbomode.overnight_scanner import OvernightScanner

# Setup logging
logger = logging.getLogger('turbomode_scheduler')
logger.setLevel(logging.INFO)

# Scheduler instance (global)
scheduler = None
state_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'turbomode_scheduler_state.json')


def load_state():
    """Load scheduler state from file"""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'enabled': False,
        'schedule_time': '23:00',  # 11 PM default
        'last_run': None,
        'next_run': None
    }


def save_state(state):
    """Save scheduler state to file"""
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def run_overnight_scan():
    """
    Run TurboMode overnight S&P 500 scan
    Scans all 500 symbols and generates top signals
    """
    logger.info("=" * 80)
    logger.info("TURBOMODE - OVERNIGHT S&P 500 SCAN STARTED")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    try:
        # Initialize scanner
        scanner = OvernightScanner(
            db_path="backend/data/turbomode.db",
            ml_db_path="backend/backend/data/advanced_ml_system.db"
        )

        # Run full scan (500 symbols, top 100 of each type)
        results = scanner.scan_all(max_signals_per_type=100)

        # Print summary
        logger.info(f"✅ Scan complete!")
        logger.info(f"   Total scanned: {results['stats']['total_scanned']}")
        logger.info(f"   BUY signals: {results['stats']['buy_count']}")
        logger.info(f"   SELL signals: {results['stats']['sell_count']}")
        logger.info(f"   Saved to DB: BUY={results['stats']['saved_buy']}, SELL={results['stats']['saved_sell']}")

        # Print top 5 signals for review
        scanner.print_top_signals(results, top_n=5)

        # Update last run time
        state = load_state()
        state['last_run'] = datetime.now().isoformat()
        save_state(state)

        logger.info("=" * 80)
        logger.info("TURBOMODE - SCAN COMPLETED")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"❌ TurboMode scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def start_scheduler(schedule_time='23:00'):
    """
    Start the TurboMode background scheduler
    Runs inside Flask application at specified time

    Args:
        schedule_time: Time to run daily (24-hour format, e.g., '23:00' for 11 PM)
    """
    global scheduler

    if scheduler is not None and scheduler.running:
        logger.info("TurboMode scheduler already running")
        return False

    # Parse schedule time
    hour, minute = map(int, schedule_time.split(':'))

    # Create scheduler
    scheduler = BackgroundScheduler(daemon=True)

    # Add job to run daily at 11 PM
    scheduler.add_job(
        run_overnight_scan,
        trigger=CronTrigger(hour=hour, minute=minute),
        id='turbomode_overnight_scan',
        name='TurboMode - S&P 500 Overnight Scan',
        replace_existing=True
    )

    # Start scheduler
    scheduler.start()

    # Save state
    state = {
        'enabled': True,
        'schedule_time': schedule_time,
        'last_run': None,
        'next_run': scheduler.get_job('turbomode_overnight_scan').next_run_time.isoformat() if scheduler.get_job('turbomode_overnight_scan') else None
    }
    save_state(state)

    logger.info(f"✅ TurboMode Scheduler STARTED - Will run daily at {schedule_time}")
    logger.info(f"   Next run: {state['next_run']}")

    return True


def stop_scheduler():
    """Stop the TurboMode background scheduler"""
    global scheduler

    if scheduler is None or not scheduler.running:
        logger.info("TurboMode scheduler not running")
        return False

    scheduler.shutdown()
    scheduler = None

    # Save state
    state = load_state()
    state['enabled'] = False
    state['next_run'] = None
    save_state(state)

    logger.info("✅ TurboMode Scheduler STOPPED")

    return True


def get_status():
    """Get current TurboMode scheduler status"""
    global scheduler

    state = load_state()

    is_running = scheduler is not None and scheduler.running

    if is_running and scheduler.get_job('turbomode_overnight_scan'):
        next_run = scheduler.get_job('turbomode_overnight_scan').next_run_time
        state['next_run'] = next_run.isoformat() if next_run else None

    return {
        'enabled': is_running,
        'schedule_time': state.get('schedule_time', '23:00'),
        'last_run': state.get('last_run'),
        'next_run': state.get('next_run'),
        'scheduler_running': is_running
    }


def init_turbomode_scheduler():
    """
    Initialize TurboMode scheduler on Flask startup
    AUTO-STARTS by default at 11 PM
    """
    state = load_state()

    # If this is first run, enable by default at 11 PM
    if 'enabled' not in state:
        logger.info("First run detected - ENABLING TurboMode scheduler at 23:00 (11 PM)")
        start_scheduler(schedule_time='23:00')
        return

    # If scheduler was previously enabled, restore it
    if state.get('enabled', False):
        logger.info("Restoring TurboMode scheduler from previous state...")
        start_scheduler(schedule_time=state.get('schedule_time', '23:00'))
    else:
        logger.info("TurboMode scheduler was manually disabled - staying off")


if __name__ == '__main__':
    # Test the scheduler
    print("Testing TurboMode Scheduler...")
    print("=" * 60)

    # Start scheduler
    init_turbomode_scheduler()

    # Get status
    status = get_status()
    print(f"\nScheduler Status:")
    print(f"  Enabled: {status['enabled']}")
    print(f"  Schedule Time: {status['schedule_time']}")
    print(f"  Next Run: {status['next_run']}")
    print(f"  Last Run: {status['last_run']}")

    print("\n[OK] TurboMode Scheduler initialized!")
    print("Press Ctrl+C to exit...")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        stop_scheduler()
        print("Goodbye!")
