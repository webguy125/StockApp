"""
Master Market Data Scheduler
Schedules nightly ingestion of market data at 10:45 PM

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Location: C:\StockApp\master_market_data\market_data.db

This runs nightly to keep the Master Market Data DB updated.
"""

import os
import sys
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# Setup logging
logger = logging.getLogger('master_data_scheduler')
logger.setLevel(logging.INFO)

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from master_market_data.ingest_market_data import MarketDataIngestion
from backend.advanced_ml.config.core_symbols import get_all_core_symbols, CRYPTO_SYMBOLS

# Scheduler instance (global)
scheduler = None
state_file = os.path.join(os.path.dirname(__file__), 'master_data_scheduler_state.json')


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
        'schedule_time': '22:45',  # 10:45 PM default
        'last_run': None,
        'next_run': None
    }


def save_state(state):
    """Save scheduler state to file"""
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def run_nightly_ingestion():
    """
    Run nightly Master Market Data ingestion
    Fetches latest data for all core symbols
    """
    logger.info("=" * 80)
    logger.info("MASTER MARKET DATA - NIGHTLY INGESTION")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    try:
        # Initialize ingestion
        ingestion = MarketDataIngestion()

        # Ingest data for all core symbols (stocks + crypto)
        # Using 5d period to get recent data and update existing records
        all_symbols = get_all_core_symbols() + CRYPTO_SYMBOLS

        results = ingestion.ingest_multiple_symbols(
            all_symbols,
            period='5d',  # Last 5 days to catch up
            timeframe='1d'
        )

        # Print summary
        logger.info(f"[OK] Ingestion complete!")
        logger.info(f"   Symbols processed: {results['total_symbols']}")
        logger.info(f"   Successful: {results['successful']}")
        logger.info(f"   Failed: {results['failed']}")
        logger.info(f"   Candles ingested: {results['total_candles']:,}")

        # Update last run time
        state = load_state()
        state['last_run'] = datetime.now().isoformat()
        save_state(state)

        logger.info("=" * 80)
        logger.info("MASTER MARKET DATA - INGESTION COMPLETED")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"[ERROR] Master data ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def start_scheduler(schedule_time='22:45'):
    """
    Start the Master Data background scheduler
    Runs inside Flask application at specified time

    Args:
        schedule_time: Time to run daily (24-hour format, e.g., '22:45' for 10:45 PM)
    """
    global scheduler

    if scheduler is not None and scheduler.running:
        logger.info("Master Data scheduler already running")
        return False

    # Parse schedule time
    hour, minute = map(int, schedule_time.split(':'))

    # Create scheduler
    scheduler = BackgroundScheduler(daemon=True)

    # Add job to run daily at 10:45 PM
    scheduler.add_job(
        run_nightly_ingestion,
        trigger=CronTrigger(hour=hour, minute=minute),
        id='master_data_nightly_ingestion',
        name='Master Market Data - Nightly Ingestion',
        replace_existing=True
    )

    # Start scheduler
    scheduler.start()

    # Save state
    state = {
        'enabled': True,
        'schedule_time': schedule_time,
        'last_run': None,
        'next_run': scheduler.get_job('master_data_nightly_ingestion').next_run_time.isoformat() if scheduler.get_job('master_data_nightly_ingestion') else None
    }
    save_state(state)

    logger.info(f"[OK] Master Data Scheduler STARTED - Will run daily at {schedule_time}")
    logger.info(f"   Next run: {state['next_run']}")

    return True


def stop_scheduler():
    """Stop the Master Data background scheduler"""
    global scheduler

    if scheduler is None or not scheduler.running:
        logger.info("Master Data scheduler not running")
        return False

    scheduler.shutdown()
    scheduler = None

    # Save state
    state = load_state()
    state['enabled'] = False
    state['next_run'] = None
    save_state(state)

    logger.info("[OK] Master Data Scheduler STOPPED")

    return True


def get_status():
    """Get current Master Data scheduler status"""
    global scheduler

    state = load_state()

    is_running = scheduler is not None and scheduler.running

    if is_running and scheduler.get_job('master_data_nightly_ingestion'):
        next_run = scheduler.get_job('master_data_nightly_ingestion').next_run_time
        state['next_run'] = next_run.isoformat() if next_run else None

    return {
        'enabled': is_running,
        'schedule_time': state.get('schedule_time', '22:45'),
        'last_run': state.get('last_run'),
        'next_run': state.get('next_run'),
        'scheduler_running': is_running
    }


def init_master_data_scheduler():
    """
    Initialize Master Data scheduler on Flask startup
    AUTO-STARTS by default at 10:45 PM
    """
    state = load_state()

    # If this is first run, enable by default at 10:45 PM
    if 'enabled' not in state:
        logger.info("First run detected - ENABLING Master Data scheduler at 22:45 (10:45 PM)")
        start_scheduler(schedule_time='22:45')
        return

    # If scheduler was previously enabled, restore it
    if state.get('enabled', False):
        logger.info("Restoring Master Data scheduler from previous state...")
        start_scheduler(schedule_time=state.get('schedule_time', '22:45'))
    else:
        logger.info("Master Data scheduler was manually disabled - staying off")


if __name__ == '__main__':
    # Test the scheduler
    print("Testing Master Data Scheduler...")
    print("=" * 60)

    # Start scheduler
    init_master_data_scheduler()

    # Get status
    status = get_status()
    print(f"\nScheduler Status:")
    print(f"  Enabled: {status['enabled']}")
    print(f"  Schedule Time: {status['schedule_time']}")
    print(f"  Next Run: {status['next_run']}")
    print(f"  Last Run: {status['last_run']}")

    print("\n[OK] Master Data Scheduler initialized!")
    print("Press Ctrl+C to exit...")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        stop_scheduler()
        print("Goodbye!")
