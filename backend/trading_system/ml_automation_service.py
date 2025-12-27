"""
ML Automation Service
Built-in scheduler that runs inside Flask application
Works on any platform (Windows, Linux, cloud hosting)
NO EXTERNAL DEPENDENCIES (no Windows Task Scheduler needed)
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

from trading_system.core.trading_system import TradingSystem
from trading_system.automated_learner import AutomatedLearner

# TurboMode scanner
try:
    sys.path.insert(0, os.path.join(parent_dir, '..'))
    from backend.turbomode.overnight_scanner import OvernightScanner
    TURBOMODE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TurboMode not available: {e}")
    TURBOMODE_AVAILABLE = False

# Setup logging
logger = logging.getLogger('ml_automation')
logger.setLevel(logging.INFO)

# Scheduler instance (global)
scheduler = None
automation_state_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_automation_state.json')


def load_automation_state():
    """Load automation state from file"""
    if os.path.exists(automation_state_file):
        try:
            with open(automation_state_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'enabled': False,
        'schedule_time': '18:00',  # 6 PM default
        'last_run': None,
        'next_run': None
    }


def save_automation_state(state):
    """Save automation state to file"""
    os.makedirs(os.path.dirname(automation_state_file), exist_ok=True)
    with open(automation_state_file, 'w') as f:
        json.dump(state, f, indent=2)


def run_daily_cycle():
    """
    Run complete ML cycle:
    1. Scan market
    2. Simulate trades on top signals
    3. Check outcomes of existing positions
    4. Retrain model if needed
    """
    logger.info("=" * 80)
    logger.info("ML AUTOMATION - DAILY CYCLE STARTED")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    try:
        # Step 1: Market Scan (ENTIRE S&P 500!)
        logger.info("Step 1/2: Running market scan (ENTIRE S&P 500 + crypto)...")
        system = TradingSystem()
        signals = system.run_daily_scan(max_stocks=500, include_crypto=True)
        logger.info(f"✅ Generated {len(signals)} signals")

        # Step 2: Automated Learning (OPTIONS TRADING SETTINGS)
        logger.info("Step 2/2: Running automated learner...")
        learner = AutomatedLearner(
            hold_period_days=14,        # 14 days = 2 weeks (good for monthly options)
            win_threshold_pct=10.0,     # 10% gain target (options move bigger!)
            loss_threshold_pct=-5.0,    # 5% max loss (tight stop for options)
            max_simulated_positions=30  # 30 positions for fast learning
        )
        learner.run_cycle()
        logger.info("✅ Learning cycle complete")

        # Update last run time
        state = load_automation_state()
        state['last_run'] = datetime.now().isoformat()
        save_automation_state(state)

        logger.info("=" * 80)
        logger.info("ML AUTOMATION - DAILY CYCLE COMPLETED")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"❌ Daily cycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def start_automation(schedule_time='18:00'):
    """
    Start the background scheduler
    Runs inside Flask application - no external dependencies

    Args:
        schedule_time: Time to run daily (24-hour format, e.g., '18:00' for 6 PM)
    """
    global scheduler

    if scheduler is not None and scheduler.running:
        logger.info("Automation already running")
        return False

    # Parse schedule time
    hour, minute = map(int, schedule_time.split(':'))

    # Create scheduler
    scheduler = BackgroundScheduler(daemon=True)

    # Add job to run daily at specified time
    scheduler.add_job(
        run_daily_cycle,
        trigger=CronTrigger(hour=hour, minute=minute),
        id='ml_daily_cycle',
        name='ML Trading System - Daily Cycle',
        replace_existing=True
    )

    # Start scheduler
    scheduler.start()

    # Save state
    state = {
        'enabled': True,
        'schedule_time': schedule_time,
        'last_run': None,
        'next_run': scheduler.get_job('ml_daily_cycle').next_run_time.isoformat() if scheduler.get_job('ml_daily_cycle') else None
    }
    save_automation_state(state)

    logger.info(f"✅ ML Automation STARTED - Will run daily at {schedule_time}")
    logger.info(f"Next run: {state['next_run']}")

    return True


def stop_automation():
    """Stop the background scheduler"""
    global scheduler

    if scheduler is None or not scheduler.running:
        logger.info("Automation not running")
        return False

    scheduler.shutdown()
    scheduler = None

    # Save state
    state = load_automation_state()
    state['enabled'] = False
    state['next_run'] = None
    save_automation_state(state)

    logger.info("✅ ML Automation STOPPED")

    return True


def get_automation_status():
    """Get current automation status"""
    global scheduler

    state = load_automation_state()

    is_running = scheduler is not None and scheduler.running

    if is_running and scheduler.get_job('ml_daily_cycle'):
        next_run = scheduler.get_job('ml_daily_cycle').next_run_time
        state['next_run'] = next_run.isoformat() if next_run else None

    return {
        'enabled': is_running,
        'schedule_time': state.get('schedule_time', '18:00'),
        'last_run': state.get('last_run'),
        'next_run': state.get('next_run'),
        'scheduler_running': is_running
    }


def initialize_automation():
    """
    Initialize automation on Flask startup
    AUTO-STARTS by default - no clicking needed!
    """
    state = load_automation_state()

    # If this is first run (no state file), enable by default
    if 'enabled' not in state:
        logger.info("First run detected - ENABLING automation by default at 18:00 (6 PM)")
        start_automation(schedule_time='18:00')
        return

    # If automation was previously enabled, restore it
    if state.get('enabled', False):
        logger.info("Restoring automation from previous state...")
        start_automation(schedule_time=state.get('schedule_time', '18:00'))
    else:
        logger.info("Automation was manually disabled - staying off (click 'Start Auto' to enable)")


# Auto-initialize when imported (Flask will call this on startup)
def init_automation_service():
    """Call this from Flask app startup"""
    logger.info("ML Automation Service initializing...")
    initialize_automation()
    logger.info("ML Automation Service ready")
