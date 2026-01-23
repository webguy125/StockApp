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

from turbomode.core_engine.overnight_scanner import OvernightScanner
from turbomode.outcome_tracker import track_signal_outcomes
from turbomode.training_sample_generator import generate_training_samples_from_outcomes
from turbomode.automated_retrainer import automated_model_retraining
from turbomode.meta_retrain import maybe_retrain_meta
from turbomode.task_monitor import log_task_result, send_daily_report
import time
import subprocess

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
    Run TurboMode overnight scan on 40 curated stocks
    Uses GPU-accelerated 8-model ensemble with 179 features
    """
    logger.info("=" * 80)
    logger.info("TURBOMODE - OVERNIGHT SCAN (40 CURATED STOCKS)")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # Initialize scanner (scans 40 curated stocks from CORE_SYMBOLS)
        scanner = OvernightScanner(
            db_path=r"C:\StockApp\backend\data\turbomode.db",
            model_path=r"C:\StockApp\backend\turbomode\models\trained"
        )

        # Run full scan (40 curated stocks, top 100 signals of each type)
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

        # Log success
        duration = time.time() - start_time
        log_task_result('overnight_scan', True, duration=duration)

        return True

    except Exception as e:
        logger.error(f"❌ TurboMode scan failed: {e}")
        import traceback
        traceback.print_exc()

        # Log failure
        duration = time.time() - start_time
        log_task_result('overnight_scan', False, error_msg=str(e), duration=duration)

        return False


def run_outcome_tracker_monitored():
    """Wrapper for outcome tracker with monitoring"""
    start_time = time.time()
    try:
        success = track_signal_outcomes()
        duration = time.time() - start_time
        log_task_result('outcome_tracker', success, duration=duration)
        return success
    except Exception as e:
        duration = time.time() - start_time
        log_task_result('outcome_tracker', False, error_msg=str(e), duration=duration)
        return False


def run_sample_generator_monitored():
    """Wrapper for training sample generator with monitoring"""
    start_time = time.time()
    try:
        success = generate_training_samples_from_outcomes()
        duration = time.time() - start_time
        log_task_result('sample_generator', success, duration=duration)
        return success
    except Exception as e:
        duration = time.time() - start_time
        log_task_result('sample_generator', False, error_msg=str(e), duration=duration)
        return False


def run_monthly_retrain_monitored():
    """Wrapper for monthly retraining with monitoring"""
    start_time = time.time()
    try:
        success = automated_model_retraining()
        duration = time.time() - start_time
        log_task_result('monthly_retrain', success, duration=duration)
        return success
    except Exception as e:
        duration = time.time() - start_time
        log_task_result('monthly_retrain', False, error_msg=str(e), duration=duration)
        return False


def run_meta_retrain_monitored():
    """Wrapper for meta-learner retraining with monitoring"""
    start_time = time.time()
    try:
        success = maybe_retrain_meta()
        duration = time.time() - start_time
        log_task_result('meta_retrain', success, duration=duration)
        return success
    except Exception as e:
        duration = time.time() - start_time
        log_task_result('meta_retrain', False, error_msg=str(e), duration=duration)
        return False


def run_backtest_generation_monitored():
    """
    Wrapper for backtest data generation with monitoring
    Regenerates training data for all 230 stocks + 3 crypto (10 years history)
    Automatically runs feature extraction after backtest completes
    """
    start_time = time.time()
    try:
        logger.info("=" * 80)
        logger.info("BACKTEST GENERATION - MONTHLY DATA REFRESH")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Run generate_backtest_data.py (which now calls extract_features.py automatically)
        script_path = os.path.join(current_dir, 'core_engine', 'generate_backtest_data.py')
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=current_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("[OK] Backtest generation completed successfully!")
            duration = time.time() - start_time
            log_task_result('backtest_generation', True, duration=duration)
            return True
        else:
            logger.error(f"[ERROR] Backtest generation failed with code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            duration = time.time() - start_time
            log_task_result('backtest_generation', False, error_msg=result.stderr, duration=duration)
            return False

    except Exception as e:
        logger.error(f"[ERROR] Backtest generation failed: {e}")
        import traceback
        traceback.print_exc()
        duration = time.time() - start_time
        log_task_result('backtest_generation', False, error_msg=str(e), duration=duration)
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

    # Add outcome tracker job (Daily at 2 AM)
    scheduler.add_job(
        run_outcome_tracker_monitored,
        trigger=CronTrigger(hour=2, minute=0),
        id='turbomode_outcome_tracker',
        name='TurboMode - Outcome Tracker',
        replace_existing=True
    )

    # Add training sample generator (Sunday at 3 AM)
    scheduler.add_job(
        run_sample_generator_monitored,
        trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),
        id='turbomode_sample_generator',
        name='TurboMode - Training Sample Generator',
        replace_existing=True
    )

    # Add backtest data generation (1st of month at 3 AM - 1 hour before model retraining)
    scheduler.add_job(
        run_backtest_generation_monitored,
        trigger=CronTrigger(day=1, hour=3, minute=0),
        id='turbomode_backtest_generation',
        name='TurboMode - Monthly Backtest Data Generation',
        replace_existing=True
    )

    # Add automated retrainer (1st of month at 4 AM)
    scheduler.add_job(
        run_monthly_retrain_monitored,
        trigger=CronTrigger(day=1, hour=4, minute=0),
        id='turbomode_monthly_retrain',
        name='TurboMode - Monthly Model Retraining',
        replace_existing=True
    )

    # Add meta-learner retraining (Every 6 weeks on Sunday at 11:45 PM)
    # Calculate first run time: 6 weeks from Jan 11, 2026
    from datetime import datetime, timedelta
    first_run = datetime(2026, 1, 11, 23, 45) + timedelta(weeks=6)

    scheduler.add_job(
        run_meta_retrain_monitored,
        trigger=CronTrigger(
            day_of_week='sun',
            hour=23,
            minute=45,
            start_date=first_run
        ),
        id='turbomode_meta_retrain',
        name='TurboMode - Meta-Learner Retraining (6-weekly)',
        replace_existing=True,
        misfire_grace_time=3600  # 1 hour grace period
    )

    # Add daily SMS report (Daily at 8:00 AM)
    scheduler.add_job(
        send_daily_report,
        trigger=CronTrigger(hour=8, minute=0),
        id='turbomode_daily_report',
        name='TurboMode - Daily Email Report',
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

    logger.info(f"[OK] TurboMode Scheduler STARTED")
    logger.info(f"   Overnight Scan: Daily at {schedule_time}")
    logger.info(f"   Outcome Tracker: Daily at 02:00")
    logger.info(f"   Sample Generator: Sunday at 03:00")
    logger.info(f"   Backtest Generation: 1st of month at 03:00")
    logger.info(f"   Model Retraining: 1st of month at 04:00")
    logger.info(f"   Meta-Learner Retrain: Every 6 weeks, Sunday at 23:45 (first run: {first_run.strftime('%Y-%m-%d')})")
    logger.info(f"   Daily Email Report: Daily at 08:00")
    logger.info(f"   Next scan: {state['next_run']}")

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

    logger.info("[OK] TurboMode Scheduler STOPPED")

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
