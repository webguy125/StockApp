"""
Options Scanner Scheduler

Runs the options scanner every 5-10 minutes during market hours
to keep the options_intel.db cache fresh.
"""
import os
import sys
import time
from datetime import datetime, time as dt_time
import schedule
import traceback

# Add backend to path
SCANNER_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCANNER_DIR, '..', '..', '..'))

from backend.turbomode.options.Scanner.options_scanner import run_options_scanner
from backend.turbomode.options.Scanner.scanner_logger import log

# Configuration
SCAN_INTERVAL_MINUTES = 5
MARKET_OPEN = dt_time(9, 30)   # 9:30 AM
MARKET_CLOSE = dt_time(16, 0)  # 4:00 PM

# Scheduler logging
SCHEDULER_LOG_FILE = os.path.join(SCANNER_DIR, 'scheduler.log')

# Metrics tracking
scan_count = 0
error_count = 0
total_runtime = 0.0


def log_scheduler(message):
    """
    Log scheduler-specific messages to both console and file.

    Args:
        message: Message to log
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{timestamp}] {message}'
    print(line)

    try:
        with open(SCHEDULER_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception as e:
        print(f'[SCHEDULER LOG ERROR] Failed to write to scheduler log: {e}')


def is_market_hours():
    """
    Check if current time is within market hours.

    Returns:
        True if within market hours (9:30 AM - 4:00 PM EST), False otherwise
    """
    now = datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def is_weekday():
    """
    Check if today is a weekday (Monday-Friday).

    Returns:
        True if weekday, False if weekend
    """
    return datetime.now().weekday() < 5  # 0-4 = Mon-Fri


def should_run_scanner():
    """
    Determine if scanner should run based on market hours and day.

    Returns:
        True if scanner should run, False otherwise
    """
    return is_weekday() and is_market_hours()


def safe_run_scanner():
    """
    Execute scanner with error recovery and performance metrics.

    This wrapper ensures the scheduler continues running even if the scanner fails.
    Tracks performance metrics and error counts.
    """
    global scan_count, error_count, total_runtime

    if not should_run_scanner():
        log_scheduler('[SCHEDULER] Outside market hours, skipping scan')
        return

    scan_count += 1
    log_scheduler(f'[SCHEDULER] Running scheduled scan #{scan_count}...')

    start_time = time.time()

    try:
        # Run the scanner
        result = run_options_scanner()

        end_time = time.time()
        runtime = end_time - start_time
        total_runtime += runtime
        avg_runtime = total_runtime / scan_count

        # Log results with metrics
        log_scheduler(f'[SCHEDULER] Scan #{scan_count} completed successfully')
        log_scheduler(f'[SCHEDULER] Runtime: {runtime:.2f}s (avg: {avg_runtime:.2f}s)')
        log_scheduler(f'[SCHEDULER] Result: {result.get("count", 0)}/{result.get("total", 0)} signals cached')

        if result.get('errors'):
            log_scheduler(f'[SCHEDULER] Warnings: {len(result["errors"])} symbols failed')

        # Log aggregate metrics
        log_scheduler(f'[SCHEDULER] Metrics: {scan_count} scans, {error_count} errors, {avg_runtime:.2f}s avg runtime')

    except Exception as e:
        error_count += 1
        end_time = time.time()
        runtime = end_time - start_time

        log_scheduler(f'[SCHEDULER ERROR] Scan #{scan_count} failed after {runtime:.2f}s')
        log_scheduler(f'[SCHEDULER ERROR] Exception: {str(e)}')
        log_scheduler(f'[SCHEDULER ERROR] Error count: {error_count}/{scan_count}')

        # Log full traceback for debugging
        log_scheduler('[SCHEDULER ERROR] Traceback:')
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                log_scheduler(f'  {line}')

        # Error recovery: Continue running despite failure
        log_scheduler('[SCHEDULER] Error recovery: Continuing scheduler despite failure')


def get_scheduler_stats():
    """
    Get current scheduler statistics.

    Returns:
        Dictionary with scheduler metrics
    """
    return {
        'total_scans': scan_count,
        'total_errors': error_count,
        'error_rate': error_count / scan_count if scan_count > 0 else 0.0,
        'avg_runtime': total_runtime / scan_count if scan_count > 0 else 0.0,
        'total_runtime': total_runtime
    }


def start_scheduler():
    """
    Start the scheduler to run scanner every N minutes.
    """
    log_scheduler('[SCHEDULER] ========================================')
    log_scheduler('[SCHEDULER] Options Scanner Scheduler Starting')
    log_scheduler('[SCHEDULER] ========================================')
    log_scheduler(f'[SCHEDULER] Configuration:')
    log_scheduler(f'[SCHEDULER]   Scan interval: {SCAN_INTERVAL_MINUTES} minutes')
    log_scheduler(f'[SCHEDULER]   Market hours: {MARKET_OPEN} - {MARKET_CLOSE}')
    log_scheduler(f'[SCHEDULER]   Days: Monday - Friday only')
    log_scheduler(f'[SCHEDULER]   Log file: {SCHEDULER_LOG_FILE}')
    log_scheduler('[SCHEDULER] ========================================')

    # Validation
    log_scheduler('[SCHEDULER] Validation: Scheduler successfully imported and ready to run scanner')

    # Schedule the scanner with error recovery wrapper
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(safe_run_scanner)

    # Run initial scan if within market hours
    log_scheduler('[SCHEDULER] Running initial scan...')
    safe_run_scanner()

    # Keep scheduler running
    log_scheduler('[SCHEDULER] Scheduler is now running. Press Ctrl+C to stop.')

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        log_scheduler('[SCHEDULER] ========================================')
        log_scheduler('[SCHEDULER] Scheduler stopped by user')

        # Print final statistics
        stats = get_scheduler_stats()
        log_scheduler('[SCHEDULER] Final Statistics:')
        log_scheduler(f'[SCHEDULER]   Total scans: {stats["total_scans"]}')
        log_scheduler(f'[SCHEDULER]   Total errors: {stats["total_errors"]}')
        log_scheduler(f'[SCHEDULER]   Error rate: {stats["error_rate"]:.1%}')
        log_scheduler(f'[SCHEDULER]   Avg runtime: {stats["avg_runtime"]:.2f}s')
        log_scheduler(f'[SCHEDULER]   Total runtime: {stats["total_runtime"]:.2f}s')
        log_scheduler('[SCHEDULER] ========================================')


if __name__ == '__main__':
    start_scheduler()
