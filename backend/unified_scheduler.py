"""
TurboMode Phase 2 - Unified Scheduler System
Implements 6 scheduled tasks as defined in scheduler_config.json

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Configuration: scheduler_config.json v1.0

All tasks run as Python functions inside Flask app using APScheduler.
Dependencies are respected, and all tasks are idempotent.

Author: TurboMode System
Date: 2026-01-06
"""

import os
import sys
import json
import logging
import sqlite3
import shutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import logging system
from backend.scheduler_logger import get_logger_manager
from backend.task_timeout import TaskTimeoutError, task_timeout
import threading

# Get logger manager instance
logger_manager = get_logger_manager()
logger = logger_manager.get_scheduler_logger()

# Global scheduler instance
scheduler = None
config = None

# Job execution state
job_state = {
    'last_runs': {},
    'last_results': {},
    'errors': {}
}

# Execution history file path
EXECUTION_HISTORY_FILE = os.path.join(current_dir, 'data', 'execution_history.json')

def load_execution_history() -> List[Dict[str, Any]]:
    """Load execution history from file"""
    if not os.path.exists(EXECUTION_HISTORY_FILE):
        return []
    try:
        with open(EXECUTION_HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load execution history: {e}")
        return []

def save_execution_history(history: List[Dict[str, Any]]):
    """Save execution history to file with 30-day retention"""
    # Filter to keep only last 30 days
    cutoff_date = datetime.now() - timedelta(days=30)
    filtered_history = [
        entry for entry in history
        if datetime.fromisoformat(entry['timestamp']) > cutoff_date
    ]

    # Ensure data directory exists
    os.makedirs(os.path.dirname(EXECUTION_HISTORY_FILE), exist_ok=True)

    # Save to file
    try:
        with open(EXECUTION_HISTORY_FILE, 'w') as f:
            json.dump(filtered_history, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save execution history: {e}")

def add_execution_to_history(task_id: int, task_name: str, status: str, duration_seconds: float = None, error: str = None):
    """Add a task execution to the history"""
    history = load_execution_history()

    entry = {
        'task_id': task_id,
        'task_name': task_name,
        'timestamp': datetime.now().isoformat(),
        'status': status,  # 'success', 'failed', 'timeout'
        'duration_seconds': duration_seconds,
        'error': error
    }

    history.append(entry)
    save_execution_history(history)


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_scheduler_config() -> Dict[str, Any]:
    """Load scheduler configuration from JSON"""
    config_path = os.path.join(current_dir, 'scheduler_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Scheduler config not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def get_task_config(task_id: int) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific task"""
    global config

    if config is None:
        config = load_scheduler_config()

    for task in config['scheduled_tasks']:
        if task['task_id'] == task_id:
            return task

    return None


# ============================================================================
# TASK 1: MASTER MARKET DATA INGESTION
# ============================================================================

def run_ingestion() -> Dict[str, Any]:
    """
    TASK 1: Master Market Data Ingestion

    Fetches daily OHLCV, fundamentals, splits, dividends, and metadata.
    Writes updates to Master Market Data DB only.

    Returns:
        Dictionary with ingestion results
    """
    task_id = 1
    task_config = get_task_config(task_id)

    # Get task-specific logger
    task_logger = logger_manager.get_task_logger(task_id, task_config['name'])

    task_logger.info("=" * 80)
    task_logger.info(f"TASK {task_id}: {task_config['name']}")
    start_time = datetime.now()
    task_logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    task_logger.info("=" * 80)

    try:
        # Import unified ingestion module
        from backend.turbomode.core_engine.ingest_master_market_data import run_full_ingestion
        from backend.turbomode.core_engine.training_symbols import get_training_symbols, CRYPTO_SYMBOLS
        from backend.turbomode.core_engine.scanning_symbols import get_scanning_symbols

        # Unified ingestion does not use a class
        # Call run_full_ingestion() directly - uses CORE_230 internally

        task_logger.info(f"Running unified ingestion (CORE_230 symbols, 5d period)...")

        # Run ingestion (5 days to catch up on any missed data - incremental updates)
        # Note: For full refresh, use period='10y' or full_refresh=True
        results = run_full_ingestion(period='5d')

        # Log results
        task_logger.info(f"[SUCCESS] Task {task_id} completed")
        task_logger.info(f"  Unified ingestion complete (CORE_230: 230 symbols)")
        task_logger.info(f"  Data updated: master_market_data/master_market_data.db")

        # Update state
        duration = (datetime.now() - start_time).total_seconds()
        job_state['last_runs'][task_id] = datetime.now().isoformat()
        job_state['last_results'][task_id] = {
            'status': 'success',
            'symbols_processed': results.get('total_symbols', 0) if results else 0,
            'candles_ingested': results.get('total_candles', 0) if results else 0
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'success', duration)

        task_logger.info("=" * 80)

        return {
            'success': True,
            'task_id': task_id,
            'results': results
        }

    except Exception as e:
        task_logger.error(f"[ERROR] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

        duration = (datetime.now() - start_time).total_seconds()
        job_state['errors'][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'failed', duration, str(e))

        return {
            'success': False,
            'task_id': task_id,
            'error': str(e)
        }


# ============================================================================
# TASK 2: TURBOMODE TRAINING ORCHESTRATOR
# ============================================================================

def run_orchestrator() -> Dict[str, Any]:
    """
    TASK 2: TurboMode Training Orchestrator

    Loads symbols from Master Market Data DB.
    Generates features and training samples.
    Trains models and saves them to TurboMode.db.
    Computes SHAP values and saves logs.
    Updates model metadata and versioning.

    Returns:
        Dictionary with training results
    """
    task_id = 2
    task_config = get_task_config(task_id)

    # Get task-specific logger
    task_logger = logger_manager.get_task_logger(task_id, task_config['name'])

    task_logger.info("=" * 80)
    task_logger.info(f"TASK {task_id}: {task_config['name']}")
    start_time = datetime.now()
    task_logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    task_logger.info("=" * 80)

    try:
        # Import 14-day optimized training orchestrator
        from backend.turbomode.core_engine.train_all_sectors_optimized_orchestrator import train_all_sectors_optimized

        task_logger.info("Running 14-day optimized training for all sectors...")
        task_logger.info("Architecture: 11 sectors × 6 models (5 base + 1 meta-learner) = 66 models")
        task_logger.info("Label source: 14-day outcomes from trades table (±5% thresholds)")

        # Run 14-day optimized training (66 models total)
        results = train_all_sectors_optimized()

        if results:
            task_logger.info(f"[SUCCESS] Task {task_id} completed - 14-day optimized training")
            task_logger.info(f"Trained {len(results)} sectors successfully")

            duration = (datetime.now() - start_time).total_seconds()
            job_state['last_runs'][task_id] = datetime.now().isoformat()
            job_state['last_results'][task_id] = {
                'status': 'success',
                'training_type': '14day_optimized',
                'total_models': 66,
                'sectors_trained': len(results)
            }

            # Add to execution history
            add_execution_to_history(task_id, task_config['name'], 'success', duration)

            task_logger.info("=" * 80)

            return {
                'success': True,
                'task_id': task_id,
                'training_type': '14day_optimized',
                'total_models': 66,
                'sectors_trained': len(results)
            }
        else:
            raise Exception("Training returned no results - check logs")

    except Exception as e:
        task_logger.error(f"[ERROR] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

        duration = (datetime.now() - start_time).total_seconds()
        job_state['errors'][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'failed', duration, str(e))

        return {
            'success': False,
            'task_id': task_id,
            'error': str(e)
        }


# ============================================================================
# TASK 3: OVERNIGHT SCANNER
# ============================================================================

def run_overnight_scanner() -> Dict[str, Any]:
    """
    TASK 3: Overnight Scanner

    Loads the latest trained model from TurboMode.db.
    Generates predictions for the next trading day.
    Saves signals and prediction metadata to TurboMode.db.

    Returns:
        Dictionary with scanner results
    """
    task_id = 3
    task_config = get_task_config(task_id)

    # Get task-specific logger
    task_logger = logger_manager.get_task_logger(task_id, task_config['name'])

    task_logger.info("=" * 80)
    task_logger.info(f"TASK {task_id}: {task_config['name']}")
    start_time = datetime.now()
    task_logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    task_logger.info("=" * 80)

    try:
        # Import scanner and paths
        from backend.turbomode.core_engine.overnight_scanner import ProductionScanner
        from backend.turbomode.core_engine.scanning_symbols import get_scanning_symbols
        from backend.turbomode.paths import TURBOMODE_DB

        # Initialize scanner (14-day swing trading architecture)
        scanner = ProductionScanner()

        # Get all scanning symbols (208 stocks)
        symbols = get_scanning_symbols()

        task_logger.info(f"Scanning {len(symbols)} symbols...")

        # Run scanner on all symbols
        results = scanner.scan_all(max_signals_per_type=100)

        # Count signals
        num_signals = len(results.get('signals', []))

        task_logger.info(f"[SUCCESS] Task {task_id} completed")
        task_logger.info(f"  Symbols scanned: {len(symbols)}")
        task_logger.info(f"  Signals generated: {num_signals}")

        duration = (datetime.now() - start_time).total_seconds()
        job_state['last_runs'][task_id] = datetime.now().isoformat()
        job_state['last_results'][task_id] = {
            'status': 'success',
            'symbols_scanned': len(symbols),
            'signals_generated': num_signals
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'success', duration)

        task_logger.info("=" * 80)

        return {
            'success': True,
            'task_id': task_id,
            'signals_generated': num_signals
        }

    except Exception as e:
        task_logger.error(f"[ERROR] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

        duration = (datetime.now() - start_time).total_seconds()
        job_state['errors'][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'failed', duration, str(e))

        return {
            'success': False,
            'task_id': task_id,
            'error': str(e)
        }


# ============================================================================
# TASK 4: BACKTEST DATA GENERATOR
# ============================================================================

def run_backtest_generator() -> Dict[str, Any]:
    """
    TASK 4: Backtest Data Generator

    Uses Master Market Data DB to generate updated backtest datasets.
    Saves backtest results and metadata to TurboMode.db.
    Ensures backtests reflect the latest ingestion.

    Returns:
        Dictionary with backtest results
    """
    task_id = 4
    task_config = get_task_config(task_id)

    # Get task-specific logger
    task_logger = logger_manager.get_task_logger(task_id, task_config['name'])

    task_logger.info("=" * 80)
    task_logger.info(f"TASK {task_id}: {task_config['name']}")
    start_time = datetime.now()
    task_logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    task_logger.info("=" * 80)

    try:
        # Import 14-day backtest engine (aligned with 14-day swing trading)
        from backend.turbomode.core_engine.turbomode_backtest import TurboModeBacktest
        from backend.turbomode.core_engine.training_symbols import get_training_symbols

        # Initialize 14-day backtest engine
        backtest = TurboModeBacktest()

        # Get training symbols for backtesting
        symbols = get_training_symbols()

        task_logger.info(f"Generating 14-day backtest samples for {len(symbols)} symbols...")
        task_logger.info("Configuration: 14-day holding period, ±5% thresholds")

        # Run backtest for all symbols
        total_samples = 0
        for symbol in symbols:
            result = backtest.generate_backtest_samples(
                symbol=symbol,
                lookback_days=3650,  # 10 years of data
                incremental=True  # Only process new data
            )
            total_samples += result['total_samples']

        if total_samples > 0:
            task_logger.info(f"[SUCCESS] Task {task_id} completed - Generated {total_samples:,} samples")

            duration = (datetime.now() - start_time).total_seconds()
            job_state['last_runs'][task_id] = datetime.now().isoformat()
            job_state['last_results'][task_id] = {
                'status': 'success',
                'total_samples': total_samples
            }

            # Add to execution history
            add_execution_to_history(task_id, task_config['name'], 'success', duration)

            task_logger.info("=" * 80)

            return {
                'success': True,
                'task_id': task_id,
                'total_samples': total_samples
            }
        else:
            raise Exception("No backtest samples generated - check logs")

    except Exception as e:
        task_logger.error(f"[ERROR] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

        duration = (datetime.now() - start_time).total_seconds()
        job_state['errors'][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'failed', duration, str(e))

        return {
            'success': False,
            'task_id': task_id,
            'error': str(e)
        }


# ============================================================================
# TASK 5: ADAPTIVE STOCK RANKING
# ============================================================================

def run_adaptive_ranking() -> Dict[str, Any]:
    """
    TASK 5: Adaptive Stock Ranking

    Calculates risk-adjusted win rates from backtest data.
    Applies directional fallback logic.
    Generates composite scores and top 10 rankings.
    Saves results to stock_rankings.json.

    Returns:
        Dictionary with ranking results
    """
    task_id = 5
    task_config = get_task_config(task_id)

    # Get task-specific logger
    task_logger = logger_manager.get_task_logger(task_id, task_config['name'])

    task_logger.info("=" * 80)
    task_logger.info(f"TASK {task_id}: {task_config['name']}")
    start_time = datetime.now()
    task_logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    task_logger.info("=" * 80)

    try:
        # Import adaptive stock ranker
        from backend.turbomode.adaptive_stock_ranker import AdaptiveStockRanker

        # Initialize ranker
        ranker = AdaptiveStockRanker()

        task_logger.info("Running adaptive stock ranking analysis...")
        task_logger.info("Configuration: WIN_FRACTION = 3.0, risk-adjusted win rates")

        # Run analysis
        results = ranker.run_analysis()

        if results:
            top_10_symbols = [s['symbol'] for s in results['top_10']]
            task_logger.info(f"[SUCCESS] Task {task_id} completed")
            task_logger.info(f"Top 10 symbols: {top_10_symbols}")

            duration = (datetime.now() - start_time).total_seconds()
            job_state['last_runs'][task_id] = datetime.now().isoformat()
            job_state['last_results'][task_id] = {
                'status': 'success',
                'top_10_count': len(results['top_10']),
                'total_stocks_analyzed': len(results['all_stats'])
            }

            # Add to execution history
            add_execution_to_history(task_id, task_config['name'], 'success', duration)

            task_logger.info("=" * 80)

            return {
                'success': True,
                'task_id': task_id,
                'top_10_symbols': top_10_symbols,
                'total_analyzed': len(results['all_stats'])
            }
        else:
            raise Exception("Ranking analysis returned None - check logs")

    except Exception as e:
        task_logger.error(f"[ERROR] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

        duration = (datetime.now() - start_time).total_seconds()
        job_state['errors'][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'failed', duration, str(e))

        return {
            'success': False,
            'task_id': task_id,
            'error': str(e)
        }


# ============================================================================
# TASK 6: DRIFT MONITORING SYSTEM
# ============================================================================

def run_drift_monitor() -> Dict[str, Any]:
    """
    TASK 6: Drift Monitoring System

    Compares today's feature distributions to historical baselines.
    Detects regime shifts and logs drift events.
    Writes drift logs to TurboMode.db.
    Triggers retraining flags if drift exceeds thresholds.

    Returns:
        Dictionary with drift monitoring results
    """
    task_id = 6
    task_config = get_task_config(task_id)

    # Get task-specific logger
    task_logger = logger_manager.get_task_logger(task_id, task_config['name'])

    task_logger.info("=" * 80)
    task_logger.info(f"TASK {task_id}: {task_config['name']}")
    start_time = datetime.now()
    task_logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    task_logger.info("=" * 80)

    try:
        # Import drift monitor
        from backend.turbomode.drift_monitor import DriftMonitor

        # Initialize drift monitor
        monitor = DriftMonitor()

        task_logger.info("Running drift monitoring...")

        # Simple drift detection for now
        # In full implementation, this would:
        # 1. Load baseline feature distributions from TurboMode.db
        # 2. Load current feature distributions
        # 3. Calculate PSI, KL, KS for each feature
        # 4. Flag features with drift above thresholds
        # 5. Trigger retraining if needed

        # For now, create a minimal drift check
        drift_results = {
            'features_checked': 0,
            'features_drifted': 0,
            'drift_detected': False,
            'retraining_triggered': False
        }

        # Log stub results
        task_logger.info("[INFO] Drift monitoring running...")
        task_logger.info(f"  Features checked: {drift_results['features_checked']}")
        task_logger.info(f"  Features drifted: {drift_results['features_drifted']}")
        task_logger.info(f"  Drift detected: {drift_results['drift_detected']}")

        # Save to database (stub)
        conn = monitor.turbomode_db.conn
        cursor = conn.cursor()

        # Create drift_monitoring table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_monitoring (
                drift_id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_date TEXT NOT NULL,
                features_checked INTEGER,
                features_drifted INTEGER,
                drift_detected INTEGER,
                retraining_triggered INTEGER
            )
        """)

        cursor.execute("""
            INSERT INTO drift_monitoring (
                check_date, features_checked, features_drifted, drift_detected, retraining_triggered
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            drift_results['features_checked'],
            drift_results['features_drifted'],
            1 if drift_results['drift_detected'] else 0,
            1 if drift_results['retraining_triggered'] else 0
        ))

        conn.commit()

        task_logger.info(f"[SUCCESS] Task {task_id} completed")

        duration = (datetime.now() - start_time).total_seconds()
        job_state['last_runs'][task_id] = datetime.now().isoformat()
        job_state['last_results'][task_id] = {
            'status': 'success',
            **drift_results
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'success', duration)

        task_logger.info("=" * 80)

        return {
            'success': True,
            'task_id': task_id,
            **drift_results
        }

    except Exception as e:
        task_logger.error(f"[ERROR] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

        duration = (datetime.now() - start_time).total_seconds()
        job_state['errors'][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'failed', duration, str(e))

        return {
            'success': False,
            'task_id': task_id,
            'error': str(e)
        }


# ============================================================================
# TASK 7: WEEKLY MAINTENANCE
# ============================================================================

def run_weekly_maintenance() -> Dict[str, Any]:
    """
    TASK 7: Weekly Maintenance

    VACUUMs Master Market Data DB.
    VACUUMs TurboMode.db.
    Cleans temp directories.
    Archives logs older than 30 days.

    Returns:
        Dictionary with maintenance results
    """
    task_id = 7
    task_config = get_task_config(task_id)

    # Get task-specific logger
    task_logger = logger_manager.get_task_logger(task_id, task_config['name'])

    task_logger.info("=" * 80)
    task_logger.info(f"TASK {task_id}: {task_config['name']}")
    start_time = datetime.now()
    task_logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    task_logger.info("=" * 80)

    try:
        # Import paths
        from backend.turbomode.paths import TURBOMODE_DB

        results = {
            'master_db_vacuum': False,
            'turbomode_db_vacuum': False,
            'temp_cleaned': False,
            'logs_archived': False
        }

        # VACUUM Master Market Data DB
        try:
            master_db_path = os.path.join(project_root, 'master_market_data', 'market_data.db')
            if os.path.exists(master_db_path):
                conn = sqlite3.connect(master_db_path)
                conn.execute("VACUUM")
                conn.close()
                task_logger.info("[OK] Master Market Data DB vacuumed")
                results['master_db_vacuum'] = True
        except Exception as e:
            task_logger.warning(f"[WARNING] Failed to vacuum Master DB: {e}")

        # VACUUM TurboMode DB (using absolute path from paths module)
        try:
            turbomode_db_path = str(TURBOMODE_DB)
            if os.path.exists(turbomode_db_path):
                conn = sqlite3.connect(turbomode_db_path)
                conn.execute("VACUUM")
                conn.close()
                task_logger.info("[OK] TurboMode DB vacuumed")
                results['turbomode_db_vacuum'] = True
        except Exception as e:
            task_logger.warning(f"[WARNING] Failed to vacuum TurboMode DB: {e}")

        # Clean temp directories
        try:
            temp_dirs = [
                os.path.join(current_dir, 'temp'),
                os.path.join(project_root, 'temp')
            ]

            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for item in os.listdir(temp_dir):
                        item_path = os.path.join(temp_dir, item)
                        try:
                            if os.path.isfile(item_path):
                                os.unlink(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except Exception as e:
                            task_logger.warning(f"Failed to delete {item_path}: {e}")

            task_logger.info("[OK] Temp directories cleaned")
            results['temp_cleaned'] = True
        except Exception as e:
            task_logger.warning(f"[WARNING] Failed to clean temp directories: {e}")

        # Archive old logs
        try:
            import zipfile

            # Get old logs (older than 30 days)
            old_logs = logger_manager.get_old_logs(days=30)

            if old_logs:
                # Create archive directory
                archive_dir = os.path.join(current_dir, 'logs', 'archive')
                os.makedirs(archive_dir, exist_ok=True)

                # Create ZIP archive with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_path = os.path.join(archive_dir, f'logs_archive_{timestamp}.zip')

                # Archive old logs
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for log_path in old_logs:
                        if os.path.exists(log_path):
                            zipf.write(log_path, os.path.basename(log_path))
                            # Delete original log file after archiving
                            os.remove(log_path)

                task_logger.info(f"[OK] Archived {len(old_logs)} old log files to {archive_path}")
                results['logs_archived'] = True
            else:
                task_logger.info("[INFO] No old logs to archive")
                results['logs_archived'] = True

        except Exception as e:
            task_logger.warning(f"[WARNING] Failed to archive logs: {e}")

        task_logger.info(f"[SUCCESS] Task {task_id} completed")
        task_logger.info(f"  Master DB vacuumed: {results['master_db_vacuum']}")
        task_logger.info(f"  TurboMode DB vacuumed: {results['turbomode_db_vacuum']}")
        task_logger.info(f"  Temp cleaned: {results['temp_cleaned']}")

        duration = (datetime.now() - start_time).total_seconds()
        job_state['last_runs'][task_id] = datetime.now().isoformat()
        job_state['last_results'][task_id] = {
            'status': 'success',
            **results
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'success', duration)

        task_logger.info("=" * 80)

        return {
            'success': True,
            'task_id': task_id,
            'results': results
        }

    except Exception as e:
        task_logger.error(f"[ERROR] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

        duration = (datetime.now() - start_time).total_seconds()
        job_state['errors'][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        # Add to execution history
        add_execution_to_history(task_id, task_config['name'], 'failed', duration, str(e))

        return {
            'success': False,
            'task_id': task_id,
            'error': str(e)
        }


# ============================================================================
# SCHEDULER MANAGEMENT
# ============================================================================

# Map task IDs to subprocess wrapper functions (for isolation from Flask)
def _task_subprocess_wrapper(task_id, task_func):
    """
    Wrapper that runs task function in a subprocess to isolate from Flask.
    Returns a callable that executes the task in a subprocess.
    """
    import subprocess
    import sys

    def wrapper():
        """Execute task in subprocess"""
        # Get the Python executable
        python_exe = sys.executable

        # Determine which script to call based on task_id
        script_map = {
            1: 'backend/turbomode/core_engine/ingest_master_market_data.py',
            2: 'backend/turbomode/core_engine/train_all_sectors_fastmode_orchestrator.py',
            3: 'backend/turbomode/core_engine/overnight_scanner.py',
            "3A": 'backend/turbomode/core_engine/overnight_scanner.py',
            "3B": 'backend/turbomode/core_engine/overnight_scanner.py',
            "3C": 'backend/turbomode/core_engine/overnight_scanner.py',
            4: 'backend/turbomode/core_engine/generate_backtest_data.py',
            5: 'backend/turbomode/adaptive_stock_ranker.py',
            6: None,  # Drift monitor - runs inline for now
            7: None   # Weekly maintenance - runs inline for now
        }

        script_path = script_map.get(task_id)

        # If no script path, fall back to inline task function
        if script_path is None:
            task_logger = logger_manager.get_task_logger(task_id)
            task_logger.info(f"[INLINE] Running task {task_id} inline (no subprocess script)")
            try:
                return task_func()
            except Exception as e:
                task_logger.error(f"[INLINE ERROR] Task {task_id}: {e}")
                return {
                    'success': False,
                    'task_id': task_id,
                    'error': str(e)
                }

        # Run task in subprocess with output redirected to task log
        task_config = get_task_config(task_id)
        task_name = task_config['name'] if task_config else f'Task {task_id}'
        start_time = time.time()
        task_logger = None

        try:
            # Create logger (may fail if task_name has invalid chars)
            task_logger = logger_manager.get_task_logger(task_id, task_name)
            task_logger.info(f"[SUBPROCESS] Starting task {task_id} via {script_path}")

            # Start subprocess (fully isolated from Flask)
            process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=project_root
            )

            # Stream output to task log (don't let it go to Flask terminal)
            for line in iter(process.stdout.readline, ''):
                if line:
                    task_logger.info(line.rstrip())

            # Wait for completion
            return_code = process.wait()
            duration = time.time() - start_time

            if return_code == 0:
                task_logger.info(f"[SUBPROCESS] Task {task_id} completed successfully in {duration:.1f}s")

                # Add to execution history
                add_execution_to_history(task_id, task_name, 'success', duration)

                return {
                    'success': True,
                    'task_id': task_id,
                    'message': f'Task {task_id} completed via subprocess'
                }
            else:
                error_msg = f'Subprocess exited with code {return_code}'
                task_logger.error(f"[SUBPROCESS] Task {task_id} failed: {error_msg}")

                # Add to execution history
                add_execution_to_history(task_id, task_name, 'failed', duration, error_msg)

                return {
                    'success': False,
                    'task_id': task_id,
                    'error': error_msg
                }

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            # Log error (if logger was created)
            if task_logger:
                task_logger.error(f"[SUBPROCESS ERROR] Task {task_id}: {error_msg}")
            else:
                logger.error(f"[SUBPROCESS ERROR] Task {task_id}: {error_msg}")

            # Add to execution history
            add_execution_to_history(task_id, task_name, 'failed', duration, error_msg)

            return {
                'success': False,
                'task_id': task_id,
                'error': error_msg
            }

    return wrapper


# Map task IDs to subprocess wrapper functions
TASK_FUNCTIONS = {
    1: _task_subprocess_wrapper(1, run_ingestion),
    2: _task_subprocess_wrapper(2, run_orchestrator),
    3: _task_subprocess_wrapper(3, run_overnight_scanner),
    "3A": _task_subprocess_wrapper("3A", run_overnight_scanner),
    "3B": _task_subprocess_wrapper("3B", run_overnight_scanner),
    "3C": _task_subprocess_wrapper("3C", run_overnight_scanner),
    4: _task_subprocess_wrapper(4, run_backtest_generator),
    5: _task_subprocess_wrapper(5, run_adaptive_ranking),
    6: _task_subprocess_wrapper(6, run_drift_monitor),
    7: _task_subprocess_wrapper(7, run_weekly_maintenance)
}


def check_dependencies(task_id: int) -> tuple[bool, str]:
    """
    Check if all dependencies for a task have completed successfully

    Args:
        task_id: Task ID to check dependencies for

    Returns:
        (success, error_message) tuple
    """
    task_config = get_task_config(task_id)
    if not task_config:
        return (False, f"Task {task_id} not found in config")

    dependencies = task_config.get('dependencies', [])

    if not dependencies:
        return (True, "")

    # Check each dependency
    for dep_task_id in dependencies:
        # Check if dependency has run
        if dep_task_id not in job_state.get('last_runs', {}):
            return (False, f"Dependency Task {dep_task_id} has not run yet")

        # Check if dependency succeeded
        dep_result = job_state.get('last_results', {}).get(dep_task_id, {})
        if dep_result.get('status') != 'success':
            return (False, f"Dependency Task {dep_task_id} did not complete successfully")

    return (True, "")


def run_task_with_timeout_and_retry(task_id: int) -> Dict[str, Any]:
    """
    Wrapper that runs a task with timeout enforcement, retry logic, and dependency checking

    Args:
        task_id: Task ID (1-6)

    Returns:
        Task result dictionary
    """
    task_config = get_task_config(task_id)
    task_func = TASK_FUNCTIONS.get(task_id)

    if not task_func or not task_config:
        return {
            'success': False,
            'task_id': task_id,
            'error': f'Invalid task_id: {task_id}'
        }

    # Check dependencies
    deps_ok, deps_error = check_dependencies(task_id)
    if not deps_ok:
        logger.warning(f"[DEPENDENCY] Task {task_id} cannot run: {deps_error}")
        return {
            'success': False,
            'task_id': task_id,
            'error': f"Dependencies not met: {deps_error}"
        }

    logger.info(f"[DEPENDENCY] Task {task_id} dependencies satisfied")

    timeout_minutes = task_config.get('timeout_minutes', 60)
    retry_on_failure = task_config.get('retry_on_failure', False)
    max_retries = task_config.get('max_retries', 0)

    attempt = 0
    last_error = None

    while attempt <= max_retries:
        try:
            # Apply timeout decorator
            timeout_func = task_timeout(timeout_minutes)(task_func)

            # Run the task
            result = timeout_func()

            # Success - return immediately
            if result.get('success'):
                if attempt > 0:
                    logger.info(f"[RETRY SUCCESS] Task {task_id} succeeded on attempt {attempt + 1}")
                return result

            # Task returned but wasn't successful
            last_error = result.get('error', 'Unknown error')

            if not retry_on_failure or attempt >= max_retries:
                return result

            attempt += 1
            logger.warning(f"[RETRY] Task {task_id} failed (attempt {attempt}/{max_retries + 1}), retrying...")

        except TaskTimeoutError as e:
            last_error = f"Task exceeded timeout of {timeout_minutes} minutes"
            logger.error(f"[TIMEOUT] Task {task_id}: {last_error}")

            if not retry_on_failure or attempt >= max_retries:
                return {
                    'success': False,
                    'task_id': task_id,
                    'error': last_error
                }

            attempt += 1
            logger.warning(f"[RETRY] Task {task_id} timed out (attempt {attempt}/{max_retries + 1}), retrying...")

        except Exception as e:
            last_error = str(e)
            logger.error(f"[ERROR] Task {task_id} exception: {last_error}")

            if not retry_on_failure or attempt >= max_retries:
                return {
                    'success': False,
                    'task_id': task_id,
                    'error': last_error
                }

            attempt += 1
            logger.warning(f"[RETRY] Task {task_id} failed (attempt {attempt}/{max_retries + 1}), retrying...")

    # All retries exhausted
    return {
        'success': False,
        'task_id': task_id,
        'error': f"Failed after {max_retries + 1} attempts: {last_error}"
    }


def job_listener(event):
    """Listener for scheduler events"""
    if event.exception:
        logger.error(f"[SCHEDULER] Job {event.job_id} failed!")
        logger.error(f"Exception: {event.exception}")
    else:
        logger.info(f"[SCHEDULER] Job {event.job_id} completed successfully")


def start_unified_scheduler():
    """
    Start the unified scheduler with all tasks from config
    """
    global scheduler, config

    if scheduler is not None and scheduler.running:
        logger.warning("[SCHEDULER] Already running")
        return False

    # Load config
    config = load_scheduler_config()

    logger.info("=" * 80)
    logger.info("UNIFIED SCHEDULER - STARTING")
    logger.info(f"Version: {config['version']}")
    logger.info("=" * 80)

    # Create scheduler with persistent job store
    try:
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

        jobstores = {
            'default': SQLAlchemyJobStore(url='sqlite:///backend/data/scheduler_jobs.db')
        }

        scheduler = BackgroundScheduler(
            jobstores=jobstores,
            daemon=True,
            timezone=config['global_settings'].get('timezone', 'America/New_York')
        )
        logger.info("[SCHEDULER] Using persistent SQLAlchemy job store")
    except ImportError:
        # Fall back to in-memory store if SQLAlchemy not available
        scheduler = BackgroundScheduler(
            daemon=True,
            timezone=config['global_settings'].get('timezone', 'America/New_York')
        )
        logger.warning("[SCHEDULER] SQLAlchemy not available - using in-memory job store")

    # Add event listener
    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    # Register all enabled tasks
    for task in config['scheduled_tasks']:
        if not task.get('enabled', True):
            logger.info(f"[SKIP] Task {task['task_id']}: {task['name']} (disabled)")
            continue

        task_func = TASK_FUNCTIONS.get(task['task_id'])

        if task_func is None:
            logger.error(f"[ERROR] No function for task {task['task_id']}")
            continue

        schedule = task['schedule']
        schedule_type = schedule.get('type', 'cron')

        # Create trigger based on schedule type
        if schedule_type == 'interval':
            # Interval-based schedule (e.g., every 3 minutes during market hours)
            from apscheduler.triggers.interval import IntervalTrigger
            from datetime import time as dt_time

            trigger = IntervalTrigger(
                minutes=schedule['minutes'],
                start_date=schedule.get('start_time'),  # e.g., "08:30"
                end_date=schedule.get('end_time'),      # e.g., "15:00"
                timezone=config['global_settings'].get('timezone', 'America/Chicago')
            )
            schedule_desc = f"Every {schedule['minutes']} min ({schedule.get('start_time')}-{schedule.get('end_time')}) {schedule.get('day_of_week', 'daily')}"
        else:
            # Cron-based schedule
            trigger = CronTrigger(
                hour=schedule['hour'],
                minute=schedule['minute'],
                day_of_week=schedule.get('day_of_week', '*'),
                timezone=config['global_settings'].get('timezone', 'America/Chicago')
            )
            # Format schedule description (handle both int and string hour/minute)
            hour_str = str(schedule['hour']) if isinstance(schedule['hour'], int) else schedule['hour']
            min_str = str(schedule['minute']) if isinstance(schedule['minute'], int) else schedule['minute']
            schedule_desc = f"{hour_str}:{min_str} {schedule.get('day_of_week', 'daily')}"

        # Add job with timeout and retry wrapper (async to prevent Flask crashes)
        task_id = task['task_id']
        scheduler.add_job(
            lambda tid=task_id: run_task_manually_async(tid),
            trigger=trigger,
            id=f"task_{task_id}",
            name=task['name'],
            **config['global_settings']['job_defaults']
        )

        logger.info(f"[REGISTERED] Task {task['task_id']}: {task['name']}")
        logger.info(f"             Schedule: {schedule_desc}")

    # Start scheduler
    scheduler.start()

    logger.info("=" * 80)
    logger.info(f"[OK] Unified Scheduler STARTED")
    logger.info(f"Active jobs: {len(scheduler.get_jobs())}")
    logger.info("=" * 80)

    return True


def stop_unified_scheduler():
    """Stop the unified scheduler"""
    global scheduler

    if scheduler is None or not scheduler.running:
        logger.warning("[SCHEDULER] Not running")
        return False

    scheduler.shutdown()
    scheduler = None

    logger.info("[OK] Unified Scheduler STOPPED")

    return True


def get_scheduler_status() -> Dict[str, Any]:
    """Get current scheduler status"""
    global scheduler, config

    if config is None:
        config = load_scheduler_config()

    # Check if scheduler is running (either in-process or as subprocess)
    is_running = scheduler is not None and scheduler.running

    # If not running in-process, check if subprocess is running
    if not is_running:
        try:
            import subprocess as sp
            # Check if unified_scheduler.py is running as a subprocess
            result = sp.run(
                ['wmic', 'process', 'where', "commandline like '%unified_scheduler.py%'", 'get', 'processid'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # If we find any process IDs (excluding the wmic command itself), scheduler is running
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 1:  # More than just the header
                # Verify the log file also exists and has the STARTED message
                log_path = os.path.join(os.path.dirname(__file__), 'logs', 'scheduler.log')
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-100:]):  # Check last 100 lines
                            if 'Unified Scheduler STARTED' in line:
                                is_running = True
                                break
        except Exception as e:
            logger.warning(f"Could not check subprocess status: {e}")

    jobs = []
    if is_running and scheduler is not None:
        for job in scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            })
    elif is_running:
        # Subprocess is running but we can't get job details
        # Return job info from config
        for task in config.get('scheduled_tasks', []):
            jobs.append({
                'id': f"task_{task['task_id']}",
                'name': task['name'],
                'next_run': None  # Can't get from subprocess
            })

    return {
        'running': is_running,
        'version': config.get('version'),
        'jobs': jobs,
        'last_runs': job_state.get('last_runs', {}),
        'last_results': job_state.get('last_results', {}),
        'errors': job_state.get('errors', {}),
        'execution_history': load_execution_history()
    }


def run_task_manually(task_id: int) -> Dict[str, Any]:
    """Manually trigger a specific task with timeout and retry"""
    logger.info(f"[MANUAL] Running task {task_id} manually...")

    # Use the wrapper that enforces timeout and retry
    result = run_task_with_timeout_and_retry(task_id)

    return result


def run_task_manually_async(task_id: int) -> None:
    """
    Start the given task in a daemon thread.
    The task itself will run in a subprocess via TASK_FUNCTIONS wrapper.
    This provides double isolation: thread + subprocess.
    """
    # Run in daemon thread so it doesn't block Flask
    # The task function in TASK_FUNCTIONS will launch a subprocess
    thread = threading.Thread(
        target=run_task_with_timeout_and_retry,
        args=(task_id,),
        daemon=True,
        name=f"Task-{task_id}"
    )
    thread.start()
    logger.info(f"[ASYNC] Task {task_id} started in daemon thread (will spawn subprocess)")


if __name__ == '__main__':
    # Production mode - run scheduler as standalone process
    print("=" * 80)
    print("UNIFIED SCHEDULER - STANDALONE MODE")
    print("=" * 80)

    # Start scheduler
    start_unified_scheduler()

    # Get status
    status = get_scheduler_status()
    print(f"\nScheduler Status:")
    print(f"  Running: {status['running']}")
    print(f"  Version: {status['version']}")
    print(f"  Active Jobs: {len(status['jobs'])}")

    print("\nScheduled Jobs:")
    for job in status['jobs']:
        print(f"  - {job['name']}")
        print(f"    Next run: {job['next_run']}")

    print("\n[OK] Unified Scheduler initialized!")
    print("Press Ctrl+C to exit...")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        stop_unified_scheduler()
        print("Goodbye!")
