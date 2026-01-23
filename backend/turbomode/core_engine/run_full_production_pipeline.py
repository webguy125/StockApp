
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
TurboMode Full Production Pipeline Orchestrator
Runs the complete end-to-end production pipeline with no user input.

ARCHITECTURE: Single-model-per-sector (1d/5% only)

Pipeline Order:
1. Data Ingestion (Master Market Data) - ~20-25 minutes
2. Feature/Label Generation and Backtesting - ~0-2 minutes
3. Model Training (11 sectors, single model each) - ~10-15 minutes
4. Production Scanner (generate signals, update DB, manage positions) - ~2-5 minutes

This script can be run manually or scheduled for full system updates.

Usage:
    python run_full_production_pipeline.py

Duration: ~30-45 minutes for complete run (was 1.5-2 hours with old architecture)
"""

import sys
import os
from datetime import datetime
import time
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
turbomode_dir = os.path.dirname(current_dir)
backend_dir = os.path.dirname(turbomode_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('production_pipeline')

# Track overall timing
PIPELINE_START_TIME = time.time()
PIPELINE_START_TIMESTAMP = datetime.now()


def print_section_header(step_num: int, title: str):
    """Print formatted section header"""
    print("\n")
    print("=" * 80)
    print(f"STEP {step_num}: {title}")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()


def print_section_footer(step_num: int, duration_seconds: float):
    """Print formatted section footer"""
    print()
    print("-" * 80)
    print(f"STEP {step_num} COMPLETED in {duration_seconds/60:.2f} minutes")
    print("-" * 80)
    print()


def run_step_1_data_ingestion():
    """
    STEP 1: Master Market Data Ingestion
    Fetches latest OHLCV data for all symbols from IBKR/yfinance
    """
    step_num = 1
    print_section_header(step_num, "DATA INGESTION (Master Market Data DB)")
    step_start = time.time()

    try:
        # Import ingestion modules
        from backend.turbomode.core_engine.ingest_via_ibkr import IBKRMarketDataIngestion
        from backend.turbomode.core_engine.training_symbols import get_training_symbols, CRYPTO_SYMBOLS

        # Initialize ingestion
        logger.info("Initializing IBKR market data ingestion...")
        ingestion = IBKRMarketDataIngestion()

        # Get all symbols: training (230) + crypto (3)
        training_symbols = get_training_symbols()
        all_symbols = sorted(training_symbols + CRYPTO_SYMBOLS)

        logger.info(f"Total symbols to ingest: {len(all_symbols)}")
        logger.info(f"  - Training stocks: {len(training_symbols)}")
        logger.info(f"  - Crypto: {len(CRYPTO_SYMBOLS)}")

        # Run ingestion (5 days to catch up on any missed data)
        logger.info("Starting ingestion (5-day lookback)...")
        results = ingestion.ingest_multiple_symbols(
            all_symbols,
            period='5d',
            timeframe='1d'
        )

        # Log results
        logger.info(f"[SUCCESS] Data ingestion completed")
        logger.info(f"  Symbols processed: {results['total_symbols']}")
        logger.info(f"  Successful: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Total candles ingested: {results['total_candles']:,}")

        step_duration = time.time() - step_start
        print_section_footer(step_num, step_duration)
        return True

    except Exception as e:
        logger.error(f"[FAILED] Data ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        step_duration = time.time() - step_start
        print_section_footer(step_num, step_duration)
        return False


def run_step_2_backtest_generation():
    """
    STEP 2: Feature/Label Generation and Backtesting
    Generates training data with proper labels from historical data
    """
    step_num = 2
    print_section_header(step_num, "BACKTEST DATA GENERATION")
    step_start = time.time()

    try:
        # Import backtest generator
        from backend.turbomode.core_engine.backtest_generator import BacktestGenerator
        from backend.turbomode.core_engine.training_symbols import get_training_symbols

        # Initialize backtest generator
        logger.info("Initializing backtest generator...")
        generator = BacktestGenerator(lookback_days=90)

        # Get training symbols (40 stocks) for backtesting
        symbols = get_training_symbols()
        logger.info(f"Generating backtest data for {len(symbols)} training symbols...")

        # Run backtest
        run_id = generator.run_full_backtest(symbols)

        if run_id:
            logger.info(f"[SUCCESS] Backtest generation completed - Run ID: {run_id}")
            step_duration = time.time() - step_start
            print_section_footer(step_num, step_duration)
            return True
        else:
            logger.error(f"[FAILED] Backtest generation returned no run_id")
            step_duration = time.time() - step_start
            print_section_footer(step_num, step_duration)
            return False

    except Exception as e:
        logger.error(f"[FAILED] Backtest generation failed: {e}")
        import traceback
        traceback.print_exc()
        step_duration = time.time() - step_start
        print_section_footer(step_num, step_duration)
        return False


def run_step_3_model_training():
    """
    STEP 3: Model Training - Single-Model-Per-Sector Architecture
    Trains 11 models total: 1 LightGBM model per sector

    ARCHITECTURE: Single-model-per-sector (1d/5% only)
    - Label: label_1d_5pct (1-day horizon, ±5% thresholds)
    - Model: Single LightGBM classifier per sector
    - Total: 11 models (one per sector)
    - Expected duration: 10-15 minutes
    """
    step_num = 3
    print_section_header(step_num, "MODEL TRAINING (Single-Model-Per-Sector)")
    step_start = time.time()

    try:
        # Import and run training orchestrator directly (not as subprocess)
        logger.info("Starting single-model training orchestrator...")
        logger.info("ARCHITECTURE: Single-model-per-sector (1d/5% only)")
        logger.info("This will train:")
        logger.info("  - 11 sectors")
        logger.info("  - 1 model per sector (LightGBM)")
        logger.info("  - Label: label_1d_5pct (1-day horizon, ±5% thresholds)")
        logger.info("  - Total: 11 models")
        logger.info("  - Expected duration: 13-15 minutes")
        logger.info("")

        # Import and run the orchestrator function directly
        from backend.turbomode.core_engine.train_all_sectors_optimized_orchestrator import train_all_sectors_optimized

        # Run training
        results = train_all_sectors_optimized()

        # Check results
        all_succeeded = all(r.get('status') == 'completed' for r in results.values())

        if all_succeeded:
            logger.info("[SUCCESS] Model training completed")
            logger.info(f"Trained {len(results)} sectors successfully")

            step_duration = time.time() - step_start
            print_section_footer(step_num, step_duration)
            return True
        else:
            failed_sectors = [s for s, r in results.items() if r.get('status') != 'completed']
            logger.error(f"[FAILED] Some sectors failed: {failed_sectors}")

            step_duration = time.time() - step_start
            print_section_footer(step_num, step_duration)
            return False

    except Exception as e:
        logger.error(f"[FAILED] Model training failed: {e}")
        import traceback
        traceback.print_exc()
        step_duration = time.time() - step_start
        print_section_footer(step_num, step_duration)
        return False


def run_step_4_production_scanner():
    """
    STEP 4: Production Scanner
    Scans all 230 symbols with trained models
    Generates BUY/SELL signals
    Updates TurboMode database
    Manages position state
    """
    step_num = 4
    print_section_header(step_num, "PRODUCTION SCANNER (Signal Generation)")
    step_start = time.time()

    try:
        # Import scanner
        from backend.turbomode.core_engine.overnight_scanner import ProductionScanner
        from backend.turbomode.core_engine.scanning_symbols import get_scanning_symbols

        # Initialize scanner (single-model architecture, no horizon parameter)
        logger.info("Initializing production scanner...")
        scanner = ProductionScanner()

        # Get all scanning symbols (230 stocks)
        symbols = get_scanning_symbols()
        logger.info(f"Scanning {len(symbols)} symbols with Fast Mode models...")

        # Run scanner on all symbols
        results = scanner.scan_all(max_signals_per_type=100)

        # Log results
        logger.info(f"[SUCCESS] Production scanner completed")
        logger.info(f"  BUY signals generated: {len(results.get('buy_signals', []))}")
        logger.info(f"  SELL signals generated: {len(results.get('sell_signals', []))}")
        logger.info(f"  Total signals: {len(results.get('buy_signals', [])) + len(results.get('sell_signals', []))}")

        step_duration = time.time() - step_start
        print_section_footer(step_num, step_duration)
        return True

    except Exception as e:
        logger.error(f"[FAILED] Production scanner failed: {e}")
        import traceback
        traceback.print_exc()
        step_duration = time.time() - step_start
        print_section_footer(step_num, step_duration)
        return False


def main():
    """
    Main orchestrator - runs full production pipeline
    """
    print("\n" + "=" * 80)
    print("TURBOMODE FULL PRODUCTION PIPELINE")
    print("=" * 80)
    print(f"Started: {PIPELINE_START_TIMESTAMP.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Track results
    step_results = {}

    # STEP 1: Data Ingestion
    step_results['ingestion'] = run_step_1_data_ingestion()
    if not step_results['ingestion']:
        logger.error("[CRITICAL] Data ingestion failed - pipeline cannot continue")
        return False

    # STEP 2: Backtest Generation
    step_results['backtest'] = run_step_2_backtest_generation()
    if not step_results['backtest']:
        logger.warning("[WARNING] Backtest generation failed - continuing with existing data")

    # STEP 3: Model Training
    step_results['training'] = run_step_3_model_training()
    if not step_results['training']:
        logger.error("[CRITICAL] Model training failed - pipeline cannot continue")
        return False

    # STEP 4: Production Scanner
    step_results['scanner'] = run_step_4_production_scanner()
    if not step_results['scanner']:
        logger.error("[WARNING] Production scanner failed")

    # Final summary
    total_duration = time.time() - PIPELINE_START_TIME
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Start Time:  {PIPELINE_START_TIMESTAMP.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:    {total_duration/60:.2f} minutes ({total_duration/3600:.2f} hours)")
    print()
    print("Step Results:")
    print(f"  [{'OK' if step_results['ingestion'] else 'FAIL'}] Step 1: Data Ingestion")
    print(f"  [{'OK' if step_results['backtest'] else 'WARN'}] Step 2: Backtest Generation")
    print(f"  [{'OK' if step_results['training'] else 'FAIL'}] Step 3: Model Training")
    print(f"  [{'OK' if step_results['scanner'] else 'WARN'}] Step 4: Production Scanner")
    print()

    # Determine overall success
    critical_steps_passed = step_results['ingestion'] and step_results['training']

    if critical_steps_passed:
        print("[SUCCESS] Production pipeline completed successfully")
        print("=" * 80)
        return True
    else:
        print("[FAILED] Production pipeline failed - critical steps did not complete")
        print("=" * 80)
        return False


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[CRITICAL ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
