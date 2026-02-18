#!/usr/bin/env python
"""
14-Day Swing Trading Production Pipeline

ARCHITECTURE:
- Unified 14-day holding period
- ±5% outcome thresholds (BUY >= +5%, SELL <= -5%, else HOLD)
- 66 models total: 11 sectors × 6 models per sector
- 5 base models: lightgbm_gpu, catboost_gpu, xgb_hist_gpu, xgb_linear, random_forest
- 1 meta-learner per sector
- Risk-adjusted ranking with WIN_FRACTION = 3.0

This script orchestrates the complete end-to-end production pipeline:
1. Generate 14-day backtest samples
2. Train all 66 models
3. Run production scanner (generate signals)
4. Run adaptive ranking engine

NO LEGACY LOGIC:
- No label_1d_5pct
- No backtest_generator.py
- No horizon-based models
- No percent-threshold variants

Author: TurboMode Production Team
Date: 2026-01-24
"""

import sys
import os
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
turbomode_dir = os.path.dirname(current_dir)
backend_dir = os.path.dirname(turbomode_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import traceback
from datetime import datetime

# ==================================================================================
# TIMER WRAPPER
# ==================================================================================

def timed(stage_name, func, *args, **kwargs):
    """
    Execute a function with timing and clear stage markers.

    Args:
        stage_name: Name of the pipeline stage
        func: Function to execute
        *args, **kwargs: Arguments to pass to function

    Returns:
        Result from function execution
    """
    print("\n" + "=" * 80)
    print(f"STAGE: {stage_name}")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    start_time = time.time()

    try:
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        print()
        print("-" * 80)
        print(f"STAGE COMPLETE: {stage_name}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print("-" * 80)
        print()

        return result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print()
        print("-" * 80)
        print(f"STAGE FAILED: {stage_name}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"Error: {e}")
        print("-" * 80)
        print()

        raise


# ==================================================================================
# STAGE 0: MASTER MARKET DATA INGESTION
# File: backend/turbomode/core_engine/ingest_master_market_data.py
# ==================================================================================

def run_stage_0_ingestion():
    """
    Ingest market data for all symbols.

    Uses: master_market_data/ingestion_runner.py
    - Fetches OHLCV data for all CORE_230 symbols
    - Updates master market data database
    """
    print("Importing ingestion runner...")
    from backend.turbomode.core_engine.ingest_master_market_data import run_full_ingestion

    print("Running master market data ingestion...")
    print()

    try:
        # Full historical ingestion (10 years) for training data
        run_full_ingestion(period='10y')
        return True
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        raise


# ==================================================================================
# STAGE 1: 14-DAY BACKTEST GENERATION
# File: backend/turbomode/core_engine/generate_backtest_data.py
# ==================================================================================

def run_backtest_stage():
    """
    Generate 14-day backtest samples for all 230 CORE_230 symbols.

    Uses: backend/turbomode/core_engine/generate_backtest_data.py
    - Processes all 230 training symbols
    - Holding period: 14 days
    - Buy threshold: +5%
    - Sell threshold: -5%
    - Generates BUY/SELL/HOLD outcomes
    - Expected output: ~800,000-1,000,000 backtest samples
    """
    print("Running full batch backtest generation for 230 symbols...")
    print("This will take approximately 5 minutes for 10 years of historical data")
    print()

    import subprocess
    import os

    # Run the dedicated batch backtest generator script
    # This processes all 230 symbols and writes backtest samples to backend/data/turbomode.db
    script_path = os.path.join(project_root, "backend", "turbomode", "core_engine", "generate_backtest_data.py")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Batch backtest generator not found: {script_path}")

    print(f"[BACKTEST] Executing: {script_path}")
    print()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=project_root,
        capture_output=False
    )

    if result.returncode != 0:
        raise RuntimeError(f"Batch backtest generation failed with exit code {result.returncode}")

    print()
    print("[OK] Batch backtest generation completed successfully")
    return True


# ==================================================================================
# STAGE 2: MODEL TRAINING (66 MODELS)
# File: backend/turbomode/core_engine/train_all_sectors_optimized_orchestrator.py
# ==================================================================================

def run_training_stage():
    """
    Train all 66 models (11 sectors × 6 models).

    Uses: backend/turbomode/core_engine/train_all_sectors_optimized_orchestrator.py
    - 5 base models per sector: lightgbm_gpu, catboost_gpu, xgb_hist_gpu, xgb_linear, random_forest
    - 1 meta-learner per sector
    - Label source: 14-day outcomes from trades table (±5% thresholds)
    """
    print("Importing training orchestrator...")
    from backend.turbomode.core_engine import train_all_sectors_optimized_orchestrator

    print("Training all 66 models...")
    print("Architecture: 11 sectors × 6 models (5 base + 1 meta-learner)")
    print()

    # Check for main/run function
    if hasattr(train_all_sectors_optimized_orchestrator, 'train_all_sectors_optimized'):
        return train_all_sectors_optimized_orchestrator.train_all_sectors_optimized()
    elif hasattr(train_all_sectors_optimized_orchestrator, 'main'):
        return train_all_sectors_optimized_orchestrator.main()
    else:
        raise RuntimeError("Training orchestrator has no callable entrypoint (train_all_sectors_optimized or main)")


# ==================================================================================
# STAGE 3: PRODUCTION SCANNER
# File: backend/turbomode/core_engine/overnight_scanner.py
# ==================================================================================

def run_scanner_stage():
    """
    Run production scanner to generate BUY/SELL signals.

    Uses: backend/turbomode/core_engine/overnight_scanner.py
    - Loads 14-day models (5 base + 1 meta per sector)
    - Generates predictions for all scanning symbols
    - Writes signals to database
    """
    print("Importing production scanner...")
    from backend.turbomode.core_engine import overnight_scanner

    print("Running production scanner...")
    print("Generating BUY/SELL signals using 14-day ensemble models...")
    print()

    # Instantiate scanner and run
    if hasattr(overnight_scanner, 'ProductionScanner'):
        scanner = overnight_scanner.ProductionScanner()
        if hasattr(scanner, 'scan_all'):
            results = scanner.scan_all(max_signals_per_type=100)
            print(f"Scanner complete: {len(results.get('buy_signals', []))} BUY, {len(results.get('sell_signals', []))} SELL")
            return results
        else:
            raise RuntimeError("ProductionScanner has no scan_all() method")
    elif hasattr(overnight_scanner, 'main'):
        return overnight_scanner.main()
    else:
        raise RuntimeError("overnight_scanner has no callable entrypoint (ProductionScanner or main)")


# ==================================================================================
# STAGE 4: ADAPTIVE RANKING ENGINE
# File: backend/turbomode/adaptive_stock_ranker.py
# ==================================================================================

def run_ranking_stage():
    """
    Run adaptive stock ranking engine.

    Uses: backend/turbomode/adaptive_stock_ranker.py
    - Calculates risk-adjusted win rates (WIN_FRACTION = 3.0)
    - Applies directional fallback (pnl > 0 for BUY, pnl < 0 for SELL)
    - Generates composite scores and top 10 rankings
    """
    print("Importing adaptive stock ranker...")
    from backend.turbomode.core_engine import adaptive_stock_ranker

    print("Running adaptive ranking engine...")
    print("Calculating risk-adjusted win rates with WIN_FRACTION = 3.0...")
    print()

    # Instantiate ranker and run
    if hasattr(adaptive_stock_ranker, 'AdaptiveStockRanker'):
        ranker = adaptive_stock_ranker.AdaptiveStockRanker()
        results = ranker.run_analysis()
        if results:
            print(f"Ranking complete: Top 10 = {[s['symbol'] for s in results['top_10']]}")
        return results
    elif hasattr(adaptive_stock_ranker, 'main'):
        return adaptive_stock_ranker.main()
    else:
        raise RuntimeError("adaptive_stock_ranker has no callable entrypoint (AdaptiveStockRanker or main)")


# ==================================================================================
# MAIN ORCHESTRATOR
# ==================================================================================

def main():
    """
    Execute complete 14-day swing trading production pipeline.
    """
    print("\n" + "=" * 80)
    print("14-DAY SWING TRADING PRODUCTION PIPELINE")
    print("=" * 80)
    print(f"Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("CONFIGURATION:")
    print("  Holding period: 14 days")
    print("  Thresholds: BUY >= +5%, SELL <= -5%")
    print("  Models: 66 total (11 sectors × 6 models)")
    print("  Base models: lightgbm_gpu, catboost_gpu, xgb_hist_gpu, xgb_linear, random_forest")
    print("  Meta-learner: 1 per sector")
    print("=" * 80)

    pipeline_start = time.time()
    stage_results = {}

    try:
        # STAGE 0: Ingestion
        stage_results['ingestion'] = timed(
            "0. Master Market Data Ingestion",
            run_stage_0_ingestion
        )

        # STAGE 1: Backtest
        stage_results['backtest'] = timed(
            "1. 14-Day Backtest Generation",
            run_backtest_stage
        )

        # STAGE 1 VALIDATION: Ensure all 230 symbols have backtest samples
        print("\n" + "=" * 80)
        print("STAGE 1 VALIDATION: Checking backtest sample coverage")
        print("=" * 80)

        import sqlite3
        turbomode_db_path = os.path.join(project_root, "backend", "data", "turbomode.db")
        conn = sqlite3.connect(turbomode_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM trades WHERE trade_type = 'backtest'")
            row = cursor.fetchone()
            distinct_symbols = row[0] if row and row[0] is not None else 0

            cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_type = 'backtest'")
            total_samples = cursor.fetchone()[0]

            expected_symbols = 230

            print(f"  Total backtest samples: {total_samples:,}")
            print(f"  Distinct symbols: {distinct_symbols}")
            print(f"  Expected symbols: {expected_symbols}")

            if distinct_symbols < expected_symbols:
                print()
                print(f"[ERROR] Backtest validation FAILED: expected {expected_symbols} symbols, found {distinct_symbols}")
                print("[ERROR] Not all symbols have backtest samples - cannot proceed with training")
                raise RuntimeError(f"Backtest validation failed: {distinct_symbols}/{expected_symbols} symbols")

            print()
            print(f"[OK] Backtest validation PASSED: All {distinct_symbols} symbols have training data")
        finally:
            conn.close()

        print("=" * 80)
        print()

        # STAGE 2: Training
        stage_results['training'] = timed(
            "2. Model Training (66 models)",
            run_training_stage
        )

        # STAGE 3: Scanner
        stage_results['scanner'] = timed(
            "3. Production Scanner",
            run_scanner_stage
        )

        # STAGE 4: Ranking
        stage_results['ranking'] = timed(
            "4. Adaptive Ranking Engine",
            run_ranking_stage
        )

        pipeline_end = time.time()
        total_duration = pipeline_end - pipeline_start

        # Final summary
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Start time: {datetime.fromtimestamp(pipeline_start).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {datetime.fromtimestamp(pipeline_end).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print()
        print("Stage Results:")
        print("  [OK] Stage 1: 14-Day Backtest Generation")
        print("  [OK] Stage 2: Model Training (66 models)")
        print("  [OK] Stage 3: Production Scanner")
        print("  [OK] Stage 4: Adaptive Ranking Engine")
        print()
        print("[SUCCESS] Pipeline completed successfully")
        print("=" * 80)
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline execution interrupted by user")
        print("=" * 80)
        return 130

    except Exception as e:
        pipeline_end = time.time()
        total_duration = pipeline_end - pipeline_start

        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("Traceback:")
        traceback.print_exc()
        print()
        print(f"Total duration before failure: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print("=" * 80)
        print()

        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
