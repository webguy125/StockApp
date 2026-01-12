"""
Regenerate Training Data with Corrected Labels
This script:
1. Clears old training data from turbomode.db
2. Regenerates training data using fixed historical_backtest.py
3. Verifies label distribution is correct
TurboMode autonomous database - NO DEPENDENCY ON SLIPSTREAM
"""

import sys
import os
from datetime import datetime
import time

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import sqlite3
from turbomode.turbomode_backtest import TurboModeBacktest
from turbomode.core_symbols import get_all_core_symbols  # 43 curated stocks
from turbomode.checkpoint_manager import CheckpointManager
from turbomode.schema_guardrail import run_guardrail

# ==================== START TIMER ====================
START_TIME = time.time()
START_TIMESTAMP = datetime.now()

# Get absolute path to database (TurboMode autonomous database)
# backend_path is C:\StockApp\backend
db_path = os.path.join(backend_path, "data", "turbomode.db")

print("=" * 80)
print("REGENERATE TRAINING DATA - FIXED LABEL MAPPING")
print("=" * 80)
print(f"START TIME: {START_TIMESTAMP.strftime('%Y-%m-%d %I:%M:%S %p')}")
print(f"Database: {db_path}")
print(f"File exists: {os.path.exists(db_path)}")

# Step 0: SCHEMA GUARDRAIL - Complete workflow (validate → clean → restore → validate)
try:
    guardrail_result = run_guardrail(db_path, auto_clean=True, auto_restore=True)
    if guardrail_result['final_status'] != 'CLEAN':
        print(f"[ABORT] Guardrail failed to clean schema: {guardrail_result['final_status']}")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Guardrail failed: {e}")
    print("[ABORT] Cannot proceed - unable to ensure clean schema")
    sys.exit(1)

# Step 1: Clear old training data
print("\n" + "=" * 80)
print("STEP 1: CLEAR OLD TRAINING DATA")
print("=" * 80)

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Count existing backtest trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_type = 'backtest'")
        old_count = cursor.fetchone()[0]
        print(f"\nExisting backtest trades: {old_count}")

        if old_count > 0:
            # Get current curated symbol list
            curated_symbols = set(get_all_core_symbols())

            print(f"\n[INFO] Found {old_count} existing backtest trades")

            # Count unique symbols in database
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM trades WHERE trade_type = 'backtest'")
            unique_symbols_count = cursor.fetchone()[0]

            # If symbol count matches curated list, skip validation (fast path)
            if unique_symbols_count == len(curated_symbols):
                print(f"[FAST PATH] {unique_symbols_count} symbols in DB matches {len(curated_symbols)} curated stocks")
                print(f"[LEARNING] Continuous learning enabled - will update existing + add new samples")
            else:
                # Validation needed - count contaminated trades
                print(f"[VALIDATION] {unique_symbols_count} symbols in DB vs {len(curated_symbols)} curated stocks")
                placeholders = ','.join(['?' for _ in curated_symbols])
                cursor.execute(f"""
                    SELECT COUNT(*) FROM trades
                    WHERE trade_type = 'backtest'
                    AND symbol NOT IN ({placeholders})
                """, list(curated_symbols))
                contaminated_count = cursor.fetchone()[0]

                if contaminated_count > 0:
                    print(f"[CLEANUP] Removing {contaminated_count} trades from non-curated stocks")

                    # DELETE only non-curated stocks (contamination)
                    cursor.execute(f"""
                        DELETE FROM trades
                        WHERE trade_type = 'backtest'
                        AND symbol NOT IN ({placeholders})
                    """, list(curated_symbols))
                    conn.commit()

                    # Count what remains
                    cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_type = 'backtest'")
                    remaining = cursor.fetchone()[0]
                    print(f"[OK] Kept {remaining} trades from curated stocks")
                    print(f"[LEARNING] New data will UPDATE existing + ADD fresh samples")
                else:
                    print(f"[OK] All {old_count} trades are from curated stocks - no cleanup needed")
                    print(f"[LEARNING] Will update existing data + add new samples")
        else:
            print("No existing data - starting fresh")
    except sqlite3.OperationalError:
        print("Database tables don't exist yet - will be created by backtest")

    conn.close()
else:
    print("Database doesn't exist yet - will be created by backtest")

# Step 2: Generate new training data
print("\n" + "=" * 80)
print("STEP 2: GENERATE NEW TRAINING DATA")
print("=" * 80)

backtest = TurboModeBacktest(turbomode_db_path=db_path)

# Use 43 curated stocks (40 stocks + 3 crypto)
symbols = get_all_core_symbols()

print(f"\n[CURATED] Using {len(symbols)} carefully selected stocks")
print("[INFO] Stratified by sector + market cap for optimal signal quality")
print("[INFO] Expected processing time: ~40-80 minutes (10 years of data)")
print("[INFO] Expected output: ~150,000-180,000 high-quality training samples")
print("[WHY] 10-year lookback captures multiple market cycles for robust models")
print("[LABEL LOGIC] Canonical: +5% = BUY, -5% = SELL, else = HOLD (5-day holding period)")

# Initialize checkpoint manager
checkpoint = CheckpointManager()
checkpoint.mark_backtest_start()

# Get remaining symbols (skip already processed)
symbols_to_process = checkpoint.get_remaining_symbols(symbols)

if len(symbols_to_process) == 0:
    print("\n[CHECKPOINT] All symbols already processed!")
    print(checkpoint.get_summary())
else:
    # Run backtest to generate training data
    print(f"\nGenerating training data...")
    print(f"  Symbols: {len(symbols_to_process)} (of {len(symbols)} total)")
    print(f"  Years of history: 10")
    print(f"  Hold period: 5 days")
    print(f"  Buy threshold: +5%")
    print(f"  Sell threshold: -5%")

    # Process each symbol with checkpointing
    for i, symbol in enumerate(symbols_to_process, 1):
        print(f"\n[{i}/{len(symbols_to_process)}] Processing {symbol}...")
        try:
            # Generate backtest samples for single symbol (10 years = ~3650 days)
            result = backtest.generate_backtest_samples(
                symbol=symbol,
                lookback_days=3650
            )
            samples = result.get('total_samples', 0)

            # Mark symbol as complete
            checkpoint.mark_symbol_complete(symbol, samples)

        except Exception as e:
            print(f"[ERROR] Failed to process {symbol}: {e}")
            checkpoint.mark_symbol_failed(symbol, str(e))
            continue

    # Mark backtest as complete
    checkpoint.mark_backtest_complete()

# Get final results for validation (sum of all completed symbols)
results = {'total_samples': checkpoint.state['backtest']['total_samples']}

# Step 3: Verify label distribution
print("\n" + "=" * 80)
print("STEP 3: VERIFY LABEL DISTRIBUTION")
print("=" * 80)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT outcome, COUNT(*)
    FROM trades
    WHERE trade_type = 'backtest'
    GROUP BY outcome
""")

print("\nLabel distribution in database:")
total = 0
buy_count = 0
hold_count = 0
sell_count = 0

for row in cursor.fetchall():
    outcome, count = row
    total += count
    if outcome == 'buy':
        buy_count = count
    elif outcome == 'hold':
        hold_count = count
    elif outcome == 'sell':
        sell_count = count
    print(f"  {outcome}: {count} ({count/total*100:.1f}%)")

conn.close()

print(f"\nTotal samples: {total}")

# Validate
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

if total == 0:
    print("[ERROR] No training data generated!")
    sys.exit(1)

if buy_count == 0 and sell_count == 0:
    print("[ERROR] No BUY or SELL labels found!")
    sys.exit(1)

# Check for reasonable distribution
buy_pct = buy_count / total * 100 if total > 0 else 0
sell_pct = sell_count / total * 100 if total > 0 else 0

print(f"\n[OK] Training data generated successfully")
print(f"[OK] BUY labels: {buy_count} ({buy_pct:.1f}%)")
print(f"[OK] SELL labels: {sell_count} ({sell_pct:.1f}%)")
print(f"[OK] HOLD labels: {hold_count} ({hold_count/total*100:.1f}%)")

if buy_pct < 5 or sell_pct < 5:
    print(f"\n[WARNING] Very low BUY ({buy_pct:.1f}%) or SELL ({sell_pct:.1f}%) percentage")
    print("This may indicate the win/loss thresholds are too aggressive")
else:
    print("\n[OK] Label distribution looks healthy")

print("\n" + "=" * 80)
print("NEXT STEP: Run train_turbomode_models.py to retrain all models")
print("=" * 80)

# ==================== END TIMER ====================
END_TIME = time.time()
END_TIMESTAMP = datetime.now()
TOTAL_SECONDS = END_TIME - START_TIME
HOURS = int(TOTAL_SECONDS // 3600)
MINUTES = int((TOTAL_SECONDS % 3600) // 60)
SECONDS = int(TOTAL_SECONDS % 60)

print("\n" + "=" * 80)
print("BACKTEST TIMING SUMMARY")
print("=" * 80)
print(f"START TIME:  {START_TIMESTAMP.strftime('%Y-%m-%d %I:%M:%S %p')}")
print(f"END TIME:    {END_TIMESTAMP.strftime('%Y-%m-%d %I:%M:%S %p')}")
print(f"TOTAL TIME:  {HOURS} hours, {MINUTES} minutes, {SECONDS} seconds")
print(f"             ({TOTAL_SECONDS/3600:.2f} hours)")
print("=" * 80)
