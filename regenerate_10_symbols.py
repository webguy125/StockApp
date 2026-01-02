"""
Quick 10-Symbol Training Data Regeneration (179 Features)
==========================================================

This script regenerates training data with FULL 179 FEATURES for 10 test symbols.

AFTER THIS COMPLETES:
1. Verify 179 features in output
2. Run full regeneration overnight with ALL 510 symbols:
   cd backend/turbomode
   rm -f backtest_checkpoint.json
   ../../venv/Scripts/python.exe generate_backtest_data.py

Expected Runtime: ~5-10 minutes
Expected Output: ~4,360 training samples (436 per symbol × 10 symbols)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from backend.advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from backend.advanced_ml.database.schema import AdvancedMLDatabase

# Database path
db_path = "backend/data/advanced_ml_system.db"

print("=" * 80)
print("REGENERATE TRAINING DATA - 10 SYMBOLS TEST (179 FEATURES)")
print("=" * 80)
print(f"Database: {db_path}")
print(f"File exists: {os.path.exists(db_path)}")
print()

# Test symbols (diverse set)
TEST_SYMBOLS = [
    'AAPL',  # Tech - stable
    'MSFT',  # Tech - stable
    'GOOGL', # Tech - stable
    'TSLA',  # Tech - volatile
    'NVDA',  # Tech - volatile
    'JPM',   # Financial
    'JNJ',   # Healthcare
    'XOM',   # Energy
    'WMT',   # Retail
    'DIS'    # Entertainment
]

print("=" * 80)
print("STEP 1: CLEAR OLD TRAINING DATA")
print("=" * 80)
print()

# Connect to database and clear old data
db = AdvancedMLDatabase(db_path)
conn = db.get_connection()
cursor = conn.cursor()

# Count old backtest trades
cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_type = 'backtest'")
old_count = cursor.fetchone()[0]
print(f"Old backtest trades: {old_count}")
print()

# Clear old backtest data
print("Clearing old backtest data...")
cursor.execute("DELETE FROM trades WHERE trade_type = 'backtest'")
conn.commit()
conn.close()

print(f"[OK] Deleted {old_count} old training samples")
print()

print("=" * 80)
print("STEP 2: GENERATE NEW TRAINING DATA (179 FEATURES)")
print("=" * 80)
print()

# Initialize backtest with GPU
backtest = HistoricalBacktest(db_path, use_gpu=True)

print(f"[TEST MODE] Using {len(TEST_SYMBOLS)} test symbols")
print(f"[INFO] GPU-accelerated batch processing: ~5-10 minutes")
print(f"[INFO] Expected output: ~4,360 training samples")
print()

# Generate training data
print(f"Generating training data...")
print(f"  Symbols: {len(TEST_SYMBOLS)}")
print(f"  Years of history: 2")
print(f"  Hold period: 14 days")
print(f"  Win threshold: +10%")
print(f"  Loss threshold: -5%")
print()

results = backtest.run_backtest(TEST_SYMBOLS, years=2, save_to_db=True)

print()
print("=" * 80)
print("STEP 3: VERIFY FEATURE COUNT")
print("=" * 80)
print()

# Load data and check feature count
X, y = backtest.prepare_training_data()

print(f"[DATA] Training data prepared")
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Buy: {(y == 0).sum()}")
print(f"  Hold: {(y == 1).sum()}")
print(f"  Sell: {(y == 2).sum()}")
print()

if X.shape[1] == 179:
    print("[SUCCESS] ✓ Data has correct 179 features!")
    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Train models with 179 features:")
    print("   cd backend/turbomode")
    print("   ../../venv/Scripts/python.exe train_turbomode_models.py")
    print()
    print("2. After verifying improved accuracy, regenerate FULL dataset overnight:")
    print("   cd backend/turbomode")
    print("   rm -f backtest_checkpoint.json")
    print("   ../../venv/Scripts/python.exe generate_backtest_data.py")
    print()
    print("   Expected: ~220,000 samples from 510 S&P 500 symbols")
    print("   Runtime: 10-12 hours")
    print()
else:
    print(f"[ERROR] ✗ Data has {X.shape[1]} features instead of 179!")
    print()
    print("This means extract_features_batch() is not working correctly.")
    print("Check the output above for warnings or errors.")
    print()

print("=" * 80)
print()
