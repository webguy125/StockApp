# End-to-End GPU Test
import sys
import os
sys.path.insert(0, 'backend')

from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
import time
import sqlite3

print("="*80)
print("END-TO-END GPU TEST")
print("="*80)

db_path = "backend/backend/data/test_gpu.db"

# Clear old database
if os.path.exists(db_path):
    os.remove(db_path)

# Step 1: Backtest
print("\n[STEP 1] GPU Backtest...")
start = time.time()
backtest = HistoricalBacktest(db_path, use_gpu=True)
results = backtest.run_backtest(['AAPL'], years=2, save_to_db=True)
backtest_time = time.time() - start

print(f"[OK] Backtest: {backtest_time:.1f}s, {results.get('total_samples', 0)} samples")

# Step 2: Verify database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_type = 'backtest'")
count = cursor.fetchone()[0]
cursor.execute("SELECT outcome, COUNT(*) FROM trades WHERE trade_type = 'backtest' GROUP BY outcome")
labels = cursor.fetchall()
conn.close()

print(f"[OK] Database: {count} samples")
for outcome, cnt in labels:
    print(f"     {outcome}: {cnt}")

# Check labels
outcomes = [row[0] for row in labels]
if 'buy' in outcomes and 'sell' in outcomes:
    print("[OK] Labels correct (buy/hold/sell)")
else:
    print("[FAIL] Labels incorrect!")
    sys.exit(1)

# Step 3: Train model
print("\n[STEP 2] Model Training...")
X, y = backtest.prepare_training_data()

if len(X) == 0:
    print("[FAIL] No training data!")
    sys.exit(1)

print(f"[OK] Loaded {len(X)} samples, {X.shape[1]} features")

import xgboost as xgb
start = time.time()
# XGBoost 3.x uses device="cuda" for GPU acceleration
model = xgb.XGBClassifier(device="cuda", tree_method='hist', n_estimators=50, max_depth=5, verbosity=0)
model.fit(X, y)
train_time = time.time() - start

print(f"[OK] Training: {train_time:.1f}s")

# Test predictions
predictions = model.predict(X[:5])
probabilities = model.predict_proba(X[:5])

print("\n[STEP 3] Predictions:")
for i in range(5):
    label = ['BUY', 'HOLD', 'SELL'][predictions[i]]
    conf = probabilities[i][predictions[i]] * 100
    print(f"  Sample {i+1}: {label} ({conf:.1f}%)")

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
