"""
Test database connectivity and data loading
"""
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

print("Testing database connection...")
print(f"Backend path: {backend_path}")

from advanced_ml.database.schema import AdvancedMLDatabase
from advanced_ml.backtesting.historical_backtest import HistoricalBacktest

# Database path
db_path = os.path.join(backend_path, "data", "advanced_ml_system.db")
print(f"\nDatabase path: {db_path}")
print(f"File exists: {os.path.exists(db_path)}")
print(f"File size: {os.path.getsize(db_path) / 1024 / 1024:.1f} MB")

# Test connection
print("\nTesting database connection...")
db = AdvancedMLDatabase(db_path)
print("[OK] Database connection successful")

# Test data loading
print("\nTesting data loading...")
backtest = HistoricalBacktest(db_path)
X, y = backtest.prepare_training_data()

print(f"\n[RESULTS]")
print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Labels: {len(y)}")
print(f"\nLabel distribution:")
import numpy as np
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")

print("\n[SUCCESS] All tests passed!")
