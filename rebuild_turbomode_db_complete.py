"""
Rebuild Complete TurboMode Database
Creates BOTH signal tables AND training tables in turbomode.db
"""
import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Database path
db_path = os.path.join(project_root, "backend", "data", "turbomode.db")

print("=" * 80)
print("REBUILD COMPLETE TURBOMODE DATABASE")
print("=" * 80)
print(f"Database path: {db_path}")
print()

# Check if database exists
if os.path.exists(db_path):
    print("[WARNING] Database file already exists!")
    response = input("Do you want to DELETE and rebuild? (yes/no): ")
    if response.lower() != 'yes':
        print("[ABORT] Database rebuild cancelled")
        sys.exit(0)

    # Delete existing database
    os.remove(db_path)
    print("[OK] Existing database deleted")
    print()

# Step 1: Create training tables
print("=" * 80)
print("STEP 1: CREATE TRAINING TABLES")
print("=" * 80)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. TRADES TABLE
print("[1/4] Creating 'trades' table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_date TEXT,
    exit_price REAL,
    position_size REAL DEFAULT 1.0,

    -- Outcome
    outcome TEXT DEFAULT 'open',
    profit_loss REAL,
    profit_loss_pct REAL,
    exit_reason TEXT,

    -- Entry Features (JSON - 179 features)
    entry_features_json TEXT,

    -- Trade metadata
    trade_type TEXT DEFAULT 'backtest',
    strategy TEXT,
    notes TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")
print("   [OK] 'trades' table created")

# 2. FEATURE_STORE TABLE
print("[2/4] Creating 'feature_store' table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS feature_store (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,

    -- Stored as JSON for flexibility (179 features)
    features_json TEXT NOT NULL,

    -- Quick lookup fields (most important features)
    rsi_14 REAL,
    macd_histogram REAL,
    volume_ratio REAL,
    trend_strength REAL,
    momentum_score REAL,
    volatility_score REAL,

    feature_version TEXT DEFAULT 'v1',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
)
""")
print("   [OK] 'feature_store' table created")

# 3. PRICE_DATA TABLE
print("[3/4] Creating 'price_data' table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS price_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, timeframe)
)
""")
print("   [OK] 'price_data' table created")

# 4. TRAINING_RUNS TABLE
print("[4/4] Creating 'training_runs' table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    sector TEXT,
    timeframe TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    samples_count INTEGER,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    hyperparameters TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")
print("   [OK] 'training_runs' table created")

conn.commit()

print()
print("=" * 80)
print("STEP 2: CREATE SIGNAL TABLES (from database_schema.py)")
print("=" * 80)

# Use TurboModeDB to create signal tables
from backend.turbomode.database_schema import TurboModeDB

# Close current connection
conn.close()

# Initialize TurboModeDB (will add signal tables)
db = TurboModeDB(db_path=db_path)

print()
print("=" * 80)
print("DATABASE REBUILD COMPLETE")
print("=" * 80)
print(f"Database: {db_path}")
print()

# Show all tables
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()

print("All tables created:")
for (table_name,) in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  - {table_name:30s} ({count} records)")

conn.close()

print()
print("[OK] Complete TurboMode database rebuilt successfully!")
print()
print("Training tables: trades, feature_store, price_data, training_runs")
print("Signal tables: active_signals, signal_history, sector_stats")
print("=" * 80)
