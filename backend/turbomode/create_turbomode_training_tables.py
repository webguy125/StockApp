"""
Create TurboMode Training Tables in turbomode.db
Autonomous training database - completely separate from Slipstream
"""
import sqlite3
import os

# Path to TurboMode database
db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'turbomode.db')

print("=" * 80)
print("CREATING TURBOMODE TRAINING TABLES")
print("=" * 80)
print(f"Database: {db_path}")
print()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. TRADES TABLE - Store backtest results for training
print("[1/3] Creating 'trades' table...")
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

# 2. FEATURE_STORE TABLE - Store computed features for training
print("[2/3] Creating 'feature_store' table...")
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

# 3. PRICE_DATA TABLE - Optional: Store historical price data
print("[3/3] Creating 'price_data' table...")
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

# Commit changes
conn.commit()

# Verify tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]

print()
print("=" * 80)
print("VERIFICATION")
print("=" * 80)
print("All tables in turbomode.db:")
for table in sorted(tables):
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  - {table:30s} ({count:,} records)")

conn.close()

print()
print("=" * 80)
print("TURBOMODE TRAINING DATABASE READY")
print("=" * 80)
print("TurboMode can now train independently using turbomode.db")
print("No dependency on Slipstream's slipstream.db")
print("=" * 80)
