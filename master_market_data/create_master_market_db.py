"""
Create Master Market Data Database
Shared, read-only database for raw market data (OHLCV, fundamentals, metadata)

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Location: C:\StockApp\master_market_data\market_data.db

This database is READ-ONLY for TurboMode and Slipstream.
Only admin has write access for maintenance.
"""

import sqlite3
import os
from datetime import datetime

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'market_data.db')

print("=" * 80)
print("MASTER MARKET DATA DB - INITIALIZATION")
print("=" * 80)
print(f"Location: {DB_PATH}")
print()

# Connect to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ============================================================================
# TABLE 1: CANDLES (OHLCV Data)
# ============================================================================
print("Creating table: candles")
cursor.execute("""
CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    timeframe TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    adjusted_close REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, timeframe)
)
""")

# Indexes for performance
cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol ON candles(symbol)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_timestamp ON candles(symbol, timestamp)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timeframe ON candles(timeframe)")
print("[OK] Table created with 4 indexes")

# ============================================================================
# TABLE 2: FUNDAMENTALS (Company Financial Data)
# ============================================================================
print("\nCreating table: fundamentals")
cursor.execute("""
CREATE TABLE IF NOT EXISTS fundamentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    market_cap REAL,
    enterprise_value REAL,
    trailing_pe REAL,
    forward_pe REAL,
    peg_ratio REAL,
    price_to_book REAL,
    price_to_sales REAL,
    enterprise_to_revenue REAL,
    enterprise_to_ebitda REAL,
    profit_margin REAL,
    operating_margin REAL,
    return_on_assets REAL,
    return_on_equity REAL,
    revenue REAL,
    revenue_per_share REAL,
    quarterly_revenue_growth REAL,
    gross_profit REAL,
    ebitda REAL,
    net_income REAL,
    diluted_eps REAL,
    quarterly_earnings_growth REAL,
    total_cash REAL,
    total_cash_per_share REAL,
    total_debt REAL,
    debt_to_equity REAL,
    current_ratio REAL,
    book_value_per_share REAL,
    operating_cash_flow REAL,
    levered_free_cash_flow REAL,
    beta REAL,
    fifty_two_week_change REAL,
    short_ratio REAL,
    short_percent_of_float REAL,
    shares_outstanding REAL,
    shares_short REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol ON fundamentals(symbol)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_date ON fundamentals(date)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_date ON fundamentals(symbol, date)")
print("[OK] Table created with 3 indexes")

# ============================================================================
# TABLE 3: SPLITS (Stock Split Events)
# ============================================================================
print("\nCreating table: splits")
cursor.execute("""
CREATE TABLE IF NOT EXISTS splits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    split_ratio TEXT NOT NULL,
    split_factor REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_splits_symbol ON splits(symbol)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_splits_date ON splits(date)")
print("[OK] Table created with 2 indexes")

# ============================================================================
# TABLE 4: DIVIDENDS (Dividend Payout Records)
# ============================================================================
print("\nCreating table: dividends")
cursor.execute("""
CREATE TABLE IF NOT EXISTS dividends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    dividend_amount REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividends(symbol)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_dividends_date ON dividends(date)")
print("[OK] Table created with 2 indexes")

# ============================================================================
# TABLE 5: SYMBOL_METADATA (Ticker Information)
# ============================================================================
print("\nCreating table: symbol_metadata")
cursor.execute("""
CREATE TABLE IF NOT EXISTS symbol_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    company_name TEXT,
    sector TEXT,
    industry TEXT,
    country TEXT,
    exchange TEXT,
    currency TEXT,
    quote_type TEXT,
    long_business_summary TEXT,
    website TEXT,
    employees INTEGER,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    phone TEXT,
    is_active INTEGER DEFAULT 1,
    first_trade_date DATE,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON symbol_metadata(symbol)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_sector ON symbol_metadata(sector)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_industry ON symbol_metadata(industry)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_exchange ON symbol_metadata(exchange)")
print("[OK] Table created with 4 indexes")

# ============================================================================
# TABLE 6: SECTOR_MAPPINGS (Industry Classifications)
# ============================================================================
print("\nCreating table: sector_mappings")
cursor.execute("""
CREATE TABLE IF NOT EXISTS sector_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    sector TEXT NOT NULL,
    industry TEXT,
    sub_industry TEXT,
    effective_date DATE NOT NULL,
    end_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, effective_date)
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_mappings_symbol ON sector_mappings(symbol)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_mappings_sector ON sector_mappings(sector)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_mappings_effective_date ON sector_mappings(effective_date)")
print("[OK] Table created with 3 indexes")

# ============================================================================
# TABLE 7: DATA_QUALITY_LOG (Track Data Issues)
# ============================================================================
print("\nCreating table: data_quality_log")
cursor.execute("""
CREATE TABLE IF NOT EXISTS data_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    table_name TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    issue_description TEXT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved INTEGER DEFAULT 0,
    resolved_at DATETIME
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_log_symbol ON data_quality_log(symbol)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_log_table ON data_quality_log(table_name)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_log_resolved ON data_quality_log(resolved)")
print("[OK] Table created with 3 indexes")

# ============================================================================
# TABLE 8: DB_METADATA (Database Version and Info)
# ============================================================================
print("\nCreating table: db_metadata")
cursor.execute("""
CREATE TABLE IF NOT EXISTS db_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Insert initial metadata
cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
               ('version', '1.0'))
cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
               ('created_at', datetime.now().isoformat()))
cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
               ('architecture_version', '1.1'))
cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
               ('purpose', 'Shared read-only market data for TurboMode and Slipstream'))
print("[OK] Table created with metadata")

# Commit all changes
conn.commit()

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()

print(f"\nTotal Tables Created: {len(tables)}")
for table in tables:
    table_name = table[0]
    if table_name != 'sqlite_sequence':
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        print(f"  {table_name:<25} {len(columns)} columns, {count} rows")

# Get database size
db_size = os.path.getsize(DB_PATH) / 1024  # KB
print(f"\nDatabase Size: {db_size:.2f} KB")

conn.close()

print("\n" + "=" * 80)
print("MASTER MARKET DATA DB - INITIALIZATION COMPLETE")
print("=" * 80)
print(f"Location: {DB_PATH}")
print("Status: Ready for data ingestion")
print("Access: Read-only for TurboMode and Slipstream")
print("=" * 80)
