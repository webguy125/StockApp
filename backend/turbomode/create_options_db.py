"""
Create options predictions logging database
Run this once to set up the database schema
"""

import sqlite3
import os
from datetime import datetime

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_logs', 'options_predictions.db')

# Ensure directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def create_database():
    """Create options predictions database with full schema"""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop table if exists (for fresh start)
    cursor.execute("DROP TABLE IF EXISTS options_predictions_log")

    # Create main predictions table
    cursor.execute("""
        CREATE TABLE options_predictions_log (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            -- Stock info
            symbol TEXT NOT NULL,
            stock_price_entry REAL,
            ml_signal TEXT,
            ml_confidence REAL,
            ml_target_price REAL,
            historical_vol_30d REAL,

            -- Option details
            option_type TEXT NOT NULL,
            strike REAL NOT NULL,
            expiration_date DATE NOT NULL,
            dte INTEGER,
            entry_premium REAL,
            entry_bid REAL,
            entry_ask REAL,

            -- Greeks
            entry_delta REAL,
            entry_gamma REAL,
            entry_theta REAL,
            entry_vega REAL,
            entry_rho REAL,
            entry_iv REAL,

            -- Liquidity
            entry_open_interest INTEGER,
            entry_volume INTEGER,

            -- Scoring
            rules_score INTEGER,
            ml_score REAL,
            hybrid_score REAL,

            -- Tracking
            tracking_complete BOOLEAN DEFAULT 0,
            outcome_checked_at DATETIME,
            max_premium_14d REAL,
            max_premium_date DATETIME,
            final_premium_14d REAL,
            hit_10pct_target BOOLEAN,
            days_to_target INTEGER,
            max_profit_pct REAL
        )
    """)

    # Create indexes for performance
    cursor.execute("CREATE INDEX idx_symbol ON options_predictions_log(symbol)")
    cursor.execute("CREATE INDEX idx_created_at ON options_predictions_log(created_at)")
    cursor.execute("CREATE INDEX idx_tracking ON options_predictions_log(tracking_complete)")
    cursor.execute("CREATE INDEX idx_expiration ON options_predictions_log(expiration_date)")

    conn.commit()

    print(f"[OK] Database created: {DB_PATH}")
    print(f"[OK] Table: options_predictions_log")
    print(f"[OK] Indexes created: symbol, created_at, tracking, expiration")

    # Verify
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"[OK] Tables in database: {tables}")

    conn.close()

    return DB_PATH

if __name__ == '__main__':
    db_path = create_database()
    print(f"\n✅ Options logging database ready!")
    print(f"   Path: {db_path}")
    print(f"   Ready to start logging predictions!")
