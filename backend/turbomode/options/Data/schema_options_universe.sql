-- Options Universe Database Schema
-- Stores historical options chains from EODHD for training and backtesting

CREATE TABLE IF NOT EXISTS historical_options_chains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    contract_name TEXT NOT NULL,
    expiration_date TEXT NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,  -- 'call' or 'put'
    last_price REAL,
    bid REAL,
    ask REAL,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    underlying_price REAL,
    days_to_expiration INTEGER,
    moneyness REAL,  -- strike / underlying_price
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, snapshot_date, contract_name)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_symbol_date ON historical_options_chains(symbol, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_expiration ON historical_options_chains(expiration_date);
CREATE INDEX IF NOT EXISTS idx_contract ON historical_options_chains(contract_name);
CREATE INDEX IF NOT EXISTS idx_symbol_exp ON historical_options_chains(symbol, expiration_date);

-- Metadata table to track ingestion progress
CREATE TABLE IF NOT EXISTS ingestion_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    last_ingested_date TEXT,
    total_records INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',  -- 'pending', 'in_progress', 'completed', 'failed'
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, start_date, end_date)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_symbol ON ingestion_metadata(symbol);
CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_metadata(status);
