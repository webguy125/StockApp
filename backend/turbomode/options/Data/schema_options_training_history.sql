-- Options Training History Database Schema
-- Stores outcomes of historical options positions for meta-learner training

CREATE TABLE IF NOT EXISTS options_training_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    exit_date TEXT,
    expiration_date TEXT NOT NULL,

    -- Position structure
    position_type TEXT NOT NULL,  -- 'iron_condor', 'strangle', 'spread', etc.
    short_call_strike REAL,
    short_put_strike REAL,
    long_call_strike REAL,
    long_put_strike REAL,

    -- Entry metrics
    entry_underlying_price REAL NOT NULL,
    entry_max_credit REAL,
    entry_max_profit REAL,
    entry_max_loss REAL,
    entry_prob_profit REAL,
    entry_implied_volatility REAL,
    entry_vix REAL,

    -- Exit metrics
    exit_underlying_price REAL,
    exit_pnl REAL,
    exit_pnl_percent REAL,
    exit_reason TEXT,  -- 'expired', 'stop_loss', 'take_profit', 'manual'

    -- Outcome classification
    outcome TEXT,  -- 'win', 'loss', 'breakeven'
    max_profit_captured REAL,  -- Percentage of max profit captured
    days_held INTEGER,

    -- Market context at entry
    market_regime TEXT,  -- 'bullish', 'bearish', 'neutral', 'volatile'
    sector TEXT,
    market_cap_category TEXT,

    -- Feature snapshot (for meta-learner)
    features_json TEXT,  -- JSON blob of all features at entry

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, entry_date, expiration_date, short_call_strike, short_put_strike)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_outcome_symbol ON options_training_outcomes(symbol);
CREATE INDEX IF NOT EXISTS idx_outcome_date ON options_training_outcomes(entry_date);
CREATE INDEX IF NOT EXISTS idx_outcome_result ON options_training_outcomes(outcome);
CREATE INDEX IF NOT EXISTS idx_outcome_regime ON options_training_outcomes(market_regime);
CREATE INDEX IF NOT EXISTS idx_outcome_exp ON options_training_outcomes(expiration_date);

-- Summary statistics table (aggregated for quick analysis)
CREATE TABLE IF NOT EXISTS training_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    regime TEXT NOT NULL,
    total_trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    breakevens INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    avg_pnl REAL DEFAULT 0.0,
    avg_days_held REAL DEFAULT 0.0,
    total_pnl REAL DEFAULT 0.0,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, regime)
);

CREATE INDEX IF NOT EXISTS idx_stats_symbol ON training_statistics(symbol);
CREATE INDEX IF NOT EXISTS idx_stats_regime ON training_statistics(regime);
