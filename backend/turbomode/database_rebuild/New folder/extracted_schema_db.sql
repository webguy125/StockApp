-- Extracted Schema

CREATE TABLE active_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
                confidence REAL NOT NULL,   -- Model confidence (0.0 - 1.0)

                -- Entry data (FIXED - never changes unless signal flips)
                entry_date TEXT NOT NULL,   -- ISO format: YYYY-MM-DD
                entry_price REAL NOT NULL,
                entry_min REAL,
                entry_max REAL,
                signal_timestamp TEXT NOT NULL,  -- When signal was created

                -- Current data (UPDATED each scan)
                current_price REAL NOT NULL,

                -- Directional SL/TP (BUY/SELL) - NULL for HOLD
                target_price REAL,
                stop_price REAL,

                -- Neutral SL/SL (HOLD) - NULL for BUY/SELL
                stop_upper REAL,
                stop_lower REAL,

                -- Adaptive SL/TP fields - NULL for HOLD
                atr REAL,
                sector_volatility_multiplier REAL,
                confidence_modifier REAL,
                stop_pct REAL,
                target_pct REAL,

                -- Classifications
                market_cap TEXT NOT NULL,    -- 'large_cap', 'mid_cap', 'small_cap'
                sector TEXT NOT NULL,        -- GICS sector name

                -- Probabilities
                prob_buy REAL,
                prob_sell REAL,
                prob_hold REAL,

                -- News risk
                news_risk_symbol TEXT,
                news_risk_sector TEXT,
                news_risk_global TEXT,

                -- Threshold source
                threshold_source TEXT,

                -- Lifecycle (UPDATED each scan)
                age_days INTEGER DEFAULT 0,  -- Days since signal_timestamp
                status TEXT DEFAULT 'ACTIVE', -- 'ACTIVE', 'TARGET_HIT', 'STOP_HIT', 'EXPIRED'

                -- Metadata
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                UNIQUE(symbol)  -- Only one signal per symbol (allows flipping BUY<->SELL)
            );

CREATE TABLE feature_store (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    features_json TEXT NOT NULL,
    rsi_14 REAL,
    macd_histogram REAL,
    volume_ratio REAL,
    trend_strength REAL,
    momentum_score REAL,
    volatility_score REAL,
    feature_version TEXT DEFAULT 'v1',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

CREATE TABLE price_data (
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
);

CREATE TABLE sector_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                sector TEXT NOT NULL,

                -- Signal counts
                total_buy_signals INTEGER DEFAULT 0,
                total_sell_signals INTEGER DEFAULT 0,

                -- Confidence metrics
                avg_buy_confidence REAL DEFAULT 0.0,
                avg_sell_confidence REAL DEFAULT 0.0,

                -- Performance (from recent history)
                win_rate_30d REAL DEFAULT 0.0,  -- Last 30 days
                avg_profit_30d REAL DEFAULT 0.0,

                -- Sentiment
                sentiment TEXT DEFAULT 'NEUTRAL',  -- 'BULLISH', 'BEARISH', 'NEUTRAL'

                created_at TEXT NOT NULL,

                UNIQUE(date, sector)  -- One record per sector per day
            );

CREATE TABLE signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,

                -- Entry data
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,

                -- Exit data
                exit_date TEXT NOT NULL,
                exit_price REAL NOT NULL,
                exit_reason TEXT NOT NULL,  -- 'TARGET_HIT', 'STOP_HIT', 'EXPIRED'

                -- Performance
                profit_loss_pct REAL NOT NULL,  -- Actual P&L percentage
                hold_days INTEGER NOT NULL,

                -- Classifications
                market_cap TEXT NOT NULL,
                sector TEXT NOT NULL,

                -- Metadata
                created_at TEXT NOT NULL,

                -- Quality metrics (added from DB schema merge)
                prob_buy REAL,
                prob_sell REAL,
                prob_hold REAL,
                entry_atr REAL,
                target_price REAL,
                stop_price REAL,
                rr REAL,
                directional_margin REAL,
                with_trend INTEGER
            );

CREATE TABLE sqlite_sequence(name,seq);

CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_date TEXT,
    exit_price REAL,
    position_size REAL DEFAULT 1.0,
    outcome TEXT DEFAULT 'open',
    profit_loss REAL,
    profit_loss_pct REAL,
    exit_reason TEXT,
    entry_features_json TEXT,
    trade_type TEXT DEFAULT 'backtest',
    strategy TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE training_runs (
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
);

CREATE INDEX idx_active_signals_age ON active_signals(age_days);

CREATE INDEX idx_active_signals_market_cap ON active_signals(market_cap);

CREATE INDEX idx_active_signals_sector ON active_signals(sector);

CREATE INDEX idx_active_signals_signal_type ON active_signals(signal_type);

CREATE INDEX idx_active_signals_symbol ON active_signals(symbol);

CREATE INDEX idx_history_exit_date ON signal_history(exit_date);

CREATE INDEX idx_history_sector ON signal_history(sector);

CREATE INDEX idx_history_symbol ON signal_history(symbol);

CREATE INDEX idx_sector_stats_date ON sector_stats(date);

CREATE INDEX idx_sector_stats_sector ON sector_stats(sector);

