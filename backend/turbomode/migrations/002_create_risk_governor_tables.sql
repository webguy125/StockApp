-- ============================================================================
-- REAL-TIME RISK GOVERNOR DATABASE MIGRATION
-- ============================================================================
-- Migration: 002_create_risk_governor_tables.sql
-- Version: 3.0.0
-- Date: 2026-01-25
-- Description: Creates three tables for Real-Time Risk Governor system
--              Implements schema exactly as defined in real_time_risk_governor.json v3.0.0
-- ============================================================================

-- ============================================================================
-- TABLE 1: global_risk_state
-- ============================================================================
-- Purpose: Single-row table tracking current portfolio-level risk state
-- Write Rule: UPDATE-only (except first INSERT), no DELETE
-- ============================================================================

CREATE TABLE IF NOT EXISTS global_risk_state (
    -- Primary key (single row enforced by CHECK constraint)
    id INTEGER PRIMARY KEY CHECK (id = 1),

    -- State machine fields
    current_state TEXT NOT NULL CHECK (current_state IN ('NORMAL', 'CAUTION', 'CRITICAL')),
    previous_state TEXT CHECK (previous_state IN ('NORMAL', 'CAUTION', 'CRITICAL')),

    -- Action flags (boolean as INTEGER: 0 = false, 1 = true)
    freeze_new_entries INTEGER NOT NULL DEFAULT 0 CHECK (freeze_new_entries IN (0, 1)),
    tighten_exits INTEGER NOT NULL DEFAULT 0 CHECK (tighten_exits IN (0, 1)),
    force_deleveraging INTEGER NOT NULL DEFAULT 0 CHECK (force_deleveraging IN (0, 1)),

    -- Confidence penalty fields
    confidence_penalty_active INTEGER NOT NULL DEFAULT 0 CHECK (confidence_penalty_active IN (0, 1)),
    confidence_penalty_factor REAL,

    -- Diagnostic fields
    reason TEXT,
    details TEXT,  -- JSON blob for computed metrics

    -- Timestamps
    entered_at TEXT NOT NULL,  -- ISO8601 timestamp when current_state was entered
    updated_at TEXT NOT NULL   -- ISO8601 timestamp of last update
);

-- Create index on state for fast lookups
CREATE INDEX IF NOT EXISTS idx_global_risk_state_current ON global_risk_state(current_state);

-- Initialize single row with NORMAL state
INSERT OR IGNORE INTO global_risk_state (
    id,
    current_state,
    previous_state,
    freeze_new_entries,
    tighten_exits,
    force_deleveraging,
    confidence_penalty_active,
    confidence_penalty_factor,
    reason,
    details,
    entered_at,
    updated_at
) VALUES (
    1,
    'NORMAL',
    NULL,
    0,
    0,
    0,
    0,
    NULL,
    'System initialized',
    '{}',
    datetime('now'),
    datetime('now')
);

-- ============================================================================
-- TABLE 2: symbol_risk_flags
-- ============================================================================
-- Purpose: Per-symbol risk flags (volatility spikes, confidence drops, etc.)
-- Write Rule: INSERT when flag raised, UPDATE to set cleared_at when resolved
-- ============================================================================

CREATE TABLE IF NOT EXISTS symbol_risk_flags (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Symbol identifier
    symbol TEXT NOT NULL,

    -- Risk classification
    risk_level TEXT NOT NULL CHECK (risk_level IN ('NORMAL', 'CAUTION', 'CRITICAL')),

    -- Flag details
    reason TEXT NOT NULL,  -- Human-readable reason (e.g., "volatility_spike", "confidence_collapse")
    details TEXT,          -- JSON blob with metrics at time of flag

    -- Timestamps
    flagged_at TEXT NOT NULL,  -- ISO8601 timestamp when flag was raised
    cleared_at TEXT            -- ISO8601 timestamp when flag was cleared (NULL = active)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_symbol_risk_flags_symbol ON symbol_risk_flags(symbol);
CREATE INDEX IF NOT EXISTS idx_symbol_risk_flags_risk_level ON symbol_risk_flags(risk_level);
CREATE INDEX IF NOT EXISTS idx_symbol_risk_flags_flagged_at ON symbol_risk_flags(flagged_at DESC);
CREATE INDEX IF NOT EXISTS idx_symbol_risk_flags_active ON symbol_risk_flags(symbol, cleared_at) WHERE cleared_at IS NULL;

-- ============================================================================
-- TABLE 3: risk_events
-- ============================================================================
-- Purpose: Append-only audit log of all risk state changes and transitions
-- Write Rule: INSERT-only, no UPDATE or DELETE
-- ============================================================================

CREATE TABLE IF NOT EXISTS risk_events (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Event classification
    event_type TEXT NOT NULL,  -- e.g., "state_transition", "symbol_flag_raised", "confidence_penalty_triggered"

    -- Event payload (JSON blob)
    payload TEXT NOT NULL,  -- JSON with event-specific fields

    -- Timestamp
    created_at TEXT NOT NULL  -- ISO8601 timestamp
);

-- Indexes for fast event queries
CREATE INDEX IF NOT EXISTS idx_risk_events_event_type ON risk_events(event_type);
CREATE INDEX IF NOT EXISTS idx_risk_events_created_at ON risk_events(created_at DESC);

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
