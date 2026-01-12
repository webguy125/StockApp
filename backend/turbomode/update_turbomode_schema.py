"""
Update TurboMode DB Schema
Add drift_monitoring and model_metadata tables per architecture v1.1

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Location: C:\StockApp\backend\data\turbomode.db

These tables support the autonomous learning feedback loop.
"""

import sqlite3
import os
from datetime import datetime

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'turbomode.db')

print("=" * 80)
print("TURBOMODE DB - SCHEMA UPDATE")
print("=" * 80)
print(f"Location: {DB_PATH}")
print()

# Connect to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ============================================================================
# TABLE: DRIFT_MONITORING
# ============================================================================
print("Creating table: drift_monitoring")
cursor.execute("""
CREATE TABLE IF NOT EXISTS drift_monitoring (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drift_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    threshold REAL,
    alert_triggered INTEGER DEFAULT 0,
    symbol TEXT,
    feature_name TEXT,
    reference_period_start DATE,
    reference_period_end DATE,
    evaluation_period_start DATE,
    evaluation_period_end DATE,
    psi_score REAL,
    kl_divergence REAL,
    ks_statistic REAL,
    notes TEXT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    acknowledged INTEGER DEFAULT 0,
    acknowledged_at DATETIME,
    acknowledged_by TEXT
)
""")

# Indexes
cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_drift_type ON drift_monitoring(drift_type)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_metric_name ON drift_monitoring(metric_name)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_detected_at ON drift_monitoring(detected_at)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_alert ON drift_monitoring(alert_triggered)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_symbol ON drift_monitoring(symbol)")
print("[OK] Table created with 5 indexes")

# ============================================================================
# TABLE: MODEL_METADATA
# ============================================================================
print("\nCreating table: model_metadata")
cursor.execute("""
CREATE TABLE IF NOT EXISTS model_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    architecture TEXT,
    training_started_at DATETIME NOT NULL,
    training_completed_at DATETIME,
    training_duration_seconds INTEGER,
    training_samples_count INTEGER,
    validation_samples_count INTEGER,
    test_samples_count INTEGER,
    train_accuracy REAL,
    train_precision REAL,
    train_recall REAL,
    train_f1_score REAL,
    val_accuracy REAL,
    val_precision REAL,
    val_recall REAL,
    val_f1_score REAL,
    test_accuracy REAL,
    test_precision REAL,
    test_recall REAL,
    test_f1_score REAL,
    hyperparameters_json TEXT,
    feature_importance_json TEXT,
    confusion_matrix_json TEXT,
    shap_summary_json TEXT,
    regime_performance_json TEXT,
    sector_performance_json TEXT,
    model_file_path TEXT,
    config_version TEXT,
    promoted_to_production INTEGER DEFAULT 0,
    promoted_at DATETIME,
    retired INTEGER DEFAULT 0,
    retired_at DATETIME,
    retirement_reason TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, model_version)
)
""")

# Indexes
cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON model_metadata(model_name)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_version ON model_metadata(model_version)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON model_metadata(model_type)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_promoted ON model_metadata(promoted_to_production)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_training_completed ON model_metadata(training_completed_at)")
print("[OK] Table created with 5 indexes")

# ============================================================================
# TABLE: TRAINING_RUNS (Track Full Training Sessions)
# ============================================================================
print("\nCreating table: training_runs")
cursor.execute("""
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    orchestrator_version TEXT,
    config_version TEXT NOT NULL,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    status TEXT DEFAULT 'running',
    total_duration_seconds INTEGER,
    samples_generated INTEGER,
    models_trained INTEGER,
    models_promoted INTEGER,
    drift_alerts_count INTEGER DEFAULT 0,
    error_message TEXT,
    logs_file_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_run_id ON training_runs(run_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_started_at ON training_runs(started_at)")
print("[OK] Table created with 3 indexes")

# ============================================================================
# TABLE: CONFIG_AUDIT_LOG (Track Config Changes)
# ============================================================================
print("\nCreating table: config_audit_log")
cursor.execute("""
CREATE TABLE IF NOT EXISTS config_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_version TEXT NOT NULL,
    previous_version TEXT,
    change_type TEXT NOT NULL,
    changed_fields_json TEXT,
    change_description TEXT,
    changed_by TEXT,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_config_audit_version ON config_audit_log(config_version)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_config_audit_applied_at ON config_audit_log(applied_at)")
print("[OK] Table created with 2 indexes")

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

print(f"\nTotal Tables: {len(tables)}")
for table in tables:
    table_name = table[0]
    if table_name != 'sqlite_sequence':
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        status = "NEW" if table_name in ['drift_monitoring', 'model_metadata', 'training_runs', 'config_audit_log'] else ""
        print(f"  {table_name:<30} {len(columns):>2} columns, {count:>5} rows  {status}")

# Get database size
db_size = os.path.getsize(DB_PATH) / 1024  # KB
print(f"\nDatabase Size: {db_size:.2f} KB")

conn.close()

print("\n" + "=" * 80)
print("TURBOMODE DB - SCHEMA UPDATE COMPLETE")
print("=" * 80)
print("New Tables Added:")
print("  - drift_monitoring: Track data/model drift")
print("  - model_metadata: Store training metrics and metadata")
print("  - training_runs: Track full orchestrator runs")
print("  - config_audit_log: Track config changes")
print("=" * 80)
