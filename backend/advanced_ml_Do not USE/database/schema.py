"""
Advanced Database Schema for ML System v2
Completely separate from trading_system.db

Database: advanced_ml_system.db
Features: 300 core features (expandable to 600+)
Models: 6 specialized models + meta-learner
"""

import sqlite3
import os
from typing import Dict
from datetime import datetime


class AdvancedMLDatabase:
    """
    Enterprise-grade database for advanced ML trading
    Supports 300+ features per symbol and 6 model ensemble
    """

    def __init__(self, db_path: str = "backend/data/advanced_ml_system.db"):
        self.db_path = db_path
        self._ensure_db_exists()
        self.init_schema()

    def _ensure_db_exists(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def init_schema(self):
        """Initialize all database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # =====================================================
        # TABLE 1: RAW PRICE DATA (Multi-timeframe OHLCV)
        # =====================================================
        cursor.execute('''
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
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_price_symbol_time
            ON price_data(symbol, timestamp, timeframe)
        ''')

        # =====================================================
        # TABLE 2: FEATURE STORE (300+ features per symbol)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Stored as JSON for flexibility (300+ features)
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
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_features_symbol_time
            ON feature_store(symbol, timestamp)
        ''')

        # =====================================================
        # TABLE 3: MODEL PREDICTIONS (6 Models + Ensemble)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Random Forest
                rf_buy_prob REAL,
                rf_hold_prob REAL,
                rf_sell_prob REAL,
                rf_prediction TEXT,
                rf_confidence REAL,

                -- XGBoost
                xgb_buy_prob REAL,
                xgb_hold_prob REAL,
                xgb_sell_prob REAL,
                xgb_prediction TEXT,
                xgb_confidence REAL,

                -- LSTM (Deep Learning)
                lstm_buy_prob REAL,
                lstm_hold_prob REAL,
                lstm_sell_prob REAL,
                lstm_prediction TEXT,
                lstm_confidence REAL,

                -- Transformer (Deep Learning)
                transformer_buy_prob REAL,
                transformer_hold_prob REAL,
                transformer_sell_prob REAL,
                transformer_prediction TEXT,
                transformer_confidence REAL,

                -- CNN (Deep Learning)
                cnn_buy_prob REAL,
                cnn_hold_prob REAL,
                cnn_sell_prob REAL,
                cnn_prediction TEXT,
                cnn_confidence REAL,

                -- Autoencoder (Anomaly Detection)
                anomaly_score REAL,
                is_anomaly BOOLEAN,

                -- Meta-Learner (Ensemble)
                ensemble_buy_prob REAL,
                ensemble_hold_prob REAL,
                ensemble_sell_prob REAL,
                ensemble_prediction TEXT,
                ensemble_confidence REAL,

                -- Final Signal
                final_score REAL,
                final_direction TEXT,
                final_confidence REAL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time
            ON model_predictions(symbol, timestamp)
        ''')

        # =====================================================
        # TABLE 4: TRADES (Enhanced with all model data)
        # =====================================================
        cursor.execute('''
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

                -- Entry Predictions (all 6 models)
                entry_rf_prediction TEXT,
                entry_xgb_prediction TEXT,
                entry_lstm_prediction TEXT,
                entry_transformer_prediction TEXT,
                entry_cnn_prediction TEXT,
                entry_ensemble_prediction TEXT,
                entry_ensemble_confidence REAL,

                -- Features at entry (JSON)
                entry_features_json TEXT,

                -- Trade metadata
                trade_type TEXT DEFAULT 'backtest',
                strategy TEXT,
                notes TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome)
        ''')

        # =====================================================
        # TABLE 5: MODEL PERFORMANCE (Track each model)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                evaluation_date TEXT NOT NULL,

                -- Metrics
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                precision REAL DEFAULT 0.0,
                recall REAL DEFAULT 0.0,
                f1_score REAL DEFAULT 0.0,

                -- Trading metrics
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                avg_win_pct REAL DEFAULT 0.0,
                avg_loss_pct REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                sharpe_ratio REAL DEFAULT 0.0,

                -- Model info
                training_samples INTEGER DEFAULT 0,
                last_retrain_date TEXT,
                model_version TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # =====================================================
        # TABLE 6: BACKTEST RESULTS (Historical validation)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id TEXT PRIMARY KEY,
                backtest_name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,

                -- Summary
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                total_return_pct REAL DEFAULT 0.0,
                sharpe_ratio REAL DEFAULT 0.0,
                max_drawdown_pct REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,

                -- Detailed results (JSON)
                config_json TEXT,
                results_json TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # =====================================================
        # TABLE 7: DRIFT MONITORING (Phase 2 - Module 6)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                drift_type TEXT NOT NULL,
                ks_statistic REAL,
                drift_detected INTEGER,
                details_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_drift_type_time
            ON drift_monitoring(drift_type, timestamp)
        ''')

        # =====================================================
        # TABLE 8: ERROR REPLAY BUFFER (Phase 2 - Module 8)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_replay_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                regime TEXT,
                features_json TEXT NOT NULL,
                true_label INTEGER NOT NULL,
                predicted_label INTEGER NOT NULL,
                confidence REAL NOT NULL,
                error_score REAL NOT NULL,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                replayed_count INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_error_score
            ON error_replay_buffer(error_score DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_regime
            ON error_replay_buffer(regime)
        ''')

        # =====================================================
        # TABLE 9: SECTOR PERFORMANCE (Phase 2 - Module 9)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sector_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                sector TEXT NOT NULL,
                regime TEXT NOT NULL,
                predicted_label INTEGER NOT NULL,
                actual_label INTEGER,
                confidence REAL NOT NULL,
                correct INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sector_performance
            ON sector_performance(sector, regime)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_performance
            ON sector_performance(symbol, regime)
        ''')

        # =====================================================
        # TABLE 10: MODEL PROMOTION HISTORY (Phase 2 - Module 10)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_promotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_path TEXT NOT NULL,
                validation_timestamp TEXT NOT NULL,
                approved INTEGER NOT NULL,
                overall_accuracy REAL,
                crash_accuracy REAL,
                min_sector_accuracy REAL,
                drift_score REAL,
                replay_improvement REAL,
                checks_passed TEXT,
                checks_failed TEXT,
                promoted_to_production INTEGER DEFAULT 0,
                notes TEXT
            )
        ''')

        # =====================================================
        # TABLE 11: DYNAMIC EVENTS (Phase 3 - Module 7)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dynamic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL UNIQUE,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                vix_peak REAL,
                drift_score REAL,
                regime_distribution TEXT,
                sample_count INTEGER DEFAULT 0,
                captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
                added_to_archive INTEGER DEFAULT 0,
                triggered_retraining INTEGER DEFAULT 0
            )
        ''')

        # =====================================================
        # TABLE 12: SHAP ANALYSIS (Phase 3 - Module 11)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shap_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                regime TEXT,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                rank INTEGER,
                analysis_date TEXT DEFAULT CURRENT_TIMESTAMP,
                sample_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_shap_model_regime
            ON shap_analysis(model_version, regime)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_shap_feature
            ON shap_analysis(feature_name)
        ''')

        # =====================================================
        # TABLE 13: TRAINING RUNS (Phase 3 - Module 12)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT DEFAULT 'running',
                trigger TEXT NOT NULL,
                samples_trained INTEGER DEFAULT 0,
                archive_samples INTEGER DEFAULT 0,
                overall_accuracy REAL,
                crash_accuracy REAL,
                promoted INTEGER DEFAULT 0,
                error_message TEXT,
                metrics_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_training_status
            ON training_runs(status, started_at)
        ''')

        # =====================================================
        # TABLE 14: EVENTS (Event Intelligence Module)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source_type TEXT NOT NULL,

                -- Event classification
                event_type TEXT,
                event_subtype TEXT,
                event_severity REAL,
                confidence REAL,

                -- Impact scores
                impact_dividend REAL,
                impact_liquidity REAL,
                impact_credit REAL,
                impact_growth REAL,

                -- Sentiment and temporal
                sentiment_score REAL,
                temporal_relevance REAL,
                is_rare_event INTEGER DEFAULT 0,

                -- Raw content (SEC or news)
                headline TEXT,
                raw_text TEXT,
                metadata_json TEXT,

                -- SEC specific
                filing_type TEXT,
                accession_number TEXT,

                -- News specific
                urgency_level TEXT,
                entity_relevance_score REAL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, source_type, event_type)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_symbol_time
            ON events(symbol, timestamp)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_type_severity
            ON events(event_type, event_severity)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_rare
            ON events(is_rare_event, event_severity)
        ''')

        # =====================================================
        # TABLE 15: EVENT FEATURES (Encoded event features per symbol)
        # =====================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Event count features
                event_count_refinancing_30d INTEGER DEFAULT 0,
                event_count_dividend_30d INTEGER DEFAULT 0,
                event_count_litigation_30d INTEGER DEFAULT 0,
                event_count_negative_news_7d INTEGER DEFAULT 0,

                -- Severity features
                max_event_severity_30d REAL DEFAULT 0.0,
                time_since_last_high_severity_event REAL DEFAULT 999.0,

                -- Impact features
                sum_impact_dividend_90d REAL DEFAULT 0.0,
                sum_impact_liquidity_90d REAL DEFAULT 0.0,
                sum_impact_credit_90d REAL DEFAULT 0.0,

                -- Sentiment features
                news_sentiment_mean_7d REAL DEFAULT 0.0,
                news_sentiment_min_7d REAL DEFAULT 0.0,

                -- Temporal features
                event_intensity_acceleration_ratio REAL DEFAULT 0.0,
                cross_source_confirmation_flag REAL DEFAULT 0.0,

                -- Complexity features
                information_asymmetry_proxy_score REAL DEFAULT 0.0,
                filing_complexity_index REAL DEFAULT 0.0,

                -- All features as JSON for flexibility
                features_json TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_event_features_symbol_time
            ON event_features(symbol, timestamp)
        ''')

        conn.commit()
        conn.close()

        print("[OK] Advanced ML database schema initialized")
        print(f"    Database: {self.db_path}")
        print(f"    Tables: 15 (price_data, feature_store, model_predictions, trades, model_performance, backtest_results, drift_monitoring, error_replay_buffer, sector_performance, model_promotion_history, dynamic_events, shap_analysis, training_runs, events, event_features)")
        print(f"    COMPLETELY SEPARATE from trading_system.db")

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()

        stats = {}
        tables = ['price_data', 'feature_store', 'model_predictions', 'trades', 'model_performance', 'backtest_results', 'drift_monitoring', 'error_replay_buffer', 'sector_performance', 'model_promotion_history', 'dynamic_events', 'shap_analysis', 'training_runs', 'events', 'event_features']

        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            stats[table] = cursor.fetchone()[0]

        conn.close()
        return stats


if __name__ == '__main__':
    # Test database creation
    db = AdvancedMLDatabase()
    stats = db.get_stats()

    print("\nDatabase Statistics:")
    for table, count in stats.items():
        print(f"  {table}: {count} records")
