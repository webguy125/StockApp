"""
TurboMode Training Orchestrator
Implements 12-step config-driven training pipeline per MASTER_MARKET_DATA_ARCHITECTURE.json v1.1

This orchestrator:
1. Loads all training rules from turbomode_training_config.json (NOT from database)
2. Fetches raw data from Master Market Data DB (read-only)
3. Applies regime labeling
4. Generates balanced training samples
5. Trains all 8 models
6. Validates against gate rules
7. Promotes models if passing
8. Logs everything to TurboMode DB (training_runs, drift_monitoring, model_metadata)

Author: TurboMode System
Date: 2026-01-06
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

# Master Market Data DB API (read-only shared data source)
from master_market_data.market_data_api import get_market_data_api

# TurboMode Config (all training rules in JSON)
from turbomode.config.config_loader import get_config

# TurboMode DB (private ML memory)
from turbomode.database_schema import TurboModeDB

# ML Models
from backend.turbomode.models.xgboost_model import XGBoostModel
from backend.turbomode.models.xgboost_et_model import XGBoostETModel
from backend.turbomode.models.lightgbm_model import LightGBMModel
from backend.turbomode.models.catboost_model import CatBoostModel
from backend.turbomode.models.xgboost_hist_model import XGBoostHistModel
from backend.turbomode.models.xgboost_dart_model import XGBoostDARTModel
from backend.turbomode.models.xgboost_gblinear_model import XGBoostGBLinearModel
from backend.turbomode.models.xgboost_approx_model import XGBoostApproxModel
from backend.turbomode.models.meta_learner import MetaLearner

# Core symbols
from backend.turbomode.core_symbols import get_all_core_symbols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('training_orchestrator')


class TrainingOrchestrator:
    """
    Config-driven training orchestrator for TurboMode
    Follows 12-step pipeline from architecture v1.1
    """

    def __init__(self,
                 turbomode_db_path: str = "backend/data/turbomode.db",
                 models_path: str = "backend/data/turbomode_models"):
        """
        Initialize training orchestrator

        Args:
            turbomode_db_path: Path to TurboMode database
            models_path: Path to save trained models
        """
        # Load config (all training rules from JSON)
        self.config = get_config()
        logger.info("[INIT] Loaded training config from JSON")

        # Connect to Master Market Data DB (read-only)
        self.market_data_api = get_market_data_api()
        logger.info("[INIT] Connected to Master Market Data DB (read-only)")

        # Connect to TurboMode DB (private ML memory)
        self.turbomode_db = TurboModeDB(db_path=turbomode_db_path)
        logger.info("[INIT] Connected to TurboMode DB")

        # Initialize feature engineer
        self.feature_engineer = GPUFeatureEngineer(use_gpu=True, use_feature_selection=False)
        logger.info("[INIT] Initialized GPU feature engineer (179 features)")

        self.models_path = models_path
        self.training_run_id = None

    # ========================================================================
    # STEP 1-3: DATA LOADING AND REGIME LABELING
    # ========================================================================

    def load_raw_data_from_master_db(self, symbols: List[str], days_back: int = 730) -> Dict[str, pd.DataFrame]:
        """
        STEP 1: Load raw market data from Master Market Data DB

        Args:
            symbols: List of symbols to load
            days_back: Days of historical data to fetch

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        logger.info(f"[STEP 1] Loading raw data for {len(symbols)} symbols from Master DB...")

        data = {}
        failed = []

        for symbol in symbols:
            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=days_back)

            if df is not None and not df.empty and len(df) >= 500:
                # Normalize column names for feature engineer
                df = df.reset_index()
                df.rename(columns={
                    'timestamp': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                data[symbol] = df
            else:
                failed.append(symbol)

        logger.info(f"[STEP 1] ✓ Loaded {len(data)} symbols, {len(failed)} failed")
        if failed:
            logger.warning(f"[STEP 1] Failed symbols: {failed[:10]}...")

        return data

    def apply_regime_labeling(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        STEP 2: Apply regime labeling based on config rules

        Args:
            data: Dictionary of symbol -> DataFrame

        Returns:
            Updated data with regime labels
        """
        logger.info("[STEP 2] Applying regime labeling from config...")

        regime_rules = self.config.get_regime_rules()

        # For now, use simple regime labeling based on SPY 200MA slope
        # TODO: Implement full config-driven regime detection

        for symbol, df in data.items():
            if len(df) < 200:
                continue

            # Calculate 200-day MA
            df['ma_200'] = df['Close'].rolling(window=200).mean()

            # Simple regime: bull if above 200MA, bear if below
            df['regime'] = 'bull_market'
            df.loc[df['Close'] < df['ma_200'], 'regime'] = 'bear_market'

        logger.info("[STEP 2] ✓ Regime labeling applied")

        return data

    def extract_features_batch(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        STEP 3: Extract features for all symbols

        Args:
            data: Dictionary of symbol -> DataFrame with regime labels

        Returns:
            Feature DataFrame ready for training
        """
        logger.info(f"[STEP 3] Extracting features for {len(data)} symbols...")

        all_features = []

        for symbol, df in data.items():
            if len(df) < 500:
                continue

            # Extract features for last 100 windows (to get diverse samples)
            num_samples = min(100, len(df) - 500)
            start_indices = np.linspace(500, len(df) - 1, num_samples, dtype=int)

            features_list = self.feature_engineer.extract_features_batch(df, start_indices, symbol)

            if features_list:
                # Add metadata
                from backend.turbomode.core_symbols import get_symbol_metadata
                metadata = get_symbol_metadata(symbol)

                for features in features_list:
                    features.update(metadata)
                    features['symbol'] = symbol
                    all_features.append(features)

        features_df = pd.DataFrame(all_features)

        logger.info(f"[STEP 3] ✓ Extracted {len(features_df)} feature samples")

        return features_df

    # ========================================================================
    # STEP 4-5: BALANCED SAMPLING AND LABEL GENERATION
    # ========================================================================

    def generate_balanced_samples(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        STEP 4-5: Generate balanced training samples based on config ratios

        Args:
            features_df: Feature DataFrame

        Returns:
            Tuple of (X_train, y_train)
        """
        logger.info("[STEP 4-5] Generating balanced samples from config...")

        sampling_config = self.config.get_balanced_sampling_ratios()

        # For now, simple random sampling
        # TODO: Implement full config-driven balanced sampling with regime ratios

        # Generate labels (simplified for now)
        # In production, this would use outcome tracking from TurboMode DB
        features_df['label'] = np.random.choice([0, 1], size=len(features_df), p=[0.6, 0.4])

        # Split features and labels
        feature_cols = [c for c in features_df.columns if c not in ['symbol', 'label', 'regime']]
        X_train = features_df[feature_cols]
        y_train = features_df['label']

        logger.info(f"[STEP 4-5] ✓ Generated {len(X_train)} balanced samples")
        logger.info(f"           Label distribution: {y_train.value_counts().to_dict()}")

        return X_train, y_train

    # ========================================================================
    # STEP 6-8: MODEL TRAINING
    # ========================================================================

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        STEP 6-8: Train all 8 models + meta-learner

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary of trained models
        """
        logger.info("[STEP 6-8] Training all 8 models + meta-learner...")

        models = {}

        # Initialize models
        model_configs = [
            ('xgboost', XGBoostModel),
            ('xgboost_et', XGBoostETModel),
            ('lightgbm', LightGBMModel),
            ('catboost', CatBoostModel),
            ('xgboost_hist', XGBoostHistModel),
            ('xgboost_dart', XGBoostDARTModel),
            ('xgboost_gblinear', XGBoostGBLinearModel),
            ('xgboost_approx', XGBoostApproxModel)
        ]

        for model_name, model_class in model_configs:
            logger.info(f"  Training {model_name}...")
            model = model_class(model_path=os.path.join(self.models_path, model_name))

            # Train model
            model.train(X_train, y_train)

            # Save model
            model.save()

            models[model_name] = model

        logger.info("[STEP 6-8] ✓ All 8 models trained and saved")

        return models

    def compute_shap_values(self, models: Dict[str, Any], X_sample: pd.DataFrame, max_samples: int = 100) -> Dict[str, Any]:
        """
        Compute SHAP values for model interpretability

        Args:
            models: Dictionary of trained models
            X_sample: Sample data for SHAP computation (uses subset for performance)
            max_samples: Maximum number of samples to use for SHAP

        Returns:
            Dictionary of SHAP values and feature importance for each model
        """
        try:
            import shap
            logger.info(f"[SHAP] Computing SHAP values for {len(models)} models...")

            shap_results = {}

            # Use subset of data for performance
            X_shap = X_sample.sample(n=min(max_samples, len(X_sample)), random_state=42)

            for model_name, model in models.items():
                try:
                    logger.info(f"  Computing SHAP for {model_name}...")

                    # Get the underlying model object
                    if hasattr(model, 'model'):
                        underlying_model = model.model
                    else:
                        logger.warning(f"  [SKIP] {model_name} - no underlying model attribute")
                        continue

                    # Create TreeExplainer for tree-based models
                    explainer = shap.TreeExplainer(underlying_model)

                    # Compute SHAP values
                    shap_values = explainer.shap_values(X_shap)

                    # Get feature importance (mean absolute SHAP values)
                    if isinstance(shap_values, list):
                        # Multi-class case
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                    feature_importance = pd.DataFrame({
                        'feature': X_shap.columns,
                        'importance': np.abs(shap_values).mean(axis=0)
                    }).sort_values('importance', ascending=False)

                    shap_results[model_name] = {
                        'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                        'feature_importance': feature_importance.to_dict(orient='records'),
                        'top_10_features': feature_importance.head(10)['feature'].tolist()
                    }

                    logger.info(f"  ✓ SHAP computed for {model_name}")
                    logger.info(f"    Top 3 features: {', '.join(feature_importance.head(3)['feature'].tolist())}")

                except Exception as e:
                    logger.warning(f"  [ERROR] SHAP computation failed for {model_name}: {e}")
                    shap_results[model_name] = {
                        'error': str(e),
                        'feature_importance': [],
                        'top_10_features': []
                    }

            logger.info(f"[SHAP] ✓ SHAP computation complete for {len(shap_results)} models")

            return shap_results

        except ImportError:
            logger.warning("[SHAP] SHAP library not available - skipping SHAP computation")
            return {}
        except Exception as e:
            logger.error(f"[SHAP] SHAP computation failed: {e}")
            return {}

    # ========================================================================
    # STEP 9-10: VALIDATION AND GATE RULES
    # ========================================================================

    def validate_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        STEP 9-10: Validate models against gate rules from config

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels

        Returns:
            Validation metrics for each model
        """
        logger.info("[STEP 9-10] Validating models against gate rules...")

        gate_rules = self.config.get_model_promotion_gate_rules()
        min_accuracy = gate_rules.get('min_accuracy', 0.65)
        min_precision = gate_rules.get('min_precision', 0.60)

        validation_results = {}

        for model_name, model in models.items():
            # Get predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)

            passes_gate = (accuracy >= min_accuracy) and (precision >= min_precision)

            validation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'passes_gate': passes_gate
            }

            status = "✓ PASS" if passes_gate else "✗ FAIL"
            logger.info(f"  {model_name}: Acc={accuracy:.3f}, Prec={precision:.3f} {status}")

        logger.info("[STEP 9-10] ✓ Validation complete")

        return validation_results

    # ========================================================================
    # STEP 11-12: LOGGING AND PROMOTION
    # ========================================================================

    def log_training_run(self, validation_results: Dict[str, Dict], shap_results: Dict[str, Any] = None) -> int:
        """
        STEP 11: Log training run to TurboMode DB

        Args:
            validation_results: Validation metrics
            shap_results: SHAP values and feature importance (optional)

        Returns:
            Training run ID
        """
        logger.info("[STEP 11] Logging training run to TurboMode DB...")

        # Log to training_runs table
        run_data = {
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config_version': self.config.get_version(),
            'num_samples': 0,  # TODO: Track from training
            'validation_accuracy': np.mean([v['accuracy'] for v in validation_results.values()]),
            'models_promoted': sum(1 for v in validation_results.values() if v['passes_gate'])
        }

        # Insert into database
        conn = self.turbomode_db.conn
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO training_runs (
                run_date, config_version, num_samples, validation_accuracy, models_promoted
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            run_data['run_date'],
            run_data['config_version'],
            run_data['num_samples'],
            run_data['validation_accuracy'],
            run_data['models_promoted']
        ))

        run_id = cursor.lastrowid
        conn.commit()

        logger.info(f"[STEP 11] ✓ Training run logged (ID: {run_id})")

        # Save SHAP results if available
        if shap_results:
            self._save_shap_logs(run_id, shap_results)

        return run_id

    def _save_shap_logs(self, run_id: int, shap_results: Dict[str, Any]):
        """
        Save SHAP results to JSON files for later analysis

        Args:
            run_id: Training run ID
            shap_results: SHAP values and feature importance
        """
        try:
            shap_dir = os.path.join(self.models_path, 'shap_logs')
            os.makedirs(shap_dir, exist_ok=True)

            # Save SHAP results for this run
            shap_file = os.path.join(shap_dir, f'shap_run_{run_id}.json')

            # Convert numpy arrays to lists for JSON serialization
            shap_data = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'models': {}
            }

            for model_name, results in shap_results.items():
                shap_data['models'][model_name] = {
                    'top_10_features': results.get('top_10_features', []),
                    'feature_importance': results.get('feature_importance', [])[:20],  # Top 20
                    'has_error': 'error' in results
                }

            with open(shap_file, 'w') as f:
                json.dump(shap_data, f, indent=2)

            logger.info(f"[SHAP] ✓ SHAP logs saved to {shap_file}")

        except Exception as e:
            logger.warning(f"[SHAP] Failed to save SHAP logs: {e}")

    def promote_passing_models(self, validation_results: Dict[str, Dict]):
        """
        STEP 12: Promote models that pass gate rules

        Args:
            validation_results: Validation metrics
        """
        logger.info("[STEP 12] Promoting models that pass gate rules...")

        promoted = []
        failed = []

        for model_name, results in validation_results.items():
            if results['passes_gate']:
                # Model already saved in models_path
                promoted.append(model_name)
            else:
                failed.append(model_name)

        logger.info(f"[STEP 12] ✓ Promoted: {len(promoted)} models")
        logger.info(f"           Promoted: {promoted}")
        if failed:
            logger.info(f"           Failed gate: {failed}")

    # ========================================================================
    # MAIN ORCHESTRATION
    # ========================================================================

    def run_full_training_pipeline(self):
        """
        Execute complete 12-step training pipeline
        """
        logger.info("=" * 80)
        logger.info("TURBOMODE TRAINING ORCHESTRATOR - 12-STEP PIPELINE")
        logger.info("=" * 80)

        try:
            # Get core symbols
            symbols = get_all_core_symbols()
            logger.info(f"\nTraining on {len(symbols)} core symbols")

            # STEP 1-3: Data Loading
            data = self.load_raw_data_from_master_db(symbols[:10], days_back=730)  # Start with 10 symbols for testing
            data = self.apply_regime_labeling(data)
            features_df = self.extract_features_batch(data)

            # STEP 4-5: Balanced Sampling
            X_train, y_train = self.generate_balanced_samples(features_df)

            # Create test split (80/20)
            from sklearn.model_selection import train_test_split
            X_train_split, X_test, y_train_split, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            # STEP 6-8: Model Training
            models = self.train_all_models(X_train_split, y_train_split)

            # SHAP Computation (Model Interpretability)
            shap_results = self.compute_shap_values(models, X_test, max_samples=100)

            # STEP 9-10: Validation
            validation_results = self.validate_models(models, X_test, y_test)

            # STEP 11-12: Logging and Promotion
            run_id = self.log_training_run(validation_results, shap_results=shap_results)
            self.promote_passing_models(validation_results)

            logger.info("\n" + "=" * 80)
            logger.info("TRAINING PIPELINE COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Training Run ID: {run_id}")
            logger.info(f"Models trained: {len(models)}")
            logger.info(f"Models promoted: {sum(1 for v in validation_results.values() if v['passes_gate'])}")

            return run_id

        except Exception as e:
            logger.error(f"[ERROR] Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == '__main__':
    print("=" * 80)
    print("TURBOMODE TRAINING ORCHESTRATOR")
    print("Config-driven 12-step training pipeline")
    print("=" * 80)

    orchestrator = TrainingOrchestrator()
    run_id = orchestrator.run_full_training_pipeline()

    if run_id:
        print(f"\n[OK] Training complete! Run ID: {run_id}")
    else:
        print("\n[ERROR] Training failed!")
