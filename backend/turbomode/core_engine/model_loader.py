
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Model Loader - Per-Sector Fast Ensemble Architecture

Loads all 6 models for a given sector:
- 5 fast base models (3 GPU, 2 CPU)
- 1 MetaLearner (LogisticRegression)

CLASS SEMANTICS:
- Index 0: BUY (go long, open bullish position)
- Index 1: SELL (go short, open bearish position)
- Index 2: HOLD (do nothing, no new position)
"""

import pickle
import os
from typing import Dict
from functools import lru_cache
import logging

# Import model registry
from backend.turbomode.core_engine.model_registry import (
    SECTORS,
    BASE_MODELS,
    META_LEARNER,
    ALL_MODELS,
    get_sector_model_paths,
    verify_sector_models_exist
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=11)  # Cache all 11 sectors
def load_sector_models(sector: str) -> Dict[str, object]:
    """
    Load all 6 models for a sector.

    ARCHITECTURE: Per-sector fast ensemble (5 base models + MetaLearner)

    Args:
        sector: Sector name (e.g., 'technology', 'financials')

    Returns:
        Dictionary mapping model names to loaded model objects:
        {
            'lightgbm_gpu': <LGBMClassifier>,
            'catboost_gpu': <CatBoostClassifier>,
            'xgb_hist_gpu': <XGBClassifier>,
            'xgb_linear': <XGBClassifier>,
            'random_forest': <RandomForestClassifier>,
            'meta_learner': <LogisticRegression>
        }

    Raises:
        ValueError: If sector is invalid
        FileNotFoundError: If any required model file is missing
    """
    # Validate sector
    if sector not in SECTORS:
        raise ValueError(
            f"Invalid sector: {sector}\n"
            f"Valid sectors: {SECTORS}"
        )

    # Check if all models exist
    model_exists = verify_sector_models_exist(sector)
    missing_models = [name for name, exists in model_exists.items() if not exists]

    if missing_models:
        raise FileNotFoundError(
            f"Missing models for sector '{sector}': {missing_models}\n"
            f"Run train_sector_models.py to train all models first."
        )

    # Load all models
    logger.info(f"[LOADER] Loading ensemble models for {sector}...")
    model_paths = get_sector_model_paths(sector)
    loaded_models = {}

    for model_name in ALL_MODELS:
        model_path = model_paths[model_name]

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            loaded_models[model_name] = model
            logger.info(f"  ✓ Loaded {model_name}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_name} from {model_path}: {e}"
            )

    logger.info(f"[SUCCESS] All {len(loaded_models)} models loaded for {sector}")
    return loaded_models


def load_base_models(sector: str) -> Dict[str, object]:
    """
    Load only the 5 base models for a sector (exclude MetaLearner).

    Args:
        sector: Sector name

    Returns:
        Dictionary with 5 base model instances
    """
    all_models = load_sector_models(sector)
    return {
        name: model
        for name, model in all_models.items()
        if name in BASE_MODELS
    }


def load_meta_learner(sector: str) -> object:
    """
    Load only the MetaLearner for a sector.

    Args:
        sector: Sector name

    Returns:
        MetaLearner model instance (LogisticRegression)
    """
    all_models = load_sector_models(sector)
    return all_models[META_LEARNER]


def verify_all_sectors() -> Dict[str, Dict[str, bool]]:
    """
    Verify which models exist for all sectors.

    Returns:
        Nested dictionary: {sector: {model_name: exists}}
    """
    results = {}
    for sector in SECTORS:
        results[sector] = verify_sector_models_exist(sector)
    return results


def print_model_status():
    """
    Print status of all sector models.
    """
    print("\n" + "=" * 80)
    print("SECTOR MODEL STATUS")
    print("=" * 80)

    for sector in SECTORS:
        print(f"\n{sector.upper()}:")
        model_status = verify_sector_models_exist(sector)

        all_exist = all(model_status.values())
        status_symbol = "✓" if all_exist else "✗"

        print(f"  [{status_symbol}] Overall: {sum(model_status.values())}/{len(model_status)} models")

        for model_name in ALL_MODELS:
            exists = model_status[model_name]
            symbol = "✓" if exists else "✗"
            print(f"    {symbol} {model_name}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Print status of all models
    print_model_status()

    # Test loading models for sectors that have all 6 models
    print("\nTesting model loading...")
    print("-" * 80)

    verification_results = verify_all_sectors()

    for sector in SECTORS:
        model_status = verification_results[sector]
        all_exist = all(model_status.values())

        if all_exist:
            try:
                print(f"\nLoading models for {sector}...")
                models = load_sector_models(sector)
                print(f"  ✓ Successfully loaded {len(models)} models")

                # Test that each model can predict
                import numpy as np
                X_test = np.random.randn(1, 179)  # 1 sample, 179 features

                # Test base models
                print(f"  Testing base model predictions...")
                for model_name in BASE_MODELS:
                    probs = models[model_name].predict_proba(X_test)
                    print(f"    ✓ {model_name}: shape={probs.shape}, sum={probs.sum():.3f}")

                # Test MetaLearner
                print(f"  Testing MetaLearner prediction...")
                # Stack base model predictions as features
                base_preds = []
                for model_name in BASE_MODELS:
                    probs = models[model_name].predict_proba(X_test)
                    base_preds.append(probs)
                stacked_features = np.concatenate(base_preds, axis=1)  # shape: (1, 15)

                meta_probs = models[META_LEARNER].predict_proba(stacked_features)
                print(f"    ✓ {META_LEARNER}: shape={meta_probs.shape}, sum={meta_probs.sum():.3f}")

            except Exception as e:
                print(f"  ✗ Failed to load/test {sector}: {e}")
        else:
            missing = [name for name, exists in model_status.items() if not exists]
            print(f"\n{sector}: Missing models - {missing}")

    print("\n" + "=" * 80)
