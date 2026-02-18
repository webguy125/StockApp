"""
Model Registry - Per-Sector Fast Ensemble Architecture

Defines the canonical paths for all 6 models per sector:
- 5 fast base models (3 GPU, 2 CPU)
- 1 per-sector MetaLearner

CLASS SEMANTICS:
- Index 0: BUY (go long, open bullish position)
- Index 1: SELL (go short, open bearish position)
- Index 2: HOLD (do nothing, no new position)
"""

import os
from typing import Dict, List

# Base directory for all trained models
MODELS_BASE_DIR = r"C:\StockApp\backend\turbomode\models\trained"

# All sectors (must match training symbols)
SECTORS = [
    'technology',
    'healthcare',
    'financials',
    'industrials',
    'consumer_discretionary',
    'consumer_staples',
    'energy',
    'materials',
    'utilities',
    'real_estate',
    'communication_services'
]

# Model names (5 base models + 1 meta-learner)
BASE_MODELS = [
    'lightgbm_gpu',
    'catboost_gpu',
    'xgb_hist_gpu',
    'xgb_linear',
    'random_forest'
]

META_LEARNER = 'meta_learner'

ALL_MODELS = BASE_MODELS + [META_LEARNER]


def get_model_path(sector: str, model_name: str) -> str:
    """
    Get absolute path to a model file.

    Args:
        sector: Sector name (e.g., 'technology')
        model_name: Model name (e.g., 'lightgbm_gpu', 'meta_learner')

    Returns:
        Absolute path to model pickle file
    """
    return os.path.join(MODELS_BASE_DIR, sector, f"{model_name}.pkl")


def get_sector_model_paths(sector: str) -> Dict[str, str]:
    """
    Get all model paths for a sector.

    Args:
        sector: Sector name

    Returns:
        Dictionary mapping model names to absolute paths
    """
    return {
        model_name: get_model_path(sector, model_name)
        for model_name in ALL_MODELS
    }


def verify_sector_models_exist(sector: str) -> Dict[str, bool]:
    """
    Check which models exist for a sector.

    Args:
        sector: Sector name

    Returns:
        Dictionary mapping model names to existence status
    """
    paths = get_sector_model_paths(sector)
    return {
        model_name: os.path.exists(path)
        for model_name, path in paths.items()
    }


def get_all_sector_model_paths() -> Dict[str, Dict[str, str]]:
    """
    Get model paths for all sectors.

    Returns:
        Nested dictionary: {sector: {model_name: path}}
    """
    return {
        sector: get_sector_model_paths(sector)
        for sector in SECTORS
    }


# Sector model registry (nested dict: sector -> model_name -> path)
SECTOR_MODEL_REGISTRY = get_all_sector_model_paths()


if __name__ == '__main__':
    # Print registry for verification
    print("SECTOR MODEL REGISTRY")
    print("=" * 80)
    for sector in SECTORS:
        print(f"\n{sector.upper()}:")
        paths = get_sector_model_paths(sector)
        for model_name, path in paths.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {model_name}: {path}")
