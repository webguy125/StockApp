"""
Centralized Path Configuration for TurboMode
Prevents path confusion and accidental file creation in wrong locations

ALL TURBOMODE SCRIPTS SHOULD IMPORT PATHS FROM THIS MODULE
Never use relative paths like "backend/data/turbomode.db" directly
"""

import os
from pathlib import Path

# Project root directory (C:\StockApp)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Backend directory
BACKEND_DIR = PROJECT_ROOT / 'backend'

# Data directory
DATA_DIR = BACKEND_DIR / 'data'

# TurboMode directory
TURBOMODE_DIR = BACKEND_DIR / 'turbomode'

# ============================================================================
# DATABASE PATHS (Absolute paths that work from anywhere)
# ============================================================================

# Main TurboMode production database
TURBOMODE_DB = DATA_DIR / 'turbomode.db'

# ============================================================================
# MODEL PATHS
# ============================================================================

# Model storage directory
MODELS_DIR = DATA_DIR / 'turbomode_models'

# Individual model directories
META_LEARNER_DIR = MODELS_DIR / 'meta_learner_v2'
XGBOOST_DIR = MODELS_DIR / 'xgboost'
LIGHTGBM_DIR = MODELS_DIR / 'lightgbm'
CATBOOST_DIR = MODELS_DIR / 'catboost'

# ============================================================================
# DATA FILE PATHS
# ============================================================================

# JSON data files
STOCK_RANKINGS_JSON = DATA_DIR / 'stock_rankings.json'
RANKING_HISTORY_JSON = DATA_DIR / 'ranking_history.json'
TURBOMODE_SCHEDULER_STATE = DATA_DIR / 'turbomode_scheduler_state.json'
ML_AUTOMATION_STATE = DATA_DIR / 'ml_automation_state.json'
AUTOMATED_LEARNER_STATE = DATA_DIR / 'automated_learner_state.json'

# TurboMode-specific data directory
TURBOMODE_DATA_DIR = TURBOMODE_DIR / 'data'
ALL_PREDICTIONS_JSON = TURBOMODE_DATA_DIR / 'all_predictions.json'

# ============================================================================
# LOG AND CONFIG PATHS
# ============================================================================

# Config directory
CONFIG_DIR = TURBOMODE_DIR / 'config'

# Session files
SESSION_DIR = PROJECT_ROOT / 'session_files'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_db_path() -> str:
    """
    Get absolute path to TurboMode production database

    Returns:
        str: Absolute path to turbomode.db

    Example:
        >>> from backend.turbomode.paths import get_db_path
        >>> db_path = get_db_path()
        >>> # Always returns: C:\\StockApp\\backend\\data\\turbomode.db
    """
    return str(TURBOMODE_DB)


def get_model_dir(model_name: str) -> Path:
    """
    Get absolute path to model directory

    Args:
        model_name: Name of model (e.g., 'xgboost', 'lightgbm')

    Returns:
        Path: Absolute path to model directory
    """
    return MODELS_DIR / model_name


def ensure_data_dirs():
    """
    Ensure all data directories exist
    Called on module import to prevent missing directory errors
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TURBOMODE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def validate_paths():
    """
    Validate that all critical paths are correctly configured
    Raises error if any path points outside project
    """
    errors = []

    # Ensure all paths are under PROJECT_ROOT
    paths_to_check = {
        'TURBOMODE_DB': TURBOMODE_DB,
        'MODELS_DIR': MODELS_DIR,
        'DATA_DIR': DATA_DIR,
        'TURBOMODE_DIR': TURBOMODE_DIR
    }

    for name, path in paths_to_check.items():
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"{name} is outside project root: {path}")

    if errors:
        raise ValueError("Path configuration errors:\n" + "\n".join(errors))


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Ensure directories exist when module is imported
ensure_data_dirs()

# Validate paths
validate_paths()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("TURBOMODE PATH CONFIGURATION")
    print("=" * 80)
    print()

    print("Project Structure:")
    print(f"  PROJECT_ROOT:     {PROJECT_ROOT}")
    print(f"  BACKEND_DIR:      {BACKEND_DIR}")
    print(f"  DATA_DIR:         {DATA_DIR}")
    print(f"  TURBOMODE_DIR:    {TURBOMODE_DIR}")
    print()

    print("Database:")
    print(f"  TURBOMODE_DB:     {TURBOMODE_DB}")
    print(f"  Exists:           {TURBOMODE_DB.exists()}")
    print()

    print("Models:")
    print(f"  MODELS_DIR:       {MODELS_DIR}")
    print(f"  META_LEARNER_DIR: {META_LEARNER_DIR}")
    print()

    print("Data Files:")
    print(f"  ALL_PREDICTIONS:  {ALL_PREDICTIONS_JSON}")
    print(f"  STOCK_RANKINGS:   {STOCK_RANKINGS_JSON}")
    print()

    print("Helper Functions:")
    print(f"  get_db_path():    {get_db_path()}")
    print()

    print("=" * 80)
    print("✅ All paths validated successfully")
    print("=" * 80)
    print()
    print("USAGE IN YOUR SCRIPTS:")
    print()
    print("  # OLD WAY (DON'T DO THIS):")
    print('  db_path = "backend/data/turbomode.db"  # ❌ Relative path')
    print()
    print("  # NEW WAY (DO THIS):")
    print("  from backend.turbomode.paths import TURBOMODE_DB")
    print("  db_path = str(TURBOMODE_DB)  # ✅ Always correct absolute path")
    print()
