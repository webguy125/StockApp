"""
14-Day Model Registry Integrity Validator

Validates the complete model registry for:
- Structural correctness (11 sectors × 6 models = 66 total)
- File existence on disk
- Legacy code contamination
- Meta-learner integrity
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, 'C:/StockApp')

from backend.turbomode.core_engine.model_registry import (
    SECTORS,
    BASE_MODELS,
    META_LEARNER,
    ALL_MODELS,
    get_sector_model_paths,
    MODELS_BASE_DIR
)

EXPECTED_MODELS = [
    'lightgbm_gpu',
    'catboost_gpu',
    'xgb_hist_gpu',
    'xgb_linear',
    'random_forest',
    'meta_learner'
]

CONTAMINATION_PATTERNS = [
    'backtest_generator',
    'label_1d_5pct',
    'label_2d',
    'label_1d',
    'horizon=',
    '_5pct',
    '_10pct',
    'trained_5pct',
    'trained_10pct',
    'HistoricalBacktest',
    'backend.advanced_ml'
]

print('=' * 80)
print('14-DAY MODEL REGISTRY INTEGRITY VALIDATION')
print('=' * 80)
print()

# ==================================================================================
# CHECK 1: VERIFY REGISTRY STRUCTURE
# ==================================================================================
print('[CHECK 1] REGISTRY STRUCTURE')
print('-' * 80)

check1_pass = True

# 1.1: Verify sector count
if len(SECTORS) == 11:
    print(f'[PASS] Sector count: {len(SECTORS)} (expected: 11)')
else:
    print(f'[FAIL] Sector count: {len(SECTORS)} (expected: 11)')
    check1_pass = False

# 1.2: Verify model list
if set(ALL_MODELS) == set(EXPECTED_MODELS):
    print(f'[PASS] Model list matches expected 6 models')
else:
    print(f'[FAIL] Model list mismatch')
    print(f'  Expected: {EXPECTED_MODELS}')
    print(f'  Actual: {ALL_MODELS}')
    check1_pass = False

# 1.3: Verify total model count
total_expected = 11 * 6
if len(SECTORS) * len(ALL_MODELS) == total_expected:
    print(f'[PASS] Total models: {total_expected}')
else:
    print(f'[FAIL] Total models: {len(SECTORS) * len(ALL_MODELS)} (expected: {total_expected})')
    check1_pass = False

print()

# ==================================================================================
# CHECK 2: VERIFY MODEL PATHS EXIST ON DISK
# ==================================================================================
print('[CHECK 2] MODEL FILE EXISTENCE')
print('-' * 80)

check2_pass = True
missing_models = []
existing_models = []

for sector in SECTORS:
    paths = get_sector_model_paths(sector)
    for model_name, model_path in paths.items():
        if os.path.exists(model_path):
            existing_models.append(f'{sector}/{model_name}')
        else:
            missing_models.append(f'{sector}/{model_name} -> {model_path}')
            check2_pass = False

print(f'Existing models: {len(existing_models)}/66')
print(f'Missing models: {len(missing_models)}/66')

if missing_models:
    print()
    print('MISSING MODEL FILES:')
    for missing in missing_models[:10]:  # Show first 10
        print(f'  [FAIL] {missing}')
    if len(missing_models) > 10:
        print(f'  ... and {len(missing_models) - 10} more')
else:
    print('[PASS] All 66 model files exist on disk')

print()

# ==================================================================================
# CHECK 3: VERIFY NO LEGACY/DEPRECATED DIRECTORIES
# ==================================================================================
print('[CHECK 3] VERIFY NO LEGACY DIRECTORIES')
print('-' * 80)

check3_pass = True
legacy_dirs = []

# Check for contaminated directories in models/
models_parent = Path(MODELS_BASE_DIR).parent
for pattern in ['trained_5pct', 'trained_10pct', '*_1d_*', '*_2d_*']:
    matches = list(models_parent.glob(pattern))
    if matches:
        legacy_dirs.extend([str(m) for m in matches])
        check3_pass = False

if legacy_dirs:
    print(f'[FAIL] Found {len(legacy_dirs)} legacy directories:')
    for legacy_dir in legacy_dirs:
        print(f'  {legacy_dir}')
else:
    print('[PASS] No legacy directories detected')

print()

# ==================================================================================
# CHECK 4: VERIFY META-LEARNER FOR EVERY SECTOR
# ==================================================================================
print('[CHECK 4] META-LEARNER INTEGRITY')
print('-' * 80)

check4_pass = True
missing_meta = []

for sector in SECTORS:
    meta_path = os.path.join(MODELS_BASE_DIR, sector, 'meta_learner.pkl')
    if not os.path.exists(meta_path):
        missing_meta.append(sector)
        check4_pass = False

if missing_meta:
    print(f'[FAIL] Missing meta-learners for {len(missing_meta)} sectors:')
    for sector in missing_meta:
        print(f'  {sector}')
else:
    print('[PASS] All 11 sectors have meta-learner models')

print()

# ==================================================================================
# CHECK 5: SCAN FOR CONTAMINATION IN REGISTRY FILE
# ==================================================================================
print('[CHECK 5] CONTAMINATION SCAN (model_registry.py)')
print('-' * 80)

check5_pass = True
contamination_found = []

registry_file = 'C:/StockApp/backend/turbomode/core_engine/model_registry.py'
with open(registry_file, 'r') as f:
    content = f.read()

for pattern in CONTAMINATION_PATTERNS:
    if pattern in content:
        contamination_found.append(pattern)
        check5_pass = False

if contamination_found:
    print(f'[FAIL] Contamination detected:')
    for pattern in contamination_found:
        print(f'  {pattern}')
else:
    print('[PASS] No contamination detected in model_registry.py')

print()

# ==================================================================================
# FINAL SUMMARY
# ==================================================================================
print('=' * 80)
print('VALIDATION SUMMARY')
print('=' * 80)

all_checks = [
    ('Registry Structure', check1_pass),
    ('Model File Existence', check2_pass),
    ('No Legacy Directories', check3_pass),
    ('Meta-Learner Integrity', check4_pass),
    ('Contamination Scan', check5_pass)
]

for check_name, passed in all_checks:
    status = '[PASS]' if passed else '[FAIL]'
    print(f'{status} {check_name}')

print()

overall_pass = all(passed for _, passed in all_checks)

if overall_pass:
    print('[SUCCESS] Model registry validation PASSED')
    print('All 66 models are correctly configured for 14-day swing trading')
else:
    print('[FAILED] Model registry validation FAILED')
    print('Issues detected - review failures above')

print()
print('=' * 80)

sys.exit(0 if overall_pass else 1)
