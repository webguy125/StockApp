"""
Scheduler Alignment Verification Script

Verifies that the scheduler is calling the correct 14-day swing trading entrypoints.
"""

import os
import re
import json

print('=' * 80)
print('SCHEDULER ALIGNMENT VERIFICATION')
print('=' * 80)
print()

# ==================================================================================
# LOAD SCHEDULER CONFIG
# ==================================================================================

config_path = 'C:/StockApp/backend/scheduler_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

print('SCHEDULER TASKS FROM CONFIG:')
print('-' * 80)
for task in config['scheduled_tasks']:
    print(f"Task {task['task_id']}: {task['name']}")
    print(f"  Function: {task['function_name']}")
    print(f"  Endpoint: {task['endpoint']}")
print()

# ==================================================================================
# READ UNIFIED_SCHEDULER.PY
# ==================================================================================

scheduler_path = 'C:/StockApp/backend/unified_scheduler.py'
with open(scheduler_path, 'r') as f:
    scheduler_content = f.read()

# ==================================================================================
# CHECK TASK 4: BACKTEST DATA GENERATOR
# ==================================================================================

print('=' * 80)
print('TASK 4: BACKTEST DATA GENERATOR VERIFICATION')
print('=' * 80)

# Find the run_backtest_generator function
backtest_func_match = re.search(
    r'def run_backtest_generator\(\).*?(?=\ndef\s|\Z)',
    scheduler_content,
    re.DOTALL
)

if backtest_func_match:
    func_body = backtest_func_match.group(0)

    # Check for correct import
    if 'from backend.turbomode.core_engine.turbomode_backtest import TurboModeBacktest' in func_body:
        print('[PASS] Imports TurboModeBacktest (correct 14-day engine)')
    else:
        print('[FAIL] Does not import TurboModeBacktest')

    # Check for wrong import
    if 'from backend.turbomode.core_engine.backtest_generator import BacktestGenerator' in func_body:
        print('[FAIL] Still imports BacktestGenerator (deprecated)')
    else:
        print('[PASS] Does not import BacktestGenerator (good)')

    # Check for 14-day configuration
    if '14-day' in func_body or '14 day' in func_body:
        print('[PASS] References 14-day configuration')
    else:
        print('[WARN] No explicit 14-day reference in logs')

    # Check for ±5% reference
    if '±5%' in func_body or '5%' in func_body:
        print('[PASS] References ±5% thresholds')
    else:
        print('[WARN] No explicit ±5% reference')
else:
    print('[FAIL] run_backtest_generator function not found')

print()

# ==================================================================================
# CHECK TASK 5: ADAPTIVE STOCK RANKING
# ==================================================================================

print('=' * 80)
print('TASK 5: ADAPTIVE STOCK RANKING VERIFICATION')
print('=' * 80)

# Find the run_adaptive_ranking function
ranking_func_match = re.search(
    r'def run_adaptive_ranking\(\).*?(?=\ndef\s|\Z)',
    scheduler_content,
    re.DOTALL
)

if ranking_func_match:
    func_body = ranking_func_match.group(0)

    print('[PASS] run_adaptive_ranking function exists')

    # Check for correct import
    if 'from backend.turbomode.adaptive_stock_ranker import AdaptiveStockRanker' in func_body:
        print('[PASS] Imports AdaptiveStockRanker')
    else:
        print('[FAIL] Does not import AdaptiveStockRanker')

    # Check for WIN_FRACTION reference
    if 'WIN_FRACTION' in func_body or '3.0' in func_body:
        print('[PASS] References WIN_FRACTION or 3.0')
    else:
        print('[WARN] No WIN_FRACTION reference')

    # Check task_id
    if 'task_id = 5' in func_body:
        print('[PASS] task_id = 5 (correct)')
    else:
        print('[FAIL] task_id is not 5')
else:
    print('[FAIL] run_adaptive_ranking function not found')

print()

# ==================================================================================
# CHECK TASK 2: TRAINING ORCHESTRATOR
# ==================================================================================

print('=' * 80)
print('TASK 2: TRAINING ORCHESTRATOR VERIFICATION')
print('=' * 80)

# Find the run_orchestrator function
orchestrator_func_match = re.search(
    r'def run_orchestrator\(\).*?(?=\ndef\s|\Z)',
    scheduler_content,
    re.DOTALL
)

if orchestrator_func_match:
    func_body = orchestrator_func_match.group(0)

    # Check for correct import
    if 'train_all_sectors_optimized_orchestrator' in func_body:
        print('[PASS] References train_all_sectors_optimized_orchestrator')
    else:
        print('[WARN] No explicit reference to training orchestrator')

    # Check for deprecated references
    deprecated_patterns = ['label_1d_5pct', 'label_2d', 'horizon=']
    found_deprecated = []
    for pattern in deprecated_patterns:
        if pattern in func_body:
            found_deprecated.append(pattern)

    if found_deprecated:
        print(f'[FAIL] Found deprecated patterns: {found_deprecated}')
    else:
        print('[PASS] No deprecated label patterns found')
else:
    print('[WARN] run_orchestrator function not found in excerpt')

print()

# ==================================================================================
# CHECK TASK 3: OVERNIGHT SCANNER
# ==================================================================================

print('=' * 80)
print('TASK 3: OVERNIGHT SCANNER VERIFICATION')
print('=' * 80)

# Find the run_overnight_scanner function
scanner_func_match = re.search(
    r'def run_overnight_scanner\(\).*?(?=\ndef\s|\Z)',
    scheduler_content,
    re.DOTALL
)

if scanner_func_match:
    func_body = scanner_func_match.group(0)

    # Check for correct import
    if 'overnight_scanner' in func_body or 'ProductionScanner' in func_body:
        print('[PASS] References overnight_scanner or ProductionScanner')
    else:
        print('[WARN] No explicit scanner reference')

    # Check for deprecated scanner references
    if 'deprecated' in func_body.lower() or 'old' in func_body.lower():
        print('[FAIL] Contains deprecated scanner references')
    else:
        print('[PASS] No deprecated scanner references')
else:
    print('[WARN] run_overnight_scanner function not found in excerpt')

print()

# ==================================================================================
# VERIFY FILE LOCATIONS
# ==================================================================================

print('=' * 80)
print('FILE LOCATION VERIFICATION')
print('=' * 80)

files_to_check = [
    ('TurboMode Backtest', 'C:/StockApp/backend/turbomode/core_engine/turbomode_backtest.py'),
    ('Adaptive Stock Ranker', 'C:/StockApp/backend/turbomode/adaptive_stock_ranker.py'),
    ('Training Orchestrator', 'C:/StockApp/backend/turbomode/core_engine/train_all_sectors_optimized_orchestrator.py'),
    ('Overnight Scanner', 'C:/StockApp/backend/turbomode/core_engine/overnight_scanner.py'),
    ('Model Registry', 'C:/StockApp/backend/turbomode/core_engine/model_registry.py'),
]

for name, path in files_to_check:
    if os.path.exists(path):
        print(f'[PASS] {name}: {path}')
    else:
        print(f'[FAIL] {name}: NOT FOUND at {path}')

print()

# ==================================================================================
# CHECK FOR DEPRECATED FILES STILL BEING REFERENCED
# ==================================================================================

print('=' * 80)
print('DEPRECATED FILE REFERENCES CHECK')
print('=' * 80)

deprecated_patterns = [
    'backtest_generator.BacktestGenerator',
    'label_1d_5pct',
    'label_2d',
    'horizon=',
    'trained_5pct',
    'trained_10pct',
    'backend.advanced_ml'
]

found_deprecated = []
for pattern in deprecated_patterns:
    if pattern in scheduler_content:
        # Get context around the match
        matches = list(re.finditer(re.escape(pattern), scheduler_content))
        if matches:
            found_deprecated.append((pattern, len(matches)))

if found_deprecated:
    print('[FAIL] Found deprecated patterns in scheduler:')
    for pattern, count in found_deprecated:
        print(f'  {pattern}: {count} occurrence(s)')
else:
    print('[PASS] No deprecated patterns found in scheduler')

print()

# ==================================================================================
# FINAL SUMMARY
# ==================================================================================

print('=' * 80)
print('VERIFICATION SUMMARY')
print('=' * 80)
print()
print('Scheduler Config Tasks:')
print(f'  Total tasks: {len(config["scheduled_tasks"])}')
print(f'  Expected: 7 (Ingestion, Training, Scanner, Backtest, Ranking, Drift, Maintenance)')
print()
print('Critical Checks:')
print('  1. Task 4 uses turbomode_backtest.py (14-day engine)')
print('  2. Task 5 uses adaptive_stock_ranker.py (risk-adjusted ranking)')
print('  3. No deprecated backtest_generator.py references')
print('  4. All required files exist in correct locations')
print()
print('Review output above for [PASS]/[FAIL]/[WARN] markers')
print()
print('=' * 80)
print('VERIFICATION COMPLETE')
print('=' * 80)
