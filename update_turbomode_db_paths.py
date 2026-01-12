"""
Update all TurboMode files to use turbomode.db instead of advanced_ml_system.db
"""
import os
import re

files_to_update = [
    "backend/turbomode/adaptive_stock_ranker.py",
    "backend/turbomode/retrain_meta_learner_only.py",
    "backend/turbomode/select_best_features.py",
    "backend/turbomode/train_specialized_meta_learner.py",
    "backend/turbomode/weekly_backtest.py"
]

replacements = [
    (r'"advanced_ml_system\.db"', '"turbomode.db"'),
    (r"'advanced_ml_system\.db'", "'turbomode.db'"),
    (r'advanced_ml_system\.db', 'turbomode.db'),
]

print("=" * 80)
print("UPDATING TURBOMODE FILES TO USE turbomode.db")
print("=" * 80)

for file_path in files_to_update:
    if not os.path.exists(file_path):
        print(f"[SKIP] {file_path} - File not found")
        continue

    print(f"\n[UPDATE] {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes_made = 0

    for pattern, replacement in replacements:
        matches = len(re.findall(pattern, content))
        if matches > 0:
            content = re.sub(pattern, replacement, content)
            print(f"  - Replaced {matches} occurrence(s) of {pattern}")
            changes_made += matches

    if changes_made > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [OK] {changes_made} changes saved")
    else:
        print(f"  [SKIP] No changes needed")

print("\n" + "=" * 80)
print("ALL TURBOMODE FILES UPDATED")
print("=" * 80)
print("TurboMode is now completely autonomous")
print("No dependency on Slipstream's database")
print("=" * 80)
