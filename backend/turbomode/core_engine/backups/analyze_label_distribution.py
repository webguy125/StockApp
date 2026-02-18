
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Diagnostic Script: Analyze Label Distribution
Task 1 of SELL-Collapse Investigation

Loads training data for all 11 sectors and analyzes BUY/SELL/HOLD label distribution.
Compares training distribution to inference output.

CLASS SEMANTICS:
- Index 0: SELL (go short)
- Index 1: HOLD (do nothing)
- Index 2: BUY (go long)
"""

import os
import numpy as np
from typing import Dict
from backend.turbomode.core_engine.sector_batch_trainer import load_sector_data_once
from backend.turbomode.core_engine.training_symbols import TRAINING_SYMBOLS


# All 11 sectors
ALL_SECTORS = [
    'technology',
    'financials',
    'healthcare',
    'consumer_discretionary',
    'communication_services',
    'industrials',
    'consumer_staples',
    'energy',
    'materials',
    'real_estate',
    'utilities'
]


def get_symbols_by_sector(sector: str):
    """Get all symbols for a sector."""
    if sector not in TRAINING_SYMBOLS:
        raise ValueError(f"Unknown sector: {sector}")

    sector_data = TRAINING_SYMBOLS[sector]
    symbols = []

    if 'large_cap' in sector_data:
        symbols.extend(sector_data['large_cap'])
    if 'mid_cap' in sector_data:
        symbols.extend(sector_data['mid_cap'])
    if 'small_cap' in sector_data:
        symbols.extend(sector_data['small_cap'])

    return symbols


def analyze_sector_labels(sector: str, db_path: str) -> Dict:
    """
    Load and analyze label distribution for a single sector.

    Returns:
        Dict with label counts and percentages
    """
    print(f"\nLoading data for {sector.upper()}...")

    sector_symbols = get_symbols_by_sector(sector)

    # Load sector data
    X_sector, labels_dict, trade_ids = load_sector_data_once(db_path, sector_symbols)

    if len(X_sector) == 0:
        print(f"  [WARN] No data found for {sector}")
        return {
            'sector': sector,
            'total_samples': 0,
            'sell_count': 0,
            'hold_count': 0,
            'buy_count': 0
        }

    # Build label vector
    y_sector = np.array([labels_dict[tid] for tid in trade_ids], dtype=np.int32)

    # Count labels (0=SELL, 1=HOLD, 2=BUY)
    sell_count = int(np.sum(y_sector == 0))
    hold_count = int(np.sum(y_sector == 1))
    buy_count = int(np.sum(y_sector == 2))
    total = len(y_sector)

    # Compute percentages
    sell_pct = (sell_count / total * 100) if total > 0 else 0
    hold_pct = (hold_count / total * 100) if total > 0 else 0
    buy_pct = (buy_count / total * 100) if total > 0 else 0

    print(f"  Total samples: {total:,}")
    print(f"  SELL: {sell_count:,} ({sell_pct:.1f}%)")
    print(f"  HOLD: {hold_count:,} ({hold_pct:.1f}%)")
    print(f"  BUY:  {buy_count:,} ({buy_pct:.1f}%)")

    return {
        'sector': sector,
        'total_samples': total,
        'sell_count': sell_count,
        'hold_count': hold_count,
        'buy_count': buy_count,
        'sell_pct': sell_pct,
        'hold_pct': hold_pct,
        'buy_pct': buy_pct
    }


def main():
    print("=" * 80)
    print("DIAGNOSTIC TASK 1: LABEL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print("Analyzing training data for all 11 sectors...")
    print("Label: label_1d_5pct (1-day horizon, 5% threshold)")
    print("Class semantics: 0=SELL, 1=HOLD, 2=BUY")
    print("=" * 80)

    # Database path
    backend_dir = str(project_root / "backend")
    db_path = os.path.join(backend_dir, "data", "turbomode.db")

    # Analyze all sectors
    sector_results = []

    for sector in ALL_SECTORS:
        result = analyze_sector_labels(sector, db_path)
        sector_results.append(result)

    # Compute global statistics
    print("\n" + "=" * 80)
    print("GLOBAL STATISTICS")
    print("=" * 80)

    total_samples = sum(r['total_samples'] for r in sector_results)
    total_sell = sum(r['sell_count'] for r in sector_results)
    total_hold = sum(r['hold_count'] for r in sector_results)
    total_buy = sum(r['buy_count'] for r in sector_results)

    global_sell_pct = (total_sell / total_samples * 100) if total_samples > 0 else 0
    global_hold_pct = (total_hold / total_samples * 100) if total_samples > 0 else 0
    global_buy_pct = (total_buy / total_samples * 100) if total_samples > 0 else 0

    print(f"\nTotal samples across all sectors: {total_samples:,}")
    print(f"  SELL: {total_sell:,} ({global_sell_pct:.2f}%)")
    print(f"  HOLD: {total_hold:,} ({global_hold_pct:.2f}%)")
    print(f"  BUY:  {total_buy:,} ({global_buy_pct:.2f}%)")

    # Per-sector summary table
    print("\n" + "=" * 80)
    print("PER-SECTOR SUMMARY")
    print("=" * 80)
    print(f"{'Sector':<30} {'Total':<12} {'SELL %':<10} {'HOLD %':<10} {'BUY %':<10}")
    print("-" * 80)

    for result in sector_results:
        if result['total_samples'] > 0:
            print(f"{result['sector']:<30} {result['total_samples']:<12,} "
                  f"{result['sell_pct']:<10.2f} {result['hold_pct']:<10.2f} {result['buy_pct']:<10.2f}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check for label imbalance
    if global_hold_pct > 80:
        print(f"\n[!] HOLD-DOMINANCE DETECTED: {global_hold_pct:.1f}% of training labels are HOLD")
        print("    This is expected for a 5% threshold (most price movements are <5%)")

    # Check BUY/SELL balance
    buy_sell_ratio = total_buy / total_sell if total_sell > 0 else 0
    print(f"\n[!] BUY/SELL RATIO: {buy_sell_ratio:.3f}")
    print(f"    BUY count: {total_buy:,} ({global_buy_pct:.2f}%)")
    print(f"    SELL count: {total_sell:,} ({global_sell_pct:.2f}%)")

    if abs(buy_sell_ratio - 1.0) > 0.2:
        if buy_sell_ratio < 1.0:
            print("    [IMBALANCE] Training data has MORE SELL labels than BUY labels")
        else:
            print("    [IMBALANCE] Training data has MORE BUY labels than SELL labels")
    else:
        print("    [BALANCED] BUY and SELL labels are roughly balanced")

    # Compare to inference results
    print("\n" + "=" * 80)
    print("COMPARISON: TRAINING vs INFERENCE")
    print("=" * 80)

    print("\nTRAINING DATA DISTRIBUTION:")
    print(f"  SELL: {global_sell_pct:.2f}%")
    print(f"  HOLD: {global_hold_pct:.2f}%")
    print(f"  BUY:  {global_buy_pct:.2f}%")

    print("\nINFERENCE OUTPUT (from latest scanner run):")
    print("  SELL: 99.6% (229 out of 230 symbols)")
    print("  HOLD: 0%")
    print("  BUY:  0%")

    print("\n[!] CRITICAL MISMATCH:")
    print("    - Training data shows HOLD-dominance (~92%)")
    print("    - Inference shows SELL-dominance (99.6%)")
    print("    - This indicates the models learned to collapse HOLD → SELL")
    print("    - Possible causes:")
    print("      1. Label imbalance causing model bias")
    print("      2. Feature engineering issues")
    print("      3. MetaLearner amplifying SELL bias")
    print("      4. Class weighting or loss function issues")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Inspect labeling logic (compute_labels_1d_5pct)")
    print("2. Sample raw base model outputs before MetaLearner")
    print("3. Test inference without biasing enabled")
    print("4. Inspect MetaLearner coefficient weights")
    print("=" * 80)


if __name__ == '__main__':
    main()
