
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Diagnostic Script: Sample Base Model Outputs
Task 3 of SELL-Collapse Investigation

Samples raw predict_proba outputs from all 5 base models BEFORE MetaLearner
to identify which models are producing SELL-biased predictions.

CLASS SEMANTICS:
- Index 0: SELL (go short) - BUT WAIT, let's verify this!
- Index 1: HOLD (do nothing)
- Index 2: BUY (go long)

NOTE: There might be a class mapping inversion issue!
"""

import numpy as np
import os
from typing import Dict, List
from backend.turbomode.core_engine.model_loader import load_sector_models
from backend.turbomode.core_engine.model_registry import BASE_MODELS, SECTORS
from backend.turbomode.core_engine.turbomode_vectorized_feature_engine import (
    generate_features_for_symbol
)


def sample_base_model_predictions(symbols: List[str], num_samples: int = 10):
    """
    Sample raw base model predictions for given symbols.

    For each symbol:
    1. Load current market data
    2. Generate features
    3. Run predict_proba on all 5 base models
    4. Print raw probabilities BEFORE MetaLearner

    This will reveal if base models are SELL-biased or if MetaLearner causes collapse.
    """
    print("=" * 80)
    print("DIAGNOSTIC TASK 3: BASE MODEL OUTPUT SAMPLING")
    print("=" * 80)
    print(f"Sampling {num_samples} symbols to analyze base model outputs")
    print("Class semantics: [SELL, HOLD, BUY] - indices [0, 1, 2]")
    print("=" * 80)
    print()

    # Map symbols to sectors
    from backend.turbomode.core_engine.training_symbols import TRAINING_SYMBOLS

    symbol_to_sector = {}
    for sector, data in TRAINING_SYMBOLS.items():
        for cap_type in ['large_cap', 'mid_cap', 'small_cap']:
            if cap_type in data:
                for symbol in data[cap_type]:
                    symbol_to_sector[symbol] = sector

    results = []

    for i, symbol in enumerate(symbols[:num_samples], 1):
        print(f"\n[{i}/{num_samples}] Symbol: {symbol}")
        print("-" * 80)

        # Get sector
        sector = symbol_to_sector.get(symbol)
        if not sector:
            print(f"  [SKIP] Symbol not in training symbols")
            continue

        print(f"  Sector: {sector}")

        try:
            # Generate features
            print(f"  Generating features...")
            features = generate_features_for_symbol(symbol)

            if features is None:
                print(f"  [SKIP] Failed to generate features")
                continue

            # Convert to numpy array
            from backend.turbomode.core_engine.feature_list import features_to_array
            X = features_to_array(features, fill_value=0.0)
            X = np.array([X], dtype=np.float32)  # shape: (1, 179)

            print(f"  Feature shape: {X.shape}")

            # Load models for this sector
            print(f"  Loading models...")
            models = load_sector_models(sector)

            # Run predict_proba on all 5 base models
            print(f"  Base model predictions:")

            base_predictions = []
            for model_name in BASE_MODELS:
                model = models[model_name]
                probs = model.predict_proba(X)  # shape: (1, 3)

                # Extract probabilities
                prob_0 = float(probs[0, 0])
                prob_1 = float(probs[0, 1])
                prob_2 = float(probs[0, 2])

                argmax_idx = int(np.argmax(probs[0]))
                argmax_label = ['SELL', 'HOLD', 'BUY'][argmax_idx]

                print(f"    {model_name:20s}: SELL={prob_0:.3f}, HOLD={prob_1:.3f}, BUY={prob_2:.3f} -> {argmax_label}")

                base_predictions.append({
                    'model': model_name,
                    'prob_sell': prob_0,
                    'prob_hold': prob_1,
                    'prob_buy': prob_2,
                    'argmax': argmax_label
                })

            # Compute average across base models
            avg_sell = np.mean([p['prob_sell'] for p in base_predictions])
            avg_hold = np.mean([p['prob_hold'] for p in base_predictions])
            avg_buy = np.mean([p['prob_buy'] for p in base_predictions])

            print(f"  Average across base models:")
            print(f"    SELL={avg_sell:.3f}, HOLD={avg_hold:.3f}, BUY={avg_buy:.3f}")

            # Stack for MetaLearner
            stacked_features = np.concatenate([probs for probs in [
                models[model_name].predict_proba(X) for model_name in BASE_MODELS
            ]], axis=1)  # shape: (1, 15)

            # Run MetaLearner
            meta_learner = models['meta_learner']
            final_probs = meta_learner.predict_proba(stacked_features)

            final_sell = float(final_probs[0, 0])
            final_hold = float(final_probs[0, 1])
            final_buy = float(final_probs[0, 2])
            final_argmax_idx = int(np.argmax(final_probs[0]))
            final_argmax = ['SELL', 'HOLD', 'BUY'][final_argmax_idx]

            print(f"  MetaLearner output:")
            print(f"    SELL={final_sell:.3f}, HOLD={final_hold:.3f}, BUY={final_buy:.3f} -> {final_argmax}")

            results.append({
                'symbol': symbol,
                'sector': sector,
                'base_avg_sell': avg_sell,
                'base_avg_hold': avg_hold,
                'base_avg_buy': avg_buy,
                'meta_sell': final_sell,
                'meta_hold': final_hold,
                'meta_buy': final_buy,
                'meta_signal': final_argmax,
                'base_predictions': base_predictions
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY ANALYSIS")
    print("=" * 80)

    if not results:
        print("No results to analyze")
        return

    # Count base model predictions
    base_sell_count = sum(1 for r in results if r['base_avg_sell'] > max(r['base_avg_hold'], r['base_avg_buy']))
    base_hold_count = sum(1 for r in results if r['base_avg_hold'] > max(r['base_avg_sell'], r['base_avg_buy']))
    base_buy_count = sum(1 for r in results if r['base_avg_buy'] > max(r['base_avg_sell'], r['base_avg_hold']))

    # Count MetaLearner predictions
    meta_sell_count = sum(1 for r in results if r['meta_signal'] == 'SELL')
    meta_hold_count = sum(1 for r in results if r['meta_signal'] == 'HOLD')
    meta_buy_count = sum(1 for r in results if r['meta_signal'] == 'BUY')

    print(f"\nBase Models (average across 5 models):")
    print(f"  SELL signals: {base_sell_count}/{len(results)} ({base_sell_count/len(results)*100:.1f}%)")
    print(f"  HOLD signals: {base_hold_count}/{len(results)} ({base_hold_count/len(results)*100:.1f}%)")
    print(f"  BUY signals:  {base_buy_count}/{len(results)} ({base_buy_count/len(results)*100:.1f}%)")

    print(f"\nMetaLearner (final output):")
    print(f"  SELL signals: {meta_sell_count}/{len(results)} ({meta_sell_count/len(results)*100:.1f}%)")
    print(f"  HOLD signals: {meta_hold_count}/{len(results)} ({meta_hold_count/len(results)*100:.1f}%)")
    print(f"  BUY signals:  {meta_buy_count}/{len(results)} ({meta_buy_count/len(results)*100:.1f}%)")

    # Average probabilities
    avg_base_sell = np.mean([r['base_avg_sell'] for r in results])
    avg_base_hold = np.mean([r['base_avg_hold'] for r in results])
    avg_base_buy = np.mean([r['base_avg_buy'] for r in results])

    avg_meta_sell = np.mean([r['meta_sell'] for r in results])
    avg_meta_hold = np.mean([r['meta_hold'] for r in results])
    avg_meta_buy = np.mean([r['meta_buy'] for r in results])

    print(f"\nAverage Probabilities:")
    print(f"  Base Models: SELL={avg_base_sell:.3f}, HOLD={avg_base_hold:.3f}, BUY={avg_base_buy:.3f}")
    print(f"  MetaLearner: SELL={avg_meta_sell:.3f}, HOLD={avg_meta_hold:.3f}, BUY={avg_meta_buy:.3f}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if avg_base_sell > 0.7:
        print("\n[!] BASE MODELS ARE SELL-BIASED")
        print(f"    Average SELL probability: {avg_base_sell:.1%}")
        print("    The problem originates in the base models themselves!")
    elif avg_meta_sell > 0.7 and avg_base_sell < 0.5:
        print("\n[!] META-LEARNER IS AMPLIFYING SELL BIAS")
        print(f"    Base models average SELL: {avg_base_sell:.1%}")
        print(f"    MetaLearner average SELL: {avg_meta_sell:.1%}")
        print("    The MetaLearner is collapsing predictions to SELL!")
    else:
        print("\n[?] UNCLEAR PATTERN")
        print("    Further investigation needed")

    # Check for class mapping inversion
    if avg_base_sell > 0.9 and avg_meta_sell > 0.9:
        print("\n[!] POSSIBLE CLASS MAPPING INVERSION")
        print("    Both base models and MetaLearner show extreme SELL bias")
        print("    This might indicate:")
        print("      1. Class 0 is actually BUY (not SELL)")
        print("      2. Training used different class semantics than inference")
        print("      3. Labels were inverted during training")

    print("\n" + "=" * 80)

    return results


if __name__ == '__main__':
    # Sample 10 symbols from scanner output
    test_symbols = [
        'AAPL',   # Technology
        'MSFT',   # Technology
        'JPM',    # Financials
        'BAC',    # Financials
        'JNJ',    # Healthcare
        'UNH',    # Healthcare
        'AMZN',   # Consumer Discretionary
        'HD',     # Consumer Discretionary
        'XOM',    # Energy
        'CVX'     # Energy
    ]

    results = sample_base_model_predictions(test_symbols, num_samples=10)
