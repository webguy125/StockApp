#!/usr/bin/env python
"""
Generate Full Predictions for Web Display

This script generates BUY/SELL/HOLD predictions for all scanning symbols
and writes them to all_predictions.json for the full model list webpage.

Unlike the production scanner (which only saves actionable BUY/SELL signals
to the database), this script generates ALL predictions including HOLD signals
for comprehensive market analysis display.

Author: TurboMode Team
Date: 2026-01-26
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = current_dir
turbomode_dir = os.path.dirname(frontend_dir)
backend_dir = os.path.join(turbomode_dir, 'backend')
project_root = os.path.dirname(turbomode_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.turbomode.core_engine.overnight_scanner import ProductionScanner


def generate_all_predictions():
    """
    Generate predictions for all symbols including HOLD signals.

    Returns:
        Dictionary with predictions data
    """
    print("=" * 80)
    print("GENERATING FULL PREDICTIONS SET (BUY/SELL/HOLD)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize scanner
    print("Initializing scanner...")
    scanner = ProductionScanner()

    # Load CORE_230 universe (canonical: ticker)
    print("Loading symbol list...")
    core_230_path = Path(r"C:/StockApp/config/symbols/CORE_230.json")
    with open(core_230_path, 'r', encoding='utf-8') as f:
        core_230_data = json.load(f)

    symbol_metadata = {entry['ticker']: entry for entry in core_230_data}
    all_symbols = list(symbol_metadata.keys())

    print(f"  Total symbols: {len(all_symbols)}")
    print()

    predictions = []
    buy_count = 0
    sell_count = 0
    hold_count = 0
    failed = 0

    print(f"Generating predictions for {len(all_symbols)} symbols...")
    print()

    for i, symbol in enumerate(all_symbols, 1):
        if i % 50 == 0:
            print(
                f"  Progress: {i}/{len(all_symbols)} ({i/len(all_symbols)*100:.1f}%) - "
                f"BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}"
            )

        try:
            df = scanner.get_ohlcv_dataframe(symbol, days_back=730)
            if df is None:
                failed += 1
                continue

            current_price = scanner.get_current_price(symbol)
            if current_price is None:
                failed += 1
                continue

            features = scanner.extract_features(df, symbol)
            if features is None:
                failed += 1
                continue

            prediction = scanner.get_prediction(symbol, features)
            if prediction is None:
                failed += 1
                continue

            metadata = symbol_metadata.get(symbol, {})
            sector = metadata.get('sector', 'unknown')

            pred_entry = {
                'symbol': symbol,
                'sector': sector,
                'market_cap_category': metadata.get('market_cap_category', 'unknown'),
                'sector_code': metadata.get('sector_code', 'unknown'),
                'prediction': prediction['signal'],
                'confidence': round(prediction['confidence'], 4),
                'prob_buy': round(prediction['prob_buy'], 4),
                'prob_sell': round(prediction['prob_sell'], 4),
                'prob_hold': round(prediction['prob_hold'], 4),
                'current_price': round(current_price, 2)
            }

            predictions.append(pred_entry)

            if prediction['signal'] == 'BUY':
                buy_count += 1
            elif prediction['signal'] == 'SELL':
                sell_count += 1
            elif prediction['signal'] == 'HOLD':
                hold_count += 1

        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            failed += 1
            continue

    print()
    print("=" * 80)
    print("PREDICTION GENERATION COMPLETE")
    print("=" * 80)
    print(f"  Total predictions: {len(predictions)}")
    if len(predictions) > 0:
        print(f"  BUY: {buy_count} ({buy_count/len(predictions)*100:.1f}%)")
        print(f"  SELL: {sell_count} ({sell_count/len(predictions)*100:.1f}%)")
        print(f"  HOLD: {hold_count} ({hold_count/len(predictions)*100:.1f}%)")
    print(f"  Failed: {failed}")
    print()

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total': len(predictions),
        'statistics': {
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count
        },
        'predictions': predictions,
        'source': 'generate_predictions_for_web.py',
        'note': 'Complete prediction set including HOLD signals for web display'
    }

    return output_data


def main():
    try:
        output_data = generate_all_predictions()

        output_file = os.path.join(
            project_root,
            'backend',
            'turbomode',
            'data',
            'all_predictions.json'
        )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f"Writing to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"[OK] Predictions saved to {output_file}")
        print()
        print("=" * 80)
        print("SUCCESS: Full predictions file generated")
        print("=" * 80)
        print()
        print("The webpage will now show:")
        print(f"  - {output_data['statistics']['buy_count']} BUY signals")
        print(f"  - {output_data['statistics']['sell_count']} SELL signals")
        print(f"  - {output_data['statistics']['hold_count']} HOLD signals")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Generation cancelled by user")
        return 130

    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
