"""
Verbose Scanner Test - Shows All Predictions

Shows prediction probabilities for all symbols to verify the system
is working correctly, even when signals don't meet entry threshold.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.turbomode.overnight_scanner import ProductionScanner
from backend.turbomode.fastmode_inference import load_fastmode_models, predict_single
from backend.turbomode.adaptive_sltp import calculate_atr, calculate_adaptive_sltp
from backend.turbomode.core_symbols import get_symbol_metadata
from master_market_data.market_data_api import get_market_data_api
from backend.turbomode.turbomode_vectorized_feature_engine import TurboModeVectorizedFeatureEngine
import pandas as pd
import numpy as np


def test_technology_sector():
    """Test the complete pipeline for technology sector symbols"""

    print("=" * 80)
    print("TECHNOLOGY SECTOR SCANNER TEST")
    print("=" * 80)

    # Initialize components
    market_data_api = get_market_data_api()
    feature_engine = TurboModeVectorizedFeatureEngine()

    # Technology sector symbols
    tech_symbols = [
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'AVGO',
        'CRM', 'ADBE', 'ORCL', 'CSCO', 'QCOM'
    ]

    horizon = '1d'
    entry_threshold = 0.60

    results = []

    for symbol in tech_symbols:
        print(f"\n{'=' * 80}")
        print(f"SYMBOL: {symbol}")
        print(f"{'=' * 80}")

        try:
            # 1. Get OHLCV data
            df = market_data_api.get_candles(symbol, timeframe='1d', days_back=730)
            if df is None or len(df) < 400:
                print(f"[ERROR] Insufficient data for {symbol}")
                continue

            df = df.reset_index()
            df.rename(columns={'timestamp': 'date'}, inplace=True)

            print(f"[1] Data: {len(df)} rows")
            current_price = float(df['close'].iloc[-1])
            print(f"    Current price: ${current_price:.2f}")

            # 2. Calculate ATR
            atr = calculate_atr(df, period=14)
            print(f"[2] ATR (14): ${atr:.2f}")

            # 3. Extract features
            features_df = feature_engine.extract_features(df)
            if features_df is None or features_df.empty:
                print(f"[ERROR] Failed to extract features for {symbol}")
                continue

            features = features_df.iloc[-1].to_dict()
            metadata = get_symbol_metadata(symbol)
            features.update(metadata)

            from backend.turbomode.feature_list import FEATURE_LIST
            feature_array = np.array([features.get(f, 0.0) for f in FEATURE_LIST], dtype=np.float32)

            print(f"[3] Features extracted: {len(feature_array)} features")
            print(f"    Sector: {metadata.get('sector')}")

            # 4. Load models and predict
            sector = metadata.get('sector', 'unknown')
            models = load_fastmode_models(sector, horizon)

            prediction = predict_single(models, feature_array)

            print(f"[4] Fast Mode Prediction:")
            print(f"    Signal: {prediction['signal']}")
            print(f"    Prob BUY:  {prediction['prob_buy']:.4f} ({prediction['prob_buy']:.2%})")
            print(f"    Prob HOLD: {prediction['prob_hold']:.4f} ({prediction['prob_hold']:.2%})")
            print(f"    Prob SELL: {prediction['prob_sell']:.4f} ({prediction['prob_sell']:.2%})")
            print(f"    Confidence: {prediction['confidence']:.2%}")

            # 5. Check against entry threshold
            print(f"[5] Entry Check (threshold: {entry_threshold:.2%}):")

            if prediction['prob_buy'] >= entry_threshold:
                signal_type = 'BUY'
                print(f"    [OK] BUY signal (prob_buy {prediction['prob_buy']:.2%} >= {entry_threshold:.2%})")
            elif prediction['prob_sell'] >= entry_threshold:
                signal_type = 'SELL'
                print(f"    [OK] SELL signal (prob_sell {prediction['prob_sell']:.2%} >= {entry_threshold:.2%})")
            else:
                signal_type = None
                print(f"    [NO] No signal - below threshold")
                print(f"         BUY: {prediction['prob_buy']:.2%} < {entry_threshold:.2%}")
                print(f"         SELL: {prediction['prob_sell']:.2%} < {entry_threshold:.2%}")

                # Still track result
                results.append({
                    'symbol': symbol,
                    'signal': prediction['signal'],
                    'prob_buy': prediction['prob_buy'],
                    'prob_sell': prediction['prob_sell'],
                    'confidence': prediction['confidence'],
                    'meets_threshold': False
                })
                continue

            # 6. Calculate adaptive SL/TP
            position_type = 'long' if signal_type == 'BUY' else 'short'

            sltp = calculate_adaptive_sltp(
                entry_price=current_price,
                atr=atr,
                sector=sector,
                confidence=prediction['confidence'],
                horizon=horizon,
                position_type=position_type,
                reward_ratio=2.5
            )

            print(f"[6] Adaptive SL/TP (ATR-based):")
            print(f"    Position: {position_type.upper()}")
            print(f"    Entry: ${current_price:.2f}")
            print(f"    Stop:  ${sltp['stop_price']:.2f} (-${sltp['stop_distance']:.2f})")
            print(f"    Target: ${sltp['target_price']:.2f} (+${sltp['target_distance']:.2f})")
            print(f"    +1R: ${sltp['r1_price']:.2f}")
            print(f"    +2R: ${sltp['r2_price']:.2f}")
            print(f"    +3R: ${sltp['r3_price']:.2f}")

            reward_ratio = sltp['target_distance'] / sltp['stop_distance']
            print(f"    Reward:Risk = {reward_ratio:.2f}:1")

            results.append({
                'symbol': symbol,
                'signal': prediction['signal'],
                'signal_type': signal_type,
                'prob_buy': prediction['prob_buy'],
                'prob_sell': prediction['prob_sell'],
                'confidence': prediction['confidence'],
                'meets_threshold': True,
                'entry_price': current_price,
                'stop_price': sltp['stop_price'],
                'target_price': sltp['target_price'],
                'atr': atr
            })

        except Exception as e:
            print(f"[ERROR] Failed to process {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    signals = [r for r in results if r.get('meets_threshold', False)]
    no_signals = [r for r in results if not r.get('meets_threshold', False)]

    print(f"Symbols tested: {len(results)}")
    print(f"Signals generated: {len(signals)}")
    print(f"No signal: {len(no_signals)}")

    if signals:
        print(f"\n[SIGNALS GENERATED]")
        for r in signals:
            print(f"  {r['symbol']}: {r['signal_type']} @ {r['confidence']:.2%} (entry: ${r['entry_price']:.2f})")

    if no_signals:
        print(f"\n[ALL PREDICTIONS - INCLUDING BELOW THRESHOLD]")
        for r in sorted(no_signals, key=lambda x: x['confidence'], reverse=True):
            print(f"  {r['symbol']}: {r['signal']} @ {r['confidence']:.2%} "
                  f"(BUY: {r['prob_buy']:.2%}, SELL: {r['prob_sell']:.2%})")

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    test_technology_sector()
