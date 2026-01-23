"""
Quick Scanner Test - Dual Threshold Comparison
Tests scanner with BOTH 5% and 10% threshold models side-by-side

Shows how the two thresholds generate different signals for the same market conditions.
Expected runtime: ~3-5 minutes for 10 symbols × 2 thresholds
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

from backend.turbomode.fastmode_inference import load_fastmode_models_5pct, load_fastmode_models_10pct, predict_single
from backend.turbomode.adaptive_sltp import calculate_atr, calculate_adaptive_sltp
from backend.turbomode.core_symbols import get_symbol_metadata
from master_market_data.market_data_api import get_market_data_api
from backend.turbomode.turbomode_vectorized_feature_engine import TurboModeVectorizedFeatureEngine
import numpy as np
from datetime import datetime


def test_dual_threshold_scanner():
    """Test scanner with both 5% and 10% threshold models"""

    print("=" * 80)
    print("DUAL-THRESHOLD SCANNER TEST")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"5% Models: trained_5pct/technology/1d/")
    print(f"10% Models: trained_10pct/technology/1d/")
    print("=" * 80)

    # Initialize components
    market_data_api = get_market_data_api()
    feature_engine = TurboModeVectorizedFeatureEngine()

    # Technology sector symbols (10 symbols for quick test)
    tech_symbols = [
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'AVGO',
        'CRM', 'ADBE', 'ORCL', 'CSCO', 'QCOM'
    ]

    horizon = '1d'
    entry_threshold = 0.60

    # Load both threshold models
    print(f"\n[INIT] Loading models...")

    try:
        models_5pct = load_fastmode_models_5pct('technology', '1d')
        print(f"[OK] 5% threshold models loaded")

        models_10pct = load_fastmode_models_10pct('technology', '1d')
        print(f"[OK] 10% threshold models loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return

    results_5pct = []
    results_10pct = []

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

            # 3. Extract features (once, used for both models)
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

            # 4A. Predict with 5% threshold models
            print(f"\n[4A] 5% THRESHOLD PREDICTION:")
            pred_5pct = predict_single(models_5pct, feature_array)

            print(f"    Signal: {pred_5pct['signal']}")
            print(f"    Prob BUY:  {pred_5pct['prob_buy']:.2%}")
            print(f"    Prob HOLD: {pred_5pct['prob_hold']:.2%}")
            print(f"    Prob SELL: {pred_5pct['prob_sell']:.2%}")
            print(f"    Confidence: {pred_5pct['confidence']:.2%}")

            signal_5pct = None
            if pred_5pct['prob_buy'] >= entry_threshold:
                signal_5pct = 'BUY'
                print(f"    [SIGNAL] BUY (prob_buy {pred_5pct['prob_buy']:.2%} >= {entry_threshold:.2%})")
            elif pred_5pct['prob_sell'] >= entry_threshold:
                signal_5pct = 'SELL'
                print(f"    [SIGNAL] SELL (prob_sell {pred_5pct['prob_sell']:.2%} >= {entry_threshold:.2%})")
            else:
                print(f"    [NO SIGNAL] Below threshold")

            # 4B. Predict with 10% threshold models
            print(f"\n[4B] 10% THRESHOLD PREDICTION:")
            pred_10pct = predict_single(models_10pct, feature_array)

            print(f"    Signal: {pred_10pct['signal']}")
            print(f"    Prob BUY:  {pred_10pct['prob_buy']:.2%}")
            print(f"    Prob HOLD: {pred_10pct['prob_hold']:.2%}")
            print(f"    Prob SELL: {pred_10pct['prob_sell']:.2%}")
            print(f"    Confidence: {pred_10pct['confidence']:.2%}")

            signal_10pct = None
            if pred_10pct['prob_buy'] >= entry_threshold:
                signal_10pct = 'BUY'
                print(f"    [SIGNAL] BUY (prob_buy {pred_10pct['prob_buy']:.2%} >= {entry_threshold:.2%})")
            elif pred_10pct['prob_sell'] >= entry_threshold:
                signal_10pct = 'SELL'
                print(f"    [SIGNAL] SELL (prob_sell {pred_10pct['prob_sell']:.2%} >= {entry_threshold:.2%})")
            else:
                print(f"    [NO SIGNAL] Below threshold")

            # 5. Compare predictions
            print(f"\n[5] COMPARISON:")
            if signal_5pct and signal_10pct:
                if signal_5pct == signal_10pct:
                    print(f"    AGREEMENT: Both models signal {signal_5pct}")
                else:
                    print(f"    CONFLICT: 5% says {signal_5pct}, 10% says {signal_10pct}")
            elif signal_5pct and not signal_10pct:
                print(f"    5% ONLY: 5% signals {signal_5pct}, 10% has no signal")
            elif signal_10pct and not signal_5pct:
                print(f"    10% ONLY: 10% signals {signal_10pct}, 5% has no signal")
            else:
                print(f"    NO SIGNALS: Neither model meets entry threshold")

            # Store results
            results_5pct.append({
                'symbol': symbol,
                'signal': pred_5pct['signal'],
                'signal_type': signal_5pct,
                'prob_buy': pred_5pct['prob_buy'],
                'prob_sell': pred_5pct['prob_sell'],
                'confidence': pred_5pct['confidence'],
                'meets_threshold': signal_5pct is not None
            })

            results_10pct.append({
                'symbol': symbol,
                'signal': pred_10pct['signal'],
                'signal_type': signal_10pct,
                'prob_buy': pred_10pct['prob_buy'],
                'prob_sell': pred_10pct['prob_sell'],
                'confidence': pred_10pct['confidence'],
                'meets_threshold': signal_10pct is not None
            })

        except Exception as e:
            print(f"[ERROR] Failed to process {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    print(f"\nSymbols tested: {len(tech_symbols)}")

    signals_5pct = [r for r in results_5pct if r.get('meets_threshold', False)]
    signals_10pct = [r for r in results_10pct if r.get('meets_threshold', False)]

    print(f"\n[5% THRESHOLD MODELS]")
    print(f"  Signals generated: {len(signals_5pct)}")
    if signals_5pct:
        for r in signals_5pct:
            print(f"    {r['symbol']}: {r['signal_type']} @ {r['confidence']:.2%}")

    print(f"\n[10% THRESHOLD MODELS]")
    print(f"  Signals generated: {len(signals_10pct)}")
    if signals_10pct:
        for r in signals_10pct:
            print(f"    {r['symbol']}: {r['signal_type']} @ {r['confidence']:.2%}")

    # Agreement analysis
    print(f"\n[AGREEMENT ANALYSIS]")
    agreements = 0
    conflicts = 0
    only_5pct = 0
    only_10pct = 0
    neither = 0

    for r5, r10 in zip(results_5pct, results_10pct):
        if r5['meets_threshold'] and r10['meets_threshold']:
            if r5['signal_type'] == r10['signal_type']:
                agreements += 1
            else:
                conflicts += 1
        elif r5['meets_threshold']:
            only_5pct += 1
        elif r10['meets_threshold']:
            only_10pct += 1
        else:
            neither += 1

    print(f"  Both agree:         {agreements}")
    print(f"  Both conflict:      {conflicts}")
    print(f"  Only 5% signals:    {only_5pct}")
    print(f"  Only 10% signals:   {only_10pct}")
    print(f"  Neither signals:    {neither}")

    # All predictions sorted by confidence
    print(f"\n[ALL PREDICTIONS - SORTED BY CONFIDENCE]")
    print(f"\n5% Threshold:")
    for r in sorted(results_5pct, key=lambda x: x['confidence'], reverse=True):
        signal_str = f"{r['signal_type']}" if r['signal_type'] else "NO SIGNAL"
        print(f"  {r['symbol']:6s}: {signal_str:10s} @ {r['confidence']:5.2%} (BUY: {r['prob_buy']:5.2%}, SELL: {r['prob_sell']:5.2%})")

    print(f"\n10% Threshold:")
    for r in sorted(results_10pct, key=lambda x: x['confidence'], reverse=True):
        signal_str = f"{r['signal_type']}" if r['signal_type'] else "NO SIGNAL"
        print(f"  {r['symbol']:6s}: {signal_str:10s} @ {r['confidence']:5.2%} (BUY: {r['prob_buy']:5.2%}, SELL: {r['prob_sell']:5.2%})")

    print(f"\n{'=' * 80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("DUAL-THRESHOLD SCANNER TEST COMPLETE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    test_dual_threshold_scanner()
