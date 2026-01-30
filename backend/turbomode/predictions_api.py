#!/usr/bin/env python

"""
Predictions API Module
Flask Blueprint for viewing all model predictions with confidence levels.
Supports: /all, /all_live, /symbol/<symbol>
Aligned with ProductionScanner and CORE_230.json.
"""

from flask import Blueprint, jsonify
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
turbomode_dir = os.path.dirname(current_dir)
backend_dir = os.path.dirname(turbomode_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Scanner (lazy-loaded)
scanner = None

# Flask Blueprint
predictions_bp = Blueprint('predictions', __name__, url_prefix='/turbomode/predictions')

# Cache for /all_live
predictions_cache = {
    'data': None,
    'timestamp': None,
    'expiry_minutes': 5
}


# ---------------------------------------------------------------------------
# Utility: Load CORE_230.json explicitly (deterministic universe)
# ---------------------------------------------------------------------------

def load_core_230_symbols():
    core_230_path = Path(r"C:/StockApp/config/symbols/CORE_230.json")
    with open(core_230_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [entry['ticker'] for entry in data], {entry['ticker']: entry for entry in data}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def get_cached_predictions():
    if predictions_cache['data'] is None or predictions_cache['timestamp'] is None:
        return None

    elapsed = (datetime.now() - predictions_cache['timestamp']).total_seconds() / 60
    if elapsed < predictions_cache['expiry_minutes']:
        print(f"[PREDICTIONS API] Using cached data ({elapsed:.1f} minutes old)")
        return predictions_cache['data']

    print(f"[PREDICTIONS API] Cache expired ({elapsed:.1f} minutes old), refreshing...")
    return None


def cache_predictions(data):
    predictions_cache['data'] = data
    predictions_cache['timestamp'] = datetime.now()
    print(f"[PREDICTIONS API] Cached {len(data['predictions'])} predictions")


# ---------------------------------------------------------------------------
# Lazy scanner initialization
# ---------------------------------------------------------------------------

def init_scanner():
    global scanner
    if scanner is None:
        print("[PREDICTIONS API] Initializing ProductionScanner...")
        from backend.turbomode.core_engine.overnight_scanner import ProductionScanner
        scanner = ProductionScanner()
        print("[PREDICTIONS API] Scanner ready")


# ---------------------------------------------------------------------------
# /all  — Read from all_predictions.json
# ---------------------------------------------------------------------------

@predictions_bp.route('/all', methods=['GET'])
def get_all_predictions():
    try:
        from backend.turbomode.database_schema import TurboModeDB

        db = TurboModeDB()
        active_signals = db.get_active_signals(limit=1000)

        predictions = []
        for signal in active_signals:
            predictions.append({
                'symbol': signal['symbol'],
                'prediction': signal['signal_type'],
                'confidence': signal['confidence'],
                'current_price': signal['current_price'],
                'entry_price': signal['entry_price'],
                'target_price': signal['target_price'],
                'stop_price': signal['stop_price'],
                'sector': signal['sector'],
                'market_cap_category': signal['market_cap'],
                'age_days': signal['age_days'],
                'entry_date': signal['entry_date'],
                'atr': signal.get('atr'),
                'sector_volatility_multiplier': signal.get('sector_volatility_multiplier'),
                'confidence_modifier': signal.get('confidence_modifier'),
                'stop_pct': signal.get('stop_pct'),
                'target_pct': signal.get('target_pct')
            })

        buy_count = len([p for p in predictions if p['prediction'] == 'BUY'])
        sell_count = len([p for p in predictions if p['prediction'] == 'SELL'])
        hold_count = len([p for p in predictions if p['prediction'] == 'HOLD'])

        response = {
            'timestamp': datetime.now().isoformat(),
            'total': len(predictions),
            'statistics': {
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count
            },
            'predictions': predictions,
            'cached': False,
            'source': 'active_signals',
            'success': True
        }

        print(f"[PREDICTIONS API] Loaded {len(predictions)} predictions from active_signals")
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# /all_live — Real-time inference for all symbols
# ---------------------------------------------------------------------------

@predictions_bp.route('/all_live', methods=['GET'])
def get_all_predictions_live():
    try:
        cached = get_cached_predictions()
        if cached is not None:
            return jsonify(cached)

        init_scanner()

        symbols, metadata_map = load_core_230_symbols()
        predictions = []

        print(f"[PREDICTIONS API] Generating LIVE predictions for {len(symbols)} symbols...")
        start_time = time.time()

        for i, symbol in enumerate(symbols, 1):
            try:
                df = scanner.get_ohlcv_dataframe(symbol, days_back=730)
                if df is None:
                    continue

                current_price = scanner.get_current_price(symbol)
                if current_price is None:
                    continue

                features = scanner.extract_features(df, symbol)
                if features is None:
                    continue

                pred = scanner.get_prediction(symbol, features)

                meta = metadata_map.get(symbol, {})

                predictions.append({
                    'symbol': symbol,
                    'signal': pred['signal'],
                    'confidence': float(pred['confidence']),
                    'current_price': float(current_price),
                    'sector': meta.get('sector', 'unknown'),
                    'market_cap_category': meta.get('market_cap_category', 'unknown')
                })

                if i % 25 == 0:
                    print(f"[PREDICTIONS API] Progress: {i}/{len(symbols)} processed...")

            except Exception as e:
                print(f"[PREDICTIONS API] Error processing {symbol}: {e}")
                continue

        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        buy_count = len([p for p in predictions if p['signal'] == 'BUY'])
        sell_count = len([p for p in predictions if p['signal'] == 'SELL'])
        hold_count = len([p for p in predictions if p['signal'] == 'HOLD'])

        elapsed = time.time() - start_time
        print(f"[PREDICTIONS API] Complete! {len(predictions)} predictions in {elapsed:.1f} seconds")

        result = {
            'success': True,
            'total': len(predictions),
            'statistics': {
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count
            },
            'predictions': predictions,
            'generation_time_seconds': round(elapsed, 1),
            'cached': False
        }

        cache_predictions(result)
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# /symbol/<symbol> — Single-symbol live inference
# ---------------------------------------------------------------------------

@predictions_bp.route('/symbol/<symbol>', methods=['GET'])
def get_symbol_prediction(symbol):
    try:
        init_scanner()
        symbol = symbol.upper()

        df = scanner.get_ohlcv_dataframe(symbol, days_back=730)
        if df is None:
            return jsonify({'error': f'No OHLCV data for {symbol}'}), 404

        current_price = scanner.get_current_price(symbol)
        if current_price is None:
            return jsonify({'error': f'Could not get price for {symbol}'}), 404

        features = scanner.extract_features(df, symbol)
        if features is None:
            return jsonify({'error': f'Could not extract features for {symbol}'}), 404

        pred = scanner.get_prediction(symbol, features)

        _, metadata_map = load_core_230_symbols()
        meta = metadata_map.get(symbol, {})

        return jsonify({
            'success': True,
            'symbol': symbol,
            'signal': pred['signal'],
            'confidence': float(pred['confidence']),
            'current_price': float(current_price),
            'sector': meta.get('sector', 'unknown'),
            'market_cap_category': meta.get('market_cap_category', 'unknown')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Standalone test mode
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    init_scanner()
    print("\nTesting predictions API...")
    test_symbols = ['TMDX', 'NVDA', 'AAPL', 'TSLA']

    for symbol in test_symbols:
        try:
            df = scanner.get_ohlcv_dataframe(symbol, days_back=730)
            if df is None:
                continue
            price = scanner.get_current_price(symbol)
            features = scanner.extract_features(df, symbol)
            pred = scanner.get_prediction(symbol, features)
            print(f"{symbol}: {pred['signal']} @ {pred['confidence']:.1%} (${price:.2f})")
        except Exception as e:
            print(f"{symbol}: Error - {e}")
