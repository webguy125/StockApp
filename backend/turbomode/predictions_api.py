"""
Predictions API Module
Flask Blueprint for viewing all model predictions with confidence levels
Shows complete transparency into model predictions
"""

from flask import Blueprint, jsonify
import os
import sys
import time
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.turbomode.overnight_scanner import OvernightScanner
from backend.turbomode.core_symbols import get_all_core_symbols, get_symbol_metadata

# Create Flask Blueprint
predictions_bp = Blueprint('predictions', __name__, url_prefix='/turbomode/predictions')

# Initialize scanner
scanner = None

# Cache for predictions (refreshes every 5 minutes)
predictions_cache = {
    'data': None,
    'timestamp': None,
    'expiry_minutes': 5
}

def get_cached_predictions():
    """Get predictions from cache if available and not expired"""
    if predictions_cache['data'] is None or predictions_cache['timestamp'] is None:
        return None

    elapsed = (datetime.now() - predictions_cache['timestamp']).total_seconds() / 60
    if elapsed < predictions_cache['expiry_minutes']:
        print(f"[PREDICTIONS API] Using cached data ({elapsed:.1f} minutes old)")
        return predictions_cache['data']

    print(f"[PREDICTIONS API] Cache expired ({elapsed:.1f} minutes old), refreshing...")
    return None

def cache_predictions(data):
    """Cache predictions data"""
    predictions_cache['data'] = data
    predictions_cache['timestamp'] = datetime.now()
    print(f"[PREDICTIONS API] Cached {len(data['predictions'])} predictions")


def init_scanner():
    """Initialize scanner with models loaded"""
    global scanner
    if scanner is None:
        print("[PREDICTIONS API] Initializing overnight scanner...")
        scanner = OvernightScanner()
        scanner._load_models()
        print("[PREDICTIONS API] Scanner ready")


@predictions_bp.route('/all', methods=['GET'])
def get_all_predictions():
    """
    FAST VERSION: Get predictions from pre-generated file

    The overnight scanner saves all 80 predictions to all_predictions.json
    This endpoint just reads that file - loads in <1 second!

    Returns:
        JSON with predictions array containing:
        - symbol: Stock ticker
        - prediction: 'buy', 'sell', or 'hold'
        - confidence: Confidence level (0.0 - 1.0)
        - current_price: Current stock price
        - sector: Sector classification
        - market_cap_category: 'large_cap', 'mid_cap', or 'small_cap'
    """
    import json

    try:
        # Read from pre-generated predictions file (scanner saves to turbomode/data/)
        predictions_file = os.path.join(os.path.dirname(__file__), 'data', 'all_predictions.json')

        if not os.path.exists(predictions_file):
            return jsonify({
                'error': 'Predictions file not found. Run the overnight scanner first.',
                'help': 'Run: python backend/turbomode/overnight_scanner.py'
            }), 404

        print(f"[PREDICTIONS API] Reading from {predictions_file}")
        with open(predictions_file, 'r') as f:
            data = json.load(f)

        # Add cache indicator and total count
        data['cached'] = True
        data['source'] = 'pre_generated_file'
        data['success'] = True

        # Calculate total if not present
        if 'total' not in data and 'predictions' in data:
            data['total'] = len(data['predictions'])

        print(f"[PREDICTIONS API] Loaded {data.get('total', len(data.get('predictions', [])))} predictions (generated at {data.get('timestamp', 'unknown')})")

        return jsonify(data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@predictions_bp.route('/all_live', methods=['GET'])
def get_all_predictions_live():
    """
    SLOW VERSION: Generate predictions in real-time (takes 2-3 minutes)
    Only use this for testing or when you need live data
    """
    try:
        # Check cache first
        cached_data = get_cached_predictions()
        if cached_data is not None:
            return jsonify(cached_data)

        # Initialize scanner if needed
        init_scanner()

        print("[PREDICTIONS API] Generating LIVE predictions for all 80 stocks (may take 2-3 minutes)...")
        start_time = time.time()

        # Get all curated symbols
        symbols = get_all_core_symbols()

        predictions = []

        for i, symbol in enumerate(symbols, 1):
            try:
                # Get current price
                current_price = scanner.get_current_price(symbol)
                if current_price is None:
                    print(f"[PREDICTIONS API] Could not get price for {symbol}, skipping")
                    continue

                # Extract features
                features = scanner.extract_features(symbol)
                if features is None:
                    print(f"[PREDICTIONS API] Could not extract features for {symbol}, skipping")
                    continue

                # Get prediction
                pred = scanner.get_prediction(features)

                # Get metadata
                metadata = get_symbol_metadata(symbol)

                # Build prediction object
                prediction_obj = {
                    'symbol': symbol,
                    'prediction': pred['prediction'],
                    'confidence': float(pred['confidence']),
                    'current_price': float(current_price),
                    'sector': metadata.get('sector', 'unknown'),
                    'market_cap_category': metadata.get('market_cap_category', 'unknown')
                }

                predictions.append(prediction_obj)

                # Progress indicator
                if i % 10 == 0:
                    print(f"[PREDICTIONS API] Progress: {i}/{len(symbols)} stocks processed...")

            except Exception as e:
                print(f"[PREDICTIONS API] Error processing {symbol}: {e}")
                continue

        # Sort by confidence descending
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # Calculate statistics
        buy_count = len([p for p in predictions if p['prediction'] == 'buy'])
        sell_count = len([p for p in predictions if p['prediction'] == 'sell'])
        hold_count = len([p for p in predictions if p['prediction'] == 'hold'])
        threshold_count = len([p for p in predictions if p['confidence'] >= 0.65])

        elapsed = time.time() - start_time
        print(f"[PREDICTIONS API] Complete! Processed {len(predictions)} stocks in {elapsed:.1f} seconds")

        result = {
            'success': True,
            'total': len(predictions),
            'statistics': {
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'threshold_met': threshold_count,
                'threshold_not_met': len(predictions) - threshold_count
            },
            'predictions': predictions,
            'generation_time_seconds': round(elapsed, 1),
            'cached': False
        }

        # Cache the result
        cache_predictions(result)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@predictions_bp.route('/symbol/<symbol>', methods=['GET'])
def get_symbol_prediction(symbol):
    """
    Get prediction for a single symbol

    Args:
        symbol: Stock ticker

    Returns:
        JSON with prediction details
    """
    try:
        # Initialize scanner if needed
        init_scanner()

        # Convert to uppercase
        symbol = symbol.upper()

        # Get current price
        current_price = scanner.get_current_price(symbol)
        if current_price is None:
            return jsonify({'error': f'Could not get price for {symbol}'}), 404

        # Extract features
        features = scanner.extract_features(symbol)
        if features is None:
            return jsonify({'error': f'Could not extract features for {symbol}'}), 404

        # Get prediction
        pred = scanner.get_prediction(features)

        # Get metadata
        metadata = get_symbol_metadata(symbol)

        return jsonify({
            'success': True,
            'symbol': symbol,
            'prediction': pred['prediction'],
            'confidence': float(pred['confidence']),
            'current_price': float(current_price),
            'sector': metadata.get('sector', 'unknown'),
            'market_cap_category': metadata.get('market_cap_category', 'unknown'),
            'meets_threshold': pred['confidence'] >= 0.65
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Test the API
    init_scanner()
    print("\nTesting predictions API...")

    # Test a few symbols
    test_symbols = ['TMDX', 'NVDA', 'AAPL', 'TSLA']

    for symbol in test_symbols:
        try:
            price = scanner.get_current_price(symbol)
            features = scanner.extract_features(symbol)
            if features:
                pred = scanner.get_prediction(features)
                print(f"{symbol}: {pred['prediction'].upper()} @ {pred['confidence']:.1%} (${price:.2f})")
        except Exception as e:
            print(f"{symbol}: Error - {e}")
