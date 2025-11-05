"""
ORD Volume Backend Server
Completely segregated Flask endpoints for ORD Volume data persistence

NO SHARED CODE - Standalone implementation with own routes and storage
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Create segregated Flask app for ORD Volume
ord_volume_app = Flask(__name__)
CORS(ord_volume_app)

# Segregated data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ord-volume')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def get_ord_volume_file_path(symbol):
    """
    Get file path for ORD Volume analysis data

    Args:
        symbol (str): Trading symbol

    Returns:
        str: Full file path
    """
    filename = f"ord_volume_{symbol}.json"
    return os.path.join(DATA_DIR, filename)


@ord_volume_app.route('/ord-volume/save', methods=['POST'])
def save_ord_volume():
    """
    Save ORD Volume analysis for a symbol

    Request body:
        {
            "symbol": "BTC-USD",
            "analysis": {
                "mode": "auto",
                "trendlines": [...],
                "labels": [...],
                "strength": "Strong",
                "color": "green",
                "waveData": [...],
                "lines": [...]
            }
        }

    Returns:
        JSON response with success status
    """
    try:
        data = request.get_json()

        if not data or 'symbol' not in data or 'analysis' not in data:
            return jsonify({'error': 'Missing required fields: symbol, analysis'}), 400

        symbol = data['symbol']
        analysis = data['analysis']

        # Get file path
        file_path = get_ord_volume_file_path(symbol)

        # Save to file
        with open(file_path, 'w') as f:
            json.dump({
                'symbol': symbol,
                'analysis': analysis
            }, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'ORD Volume analysis saved for {symbol}'
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@ord_volume_app.route('/ord-volume/load/<symbol>', methods=['GET'])
def load_ord_volume(symbol):
    """
    Load ORD Volume analysis for a symbol

    Args:
        symbol (str): Trading symbol (URL parameter)

    Returns:
        JSON response with analysis data or 404 if not found
    """
    try:
        file_path = get_ord_volume_file_path(symbol)

        if not os.path.exists(file_path):
            return jsonify({
                'error': f'No ORD Volume analysis found for {symbol}'
            }), 404

        with open(file_path, 'r') as f:
            data = json.load(f)

        return jsonify(data), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@ord_volume_app.route('/ord-volume/delete/<symbol>', methods=['DELETE'])
def delete_ord_volume(symbol):
    """
    Delete ORD Volume analysis for a symbol

    Args:
        symbol (str): Trading symbol (URL parameter)

    Returns:
        JSON response with success status
    """
    try:
        file_path = get_ord_volume_file_path(symbol)

        if not os.path.exists(file_path):
            return jsonify({
                'error': f'No ORD Volume analysis found for {symbol}'
            }), 404

        os.remove(file_path)

        return jsonify({
            'success': True,
            'message': f'ORD Volume analysis deleted for {symbol}'
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@ord_volume_app.route('/ord-volume/list', methods=['GET'])
def list_ord_volume():
    """
    List all symbols with saved ORD Volume analyses

    Returns:
        JSON response with list of symbols
    """
    try:
        # Get all ORD Volume files
        files = os.listdir(DATA_DIR)

        symbols = []
        for filename in files:
            if filename.startswith('ord_volume_') and filename.endswith('.json'):
                # Extract symbol from filename
                symbol = filename.replace('ord_volume_', '').replace('.json', '')
                symbols.append(symbol)

        return jsonify({
            'symbols': symbols,
            'count': len(symbols)
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@ord_volume_app.route('/ord-volume/health', methods=['GET'])
def health_check():
    """
    Health check endpoint

    Returns:
        JSON response with server status
    """
    return jsonify({
        'status': 'healthy',
        'service': 'ORD Volume Server',
        'data_dir': DATA_DIR
    }), 200


if __name__ == '__main__':
    # Run standalone server on different port
    print("Starting ORD Volume Server on http://127.0.0.1:5001")
    print(f"Data directory: {DATA_DIR}")
    ord_volume_app.run(
        host='127.0.0.1',
        port=5001,
        debug=True
    )
