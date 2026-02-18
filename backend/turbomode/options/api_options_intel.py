import os
import sqlite3
from flask import Blueprint, jsonify

bp = Blueprint('options_intel', __name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'Data', 'options_intel.db')


@bp.route('/options_intel/all', methods=['GET'])
def get_all_enriched():
    """
    Get all enriched options signals from cache (options_intel.db).
    This replaces the real-time computation endpoint for fast dashboard loading.

    Returns:
        JSON with success flag, count, and enriched signals data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM enriched_options_signals
            ORDER BY signal_quality_score DESC, model_confidence DESC
        ''')

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return jsonify({
            'success': True,
            'count': len(rows),
            'signals': rows,
            'message': f'Loaded {len(rows)} cached signals'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'signals': []
        }), 500


@bp.route('/options_intel/symbol/<symbol>', methods=['GET'])
def get_enriched_by_symbol(symbol):
    """
    Get enriched options signal for a specific symbol.

    Args:
        symbol: Stock symbol to query

    Returns:
        JSON with signal data for the symbol
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM enriched_options_signals
            WHERE symbol = ?
        ''', (symbol.upper(),))

        row = cursor.fetchone()
        conn.close()

        if row:
            return jsonify({
                'success': True,
                'signal': dict(row)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No cached signal found for {symbol}'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/options_intel/stats', methods=['GET'])
def get_cache_stats():
    """
    Get statistics about the cached options intelligence data.

    Returns:
        JSON with cache statistics
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get total count
        cursor.execute('SELECT COUNT(*) as total FROM enriched_options_signals')
        total = cursor.fetchone()['total']

        # Get oldest and newest timestamps
        cursor.execute('''
            SELECT
                MIN(snapshot_timestamp) as oldest,
                MAX(snapshot_timestamp) as newest
            FROM enriched_options_signals
        ''')
        timestamps = cursor.fetchone()

        # Get count by signal type
        cursor.execute('''
            SELECT base_signal, COUNT(*) as count
            FROM enriched_options_signals
            GROUP BY base_signal
        ''')
        by_signal = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return jsonify({
            'success': True,
            'stats': {
                'total_signals': total,
                'oldest_snapshot': timestamps['oldest'],
                'newest_snapshot': timestamps['newest'],
                'by_signal_type': by_signal
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
