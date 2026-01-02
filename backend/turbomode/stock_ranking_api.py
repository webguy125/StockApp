"""
Stock Ranking API Module
Flask Blueprint for adaptive stock ranking endpoints
Separate from main api_server.py for modularity
"""

from flask import Blueprint, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.turbomode.adaptive_stock_ranker import AdaptiveStockRanker
from backend.turbomode.database_schema import TurboModeDB
import sqlite3

# Create Flask Blueprint
ranking_bp = Blueprint('stock_ranking', __name__, url_prefix='/turbomode/rankings')

# Initialize ranker and scheduler
ranker = AdaptiveStockRanker()
turbomode_db = TurboModeDB()
scheduler = BackgroundScheduler()
scheduler_running = False


def run_monthly_analysis():
    """Background task to run monthly stock ranking analysis"""
    print(f"\n[STOCK RANKER] Starting monthly analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        results = ranker.run_analysis()
        if results:
            print(f"[STOCK RANKER] Analysis complete - Top 10: {[s['symbol'] for s in results['top_10']]}")
        else:
            print("[STOCK RANKER] Analysis failed - no data")
    except Exception as e:
        print(f"[STOCK RANKER] Error during monthly analysis: {e}")
        import traceback
        traceback.print_exc()


@ranking_bp.route('/current', methods=['GET'])
def get_current_rankings():
    """Get current top 10 stock rankings with current signal prices"""
    try:
        rankings = ranker.load_current_rankings()

        if rankings is None:
            return jsonify({
                'error': 'No rankings available yet. Run initial analysis first.',
                'available': False
            }), 404

        # Enrich top 10 stocks with current signal data (entry/target/stop prices)
        top_10_with_prices = []
        for stock in rankings['top_10']:
            symbol = stock['symbol']

            # Get current signals for this symbol
            conn = sqlite3.connect(turbomode_db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT entry_price, entry_min, entry_max, target_price, stop_price, confidence, signal_type
                FROM active_signals
                WHERE symbol = ? AND status = 'ACTIVE'
                ORDER BY confidence DESC
                LIMIT 1
            """, (symbol,))

            signal = cursor.fetchone()
            conn.close()

            # Add signal data to stock
            stock_with_prices = stock.copy()
            if signal:
                stock_with_prices['entry_price'] = signal['entry_price']
                stock_with_prices['entry_min'] = signal['entry_min']
                stock_with_prices['entry_max'] = signal['entry_max']
                stock_with_prices['target_price'] = signal['target_price']
                stock_with_prices['stop_price'] = signal['stop_price']
                stock_with_prices['signal_confidence'] = signal['confidence']
                stock_with_prices['signal_type'] = signal['signal_type']
                stock_with_prices['has_signal'] = True
            else:
                stock_with_prices['has_signal'] = False

            top_10_with_prices.append(stock_with_prices)

        return jsonify({
            'success': True,
            'timestamp': rankings['timestamp'],
            'top_10': top_10_with_prices,
            'available': True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@ranking_bp.route('/all', methods=['GET'])
def get_all_rankings():
    """Get rankings for all stocks"""
    try:
        rankings = ranker.load_current_rankings()

        if rankings is None:
            return jsonify({
                'error': 'No rankings available yet. Run initial analysis first.',
                'available': False
            }), 404

        return jsonify({
            'success': True,
            'timestamp': rankings['timestamp'],
            'all_stocks': rankings['all_stocks'],
            'available': True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@ranking_bp.route('/run', methods=['POST'])
def run_analysis():
    """Manually trigger stock ranking analysis"""
    try:
        print("[STOCK RANKER] Manual analysis triggered")
        results = ranker.run_analysis()

        if results is None:
            return jsonify({
                'success': False,
                'error': 'Analysis failed - no backtest data available'
            }), 500

        return jsonify({
            'success': True,
            'timestamp': results['timestamp'],
            'top_10': results['top_10'],
            'message': 'Analysis complete'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@ranking_bp.route('/scheduler/status', methods=['GET'])
def get_scheduler_status():
    """Get monthly scheduler status"""
    global scheduler_running

    try:
        if not scheduler_running:
            return jsonify({
                'enabled': False,
                'next_run': None,
                'message': 'Scheduler not running'
            })

        jobs = scheduler.get_jobs()
        if len(jobs) == 0:
            return jsonify({
                'enabled': False,
                'next_run': None,
                'message': 'No scheduled jobs'
            })

        next_run = jobs[0].next_run_time
        return jsonify({
            'enabled': True,
            'next_run': next_run.isoformat() if next_run else None,
            'schedule': 'Monthly on 1st at 2 AM'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@ranking_bp.route('/scheduler/start', methods=['POST'])
def start_scheduler():
    """Start monthly scheduler"""
    global scheduler_running

    try:
        if scheduler_running:
            return jsonify({
                'success': False,
                'message': 'Scheduler already running'
            })

        # Schedule monthly analysis on 1st of month at 2 AM
        scheduler.add_job(
            run_monthly_analysis,
            CronTrigger(day=1, hour=2, minute=0),
            id='monthly_stock_ranking',
            replace_existing=True
        )

        if not scheduler.running:
            scheduler.start()

        scheduler_running = True

        print("[STOCK RANKER] Monthly scheduler started")
        print("[STOCK RANKER] Schedule: 1st of each month at 2:00 AM")

        return jsonify({
            'success': True,
            'message': 'Monthly scheduler started',
            'schedule': '1st of each month at 2:00 AM'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@ranking_bp.route('/scheduler/stop', methods=['POST'])
def stop_scheduler():
    """Stop monthly scheduler"""
    global scheduler_running

    try:
        if not scheduler_running:
            return jsonify({
                'success': False,
                'message': 'Scheduler not running'
            })

        scheduler.remove_job('monthly_stock_ranking')
        scheduler_running = False

        print("[STOCK RANKER] Monthly scheduler stopped")

        return jsonify({
            'success': True,
            'message': 'Monthly scheduler stopped'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def init_stock_ranking_scheduler():
    """Initialize and start the monthly scheduler (called from api_server.py)"""
    global scheduler_running

    if scheduler_running:
        print("[STOCK RANKER] Scheduler already initialized")
        return

    try:
        # Schedule monthly analysis on 1st of month at 2 AM
        scheduler.add_job(
            run_monthly_analysis,
            CronTrigger(day=1, hour=2, minute=0),
            id='monthly_stock_ranking',
            replace_existing=True
        )

        if not scheduler.running:
            scheduler.start()

        scheduler_running = True

        print("[STOCK RANKER] Monthly scheduler initialized")
        print("[STOCK RANKER] Schedule: 1st of each month at 2:00 AM")

        # Run initial analysis if no rankings exist
        if not os.path.exists(ranker.rankings_file):
            print("[STOCK RANKER] No existing rankings found - running initial analysis...")
            run_monthly_analysis()

    except Exception as e:
        print(f"[STOCK RANKER] Failed to initialize scheduler: {e}")
        import traceback
        traceback.print_exc()
