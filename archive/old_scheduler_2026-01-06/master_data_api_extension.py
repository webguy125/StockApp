"""
Master Market Data API Extension for Flask
Adds endpoints for managing nightly data ingestion

To integrate into api_server.py, add:
    from backend.master_data_api_extension import init_master_data_api

Then in __name__ == '__main__' section:
    init_master_data_api(app)
"""

import sys
import os

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import jsonify
from master_market_data.master_data_scheduler import (
    init_master_data_scheduler,
    start_scheduler as start_master_data_scheduler,
    stop_scheduler as stop_master_data_scheduler,
    get_status as get_master_data_status,
    run_nightly_ingestion
)

MASTER_DATA_SCHEDULER_AVAILABLE = True


def init_master_data_api(app):
    """
    Initialize Master Market Data API endpoints

    Args:
        app: Flask application instance
    """

    @app.route('/master_data/scheduler/status', methods=['GET'])
    def get_master_data_scheduler_status():
        """Get Master Data scheduler status"""
        if not MASTER_DATA_SCHEDULER_AVAILABLE:
            return jsonify({'error': 'Master Data scheduler not available'}), 503

        try:
            status = get_master_data_status()
            return jsonify(status)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/master_data/scheduler/start', methods=['POST'])
    def start_master_data_scheduler_endpoint():
        """Start Master Data scheduler"""
        if not MASTER_DATA_SCHEDULER_AVAILABLE:
            return jsonify({'error': 'Master Data scheduler not available'}), 503

        try:
            success = start_master_data_scheduler(schedule_time='22:45')

            if success:
                status = get_master_data_status()
                return jsonify({
                    'success': True,
                    'message': 'Master Data scheduler started',
                    'next_run': status['next_run']
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Scheduler already running or failed to start'
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/master_data/scheduler/stop', methods=['POST'])
    def stop_master_data_scheduler_endpoint():
        """Stop Master Data scheduler"""
        if not MASTER_DATA_SCHEDULER_AVAILABLE:
            return jsonify({'error': 'Master Data scheduler not available'}), 503

        try:
            success = stop_master_data_scheduler()

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Master Data scheduler stopped'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Scheduler not running or failed to stop'
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/master_data/ingest/manual', methods=['POST'])
    def manual_master_data_ingestion():
        """Manually trigger Master Data ingestion"""
        if not MASTER_DATA_SCHEDULER_AVAILABLE:
            return jsonify({'error': 'Master Data scheduler not available'}), 503

        try:
            # Run ingestion in background
            import threading
            thread = threading.Thread(target=run_nightly_ingestion)
            thread.daemon = True
            thread.start()

            return jsonify({
                'success': True,
                'message': 'Master Data ingestion started in background'
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    # Initialize scheduler on startup
    print("[MASTER DATA SCHEDULER] Initializing nightly ingestion scheduler...")
    init_master_data_scheduler()
    status = get_master_data_status()
    if status['enabled']:
        print(f"[MASTER DATA SCHEDULER] Ready - Next ingestion at {status['next_run']}")
    else:
        print("[MASTER DATA SCHEDULER] Disabled")

    return app
