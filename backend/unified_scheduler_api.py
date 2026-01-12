"""
Flask API Extension for Unified Scheduler
Provides REST endpoints for scheduler control and manual task execution

All endpoints follow /scheduler/* pattern
"""

import logging
from flask import jsonify, request
from backend.unified_scheduler import (
    start_unified_scheduler,
    stop_unified_scheduler,
    get_scheduler_status,
    run_task_manually,
    TASK_FUNCTIONS
)

logger = logging.getLogger('unified_scheduler_api')


def init_unified_scheduler_api(app):
    """
    Initialize unified scheduler API endpoints in Flask app

    Args:
        app: Flask application instance

    Returns:
        Flask app with scheduler endpoints registered
    """

    # ========================================================================
    # SCHEDULER CONTROL ENDPOINTS
    # ========================================================================

    @app.route('/scheduler/status', methods=['GET'])
    def scheduler_status():
        """
        GET /scheduler/status

        Get current scheduler status including all jobs and their next run times

        Returns:
            JSON with scheduler state
        """
        try:
            status = get_scheduler_status()
            return jsonify(status), 200
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/start', methods=['POST'])
    def scheduler_start():
        """
        POST /scheduler/start

        Start the unified scheduler (if not already running)

        Returns:
            JSON with success status
        """
        try:
            success = start_unified_scheduler()

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Unified scheduler started'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Scheduler already running'
                }), 200
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/stop', methods=['POST'])
    def scheduler_stop():
        """
        POST /scheduler/stop

        Stop the unified scheduler

        Returns:
            JSON with success status
        """
        try:
            success = stop_unified_scheduler()

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Unified scheduler stopped'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Scheduler not running'
                }), 200
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # ========================================================================
    # MANUAL TASK EXECUTION ENDPOINTS
    # ========================================================================

    @app.route('/scheduler/run_ingestion', methods=['POST'])
    def manual_run_ingestion():
        """
        POST /scheduler/run_ingestion

        Manually trigger Task 1: Master Market Data Ingestion

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=1)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running ingestion: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_orchestrator', methods=['POST'])
    def manual_run_orchestrator():
        """
        POST /scheduler/run_orchestrator

        Manually trigger Task 2: TurboMode Training Orchestrator

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=2)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running orchestrator: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_overnight_scanner', methods=['POST'])
    def manual_run_overnight_scanner():
        """
        POST /scheduler/run_overnight_scanner

        Manually trigger Task 3: Overnight Scanner

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=3)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running overnight scanner: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_backtest_generator', methods=['POST'])
    def manual_run_backtest_generator():
        """
        POST /scheduler/run_backtest_generator

        Manually trigger Task 4: Backtest Data Generator

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=4)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running backtest generator: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_drift_monitor', methods=['POST'])
    def manual_run_drift_monitor():
        """
        POST /scheduler/run_drift_monitor

        Manually trigger Task 5: Drift Monitoring System

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=5)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running drift monitor: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_weekly_maintenance', methods=['POST'])
    def manual_run_weekly_maintenance():
        """
        POST /scheduler/run_weekly_maintenance

        Manually trigger Task 6: Weekly Maintenance

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=6)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running weekly maintenance: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_task/<int:task_id>', methods=['POST'])
    def manual_run_task_by_id(task_id):
        """
        POST /scheduler/run_task/<task_id>

        Manually trigger any task by ID

        Args:
            task_id: Task ID (1-6)

        Returns:
            JSON with task results
        """
        try:
            if task_id not in TASK_FUNCTIONS:
                return jsonify({
                    'success': False,
                    'error': f'Invalid task_id: {task_id}. Valid IDs: 1-6'
                }), 400

            result = run_task_manually(task_id=task_id)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running task {task_id}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # ========================================================================
    # SCHEDULER AUTO-START ON FLASK INIT
    # ========================================================================

    logger.info("=" * 80)
    logger.info("UNIFIED SCHEDULER - INITIALIZING")
    logger.info("=" * 80)

    try:
        # Auto-start scheduler when Flask starts
        start_unified_scheduler()
        logger.info("[OK] Unified scheduler started automatically")
    except Exception as e:
        logger.error(f"[ERROR] Failed to auto-start scheduler: {e}")

    logger.info("=" * 80)

    return app
