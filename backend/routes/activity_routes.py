"""
Activity Routes
Handles activity logging
"""

from flask import Blueprint, request, jsonify
import os
import json
import uuid
from datetime import datetime

activity_bp = Blueprint('activity', __name__)


def init_activity_routes(data_dir, token_required_decorator, users_db):
    """Initialize routes with configuration"""
    activity_log_file = os.path.join(data_dir, "activity_log.json")

    def log_activity(username, action, details=None):
        """Log user activity"""
        if os.path.exists(activity_log_file):
            with open(activity_log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "action": action,
            "details": details or {}
        }

        logs.append(log_entry)

        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]

        with open(activity_log_file, "w") as f:
            json.dump(logs, f, indent=2)

    @activity_bp.route("/activity/log", methods=["POST"])
    @token_required_decorator
    def create_activity_log(current_user):
        """Create an activity log entry"""
        data = request.get_json()
        action = data.get("action")
        details = data.get("details", {})

        log_activity(current_user, action, details)

        return jsonify({"success": True})

    @activity_bp.route("/activity/logs", methods=["GET"])
    @token_required_decorator
    def get_activity_logs(current_user):
        """Get activity logs (admin only for all users, regular users see their own)"""
        user = users_db.get(current_user)

        if os.path.exists(activity_log_file):
            with open(activity_log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        # Filter logs based on user role
        if user.get("role") != "admin":
            logs = [log for log in logs if log.get("username") == current_user]

        # Pagination
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        return jsonify({
            "logs": logs[offset:offset+limit],
            "total": len(logs)
        })
