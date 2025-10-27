"""
Admin Routes
Handles admin statistics and monitoring
"""

from flask import Blueprint, request, jsonify
import os
import json
from datetime import datetime

admin_bp = Blueprint('admin', __name__)


def init_admin_routes(data_dir, token_required_decorator, users_db, plugin_manager):
    """Initialize routes with configuration"""
    activity_log_file = os.path.join(data_dir, "activity_log.json")

    @admin_bp.route("/admin/stats", methods=["GET"])
    @token_required_decorator
    def get_admin_stats(current_user):
        """Get system statistics (admin only)"""
        user = users_db.get(current_user)

        if user.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403

        # Calculate stats
        total_users = len(users_db)

        # Count activity logs
        if os.path.exists(activity_log_file):
            with open(activity_log_file, "r") as f:
                logs = json.load(f)
            total_actions = len(logs)

            # Actions in last 24 hours
            now = datetime.now()
            recent_actions = sum(1 for log in logs
                               if (now - datetime.fromisoformat(log["timestamp"])).days < 1)
        else:
            total_actions = 0
            recent_actions = 0

        # Count portfolios
        portfolio_file = os.path.join(data_dir, "portfolio.json")
        has_portfolio = os.path.exists(portfolio_file)

        # Count shared charts
        shared_charts = len([f for f in os.listdir(data_dir) if f.startswith("shared_")])

        # Plugin count
        plugin_count = len(plugin_manager.discover_plugins())

        return jsonify({
            "users": {
                "total": total_users,
                "active_today": recent_actions  # Simplified metric
            },
            "activity": {
                "total_actions": total_actions,
                "actions_24h": recent_actions
            },
            "content": {
                "shared_charts": shared_charts,
                "plugins": plugin_count
            },
            "system": {
                "cache_enabled": True,
                "rate_limiting": True
            }
        })
