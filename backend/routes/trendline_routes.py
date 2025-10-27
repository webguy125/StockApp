"""
Trendline Routes
Handles trendline storage and chart sharing
"""

from flask import Blueprint, request, jsonify
import os
import json
import uuid
from datetime import datetime

trendline_bp = Blueprint('trendline', __name__)


def init_trendline_routes(data_dir):
    """Initialize routes with configuration"""
    @trendline_bp.route("/lines/<symbol>")
    def get_lines(symbol):
        """Get saved trendlines for a symbol"""
        symbol = symbol.upper()
        filename = os.path.join(data_dir, f"lines_{symbol}.json")

        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    lines = json.load(f)
                print(f"Loaded {len(lines)} lines for {symbol}")
                return jsonify(lines)
            except Exception as e:
                print("Error loading lines:", e)
                return jsonify([])
        else:
            print(f"No line file found for {symbol}")
            return jsonify([])

    @trendline_bp.route("/save_line", methods=["POST"])
    def save_line():
        """Save a trendline"""
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        line = data.get("line")

        if not symbol or not line or "id" not in line:
            print("Missing symbol or line ID")
            return jsonify({"error": "Missing symbol or line ID"}), 400

        filename = os.path.join(data_dir, f"lines_{symbol}.json")
        lines = []

        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    lines = json.load(f)
            except Exception as e:
                print("Error reading line file:", e)

        lines = [l for l in lines if l["id"] != line["id"]]
        lines.append(line)

        try:
            with open(filename, "w") as f:
                json.dump(lines, f)
            print(f"Saved line for {symbol}: {line['id']}")
            return jsonify({"success": True})
        except Exception as e:
            print("Error saving line:", e)
            return jsonify({"error": "Failed to save line"}), 500

    @trendline_bp.route("/delete_line", methods=["POST"])
    def delete_line():
        """Delete a trendline"""
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        line_id = data.get("line_id")

        if not symbol or not line_id:
            print("Missing symbol or line ID for deletion")
            return jsonify({"error": "Missing symbol or line ID"}), 400

        filename = os.path.join(data_dir, f"lines_{symbol}.json")
        if not os.path.exists(filename):
            print(f"No line file found for {symbol}")
            return jsonify({"success": True})

        try:
            with open(filename, "r") as f:
                lines = json.load(f)
            lines = [l for l in lines if l["id"] != line_id]
            with open(filename, "w") as f:
                json.dump(lines, f)
            print(f"Deleted line {line_id} for {symbol}")
            return jsonify({"success": True})
        except Exception as e:
            print("Error deleting line:", e)
            return jsonify({"error": "Failed to delete line"}), 500

    @trendline_bp.route("/clear_lines", methods=["POST"])
    def clear_lines():
        """Clear all trendlines for a symbol"""
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        filename = os.path.join(data_dir, f"lines_{symbol}.json")

        try:
            with open(filename, "w") as f:
                json.dump([], f)
            print(f"Cleared all lines for {symbol}")
            return jsonify({"success": True})
        except Exception as e:
            print("Error clearing lines:", e)
            return jsonify({"error": "Failed to clear lines"}), 500

    @trendline_bp.route("/share_chart", methods=["POST"])
    def share_chart():
        """Generate shareable URL for chart configuration"""
        data = request.get_json()

        chart_id = str(uuid.uuid4())
        share_data = {
            'id': chart_id,
            'symbol': data.get('symbol'),
            'period': data.get('period'),
            'interval': data.get('interval'),
            'indicators': data.get('indicators', []),
            'drawings': data.get('drawings', []),
            'theme': data.get('theme', 'light'),
            'created': datetime.now().isoformat()
        }

        # Save shared chart
        share_file = os.path.join(data_dir, f"shared_{chart_id}.json")
        with open(share_file, "w") as f:
            json.dump(share_data, f, indent=2)

        return jsonify({
            'share_id': chart_id,
            'url': f'http://127.0.0.1:5000/?share={chart_id}',
            'success': True
        })

    @trendline_bp.route("/get_shared/<share_id>")
    def get_shared_chart(share_id):
        """Retrieve shared chart configuration"""
        share_file = os.path.join(data_dir, f"shared_{share_id}.json")

        if os.path.exists(share_file):
            with open(share_file, "r") as f:
                share_data = json.load(f)
            return jsonify(share_data)
        else:
            return jsonify({"error": "Shared chart not found"}), 404
