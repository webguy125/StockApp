"""
Plugin Routes
Handles custom indicator plugins
"""

from flask import Blueprint, request, jsonify
import yfinance as yf
import pandas as pd

plugin_bp = Blueprint('plugin', __name__)


def init_plugin_routes(plugin_manager, cache, limiter):
    """Initialize routes with configuration"""
    @plugin_bp.route("/plugins", methods=["GET"])
    @limiter.limit("100 per hour")
    def list_plugins():
        """List all available custom indicator plugins"""
        plugin_names = plugin_manager.discover_plugins()
        plugins_info = []

        for name in plugin_names:
            plugin = plugin_manager.get_plugin(name)
            if plugin:
                plugins_info.append(plugin.get_info())

        return jsonify({
            'plugins': plugins_info,
            'total': len(plugins_info)
        })

    @plugin_bp.route("/plugins/<plugin_name>", methods=["GET"])
    @limiter.limit("100 per hour")
    def get_plugin_info(plugin_name):
        """Get detailed information about a specific plugin"""
        plugin = plugin_manager.get_plugin(plugin_name)

        if not plugin:
            return jsonify({"error": "Plugin not found"}), 404

        return jsonify(plugin.get_info())

    @plugin_bp.route("/plugins/execute", methods=["POST"])
    @cache.cached(timeout=300, query_string=True)
    @limiter.limit("50 per hour")
    def execute_plugin():
        """Execute a custom plugin on stock data"""
        data = request.get_json()
        plugin_name = data.get("plugin")
        symbol = data.get("symbol")
        period = data.get("period", "1y")
        interval = data.get("interval", "1d")
        params = data.get("params", {})

        if not plugin_name or not symbol:
            return jsonify({"error": "Plugin name and symbol are required"}), 400

        # Get stock data
        df = yf.download(symbol, period=period, interval=interval)

        if df.empty:
            return jsonify({"error": "No data found for symbol"}), 404

        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)

        # Execute plugin
        try:
            result = plugin_manager.execute_plugin(plugin_name, df, params)

            if result is None:
                return jsonify({"error": "Plugin not found or execution failed"}), 404

            # Prepare dates for response
            date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            dates = df[date_col].dt.strftime('%Y-%m-%d' if 'd' in interval else '%Y-%m-%d %H:%M:%S').tolist()

            return jsonify({
                'plugin': plugin_name,
                'symbol': symbol,
                'result': result if isinstance(result, list) else result,
                'dates': dates
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
