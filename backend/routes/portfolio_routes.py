"""
Portfolio Routes
Handles portfolio management
"""

from flask import Blueprint, request, jsonify
from services.portfolio_service import (
    get_portfolio_with_prices, buy_stock, sell_stock, clear_portfolio
)

portfolio_bp = Blueprint('portfolio', __name__)


def init_portfolio_routes(data_dir):
    """Initialize routes with configuration"""
    @portfolio_bp.route("/portfolio", methods=["GET", "POST", "DELETE"])
    def manage_portfolio():
        """Portfolio tracking and management"""
        if request.method == "GET":
            # Get portfolio
            portfolio = get_portfolio_with_prices(data_dir)
            return jsonify(portfolio)

        elif request.method == "POST":
            # Add/modify position
            data = request.get_json()
            action = data.get("action")  # "buy" or "sell"
            symbol = data["symbol"].upper()
            shares = data["shares"]
            price = data.get("price")

            try:
                if action == "buy":
                    portfolio = buy_stock(data_dir, symbol, shares, price)
                    return jsonify({"success": True, "portfolio": portfolio})
                elif action == "sell":
                    portfolio = sell_stock(data_dir, symbol, shares, price)
                    return jsonify({"success": True, "portfolio": portfolio})
                else:
                    return jsonify({"error": "Invalid action"}), 400
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

        elif request.method == "DELETE":
            # Clear portfolio
            portfolio = clear_portfolio(data_dir)
            return jsonify({"success": True})
