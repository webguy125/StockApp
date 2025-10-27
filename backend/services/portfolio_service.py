"""
Portfolio Service Module
Portfolio management and position tracking
"""

import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime


def load_portfolio(data_dir):
    """Load portfolio from file"""
    portfolio_file = os.path.join(data_dir, "portfolio.json")

    if os.path.exists(portfolio_file):
        with open(portfolio_file, "r") as f:
            return json.load(f)
    else:
        return {"positions": [], "cash": 100000.0}


def save_portfolio(data_dir, portfolio):
    """Save portfolio to file"""
    portfolio_file = os.path.join(data_dir, "portfolio.json")

    with open(portfolio_file, "w") as f:
        json.dump(portfolio, f, indent=2)


def get_portfolio_with_prices(data_dir):
    """Get portfolio with current prices and P&L"""
    portfolio = load_portfolio(data_dir)

    # Calculate current values
    total_value = portfolio["cash"]

    for position in portfolio["positions"]:
        current_data = yf.download(position["symbol"], period="1d")
        if not current_data.empty:
            if isinstance(current_data.columns, pd.MultiIndex):
                current_data.columns = current_data.columns.get_level_values(0)
            current_price = current_data['Close'].iloc[-1]
            position['current_price'] = float(current_price)
            position['current_value'] = float(current_price * position['shares'])
            position['pnl'] = float((current_price - position['avg_cost']) * position['shares'])
            position['pnl_pct'] = float((current_price - position['avg_cost']) / position['avg_cost'] * 100)
            total_value += position['current_value']

    portfolio['total_value'] = float(total_value)
    portfolio['total_pnl'] = float(total_value - 100000.0)

    return portfolio


def buy_stock(data_dir, symbol, shares, price=None):
    """Buy stock and add to portfolio"""
    portfolio = load_portfolio(data_dir)

    # Get current price if not provided
    if not price:
        stock_data = yf.download(symbol, period="1d")
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        price = stock_data['Close'].iloc[-1]

    cost = price * shares

    if cost > portfolio["cash"]:
        raise ValueError("Insufficient cash")

    # Find existing position
    existing = next((p for p in portfolio["positions"] if p["symbol"] == symbol), None)

    if existing:
        # Update average cost
        total_cost = (existing['avg_cost'] * existing['shares']) + cost
        existing['shares'] += shares
        existing['avg_cost'] = total_cost / existing['shares']
    else:
        portfolio["positions"].append({
            "symbol": symbol,
            "shares": shares,
            "avg_cost": float(price),
            "date_added": datetime.now().strftime('%Y-%m-%d')
        })

    portfolio["cash"] -= cost

    save_portfolio(data_dir, portfolio)

    return portfolio


def sell_stock(data_dir, symbol, shares, price=None):
    """Sell stock and remove from portfolio"""
    portfolio = load_portfolio(data_dir)

    # Get current price if not provided
    if not price:
        stock_data = yf.download(symbol, period="1d")
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        price = stock_data['Close'].iloc[-1]

    existing = next((p for p in portfolio["positions"] if p["symbol"] == symbol), None)

    if not existing or existing['shares'] < shares:
        raise ValueError("Insufficient shares")

    cost = price * shares
    existing['shares'] -= shares
    portfolio["cash"] += cost

    if existing['shares'] == 0:
        portfolio["positions"] = [p for p in portfolio["positions"] if p["symbol"] != symbol]

    save_portfolio(data_dir, portfolio)

    return portfolio


def clear_portfolio(data_dir):
    """Clear all positions and reset portfolio"""
    portfolio = {"positions": [], "cash": 100000.0}
    save_portfolio(data_dir, portfolio)
    return portfolio
