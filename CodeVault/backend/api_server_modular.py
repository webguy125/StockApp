"""
Modular Flask API Server
Stock Trading Application Backend
"""

from flask import Flask
from flask_cors import CORS
import os

# Import middleware
from middleware.caching import setup_cache
from middleware.rate_limiter import setup_rate_limiter
from middleware.auth_middleware import require_auth

# Import route blueprints
from routes.stock_routes import stock_bp, init_stock_routes
from routes.analysis_routes import analysis_bp
from routes.portfolio_routes import portfolio_bp, init_portfolio_routes
from routes.trendline_routes import trendline_bp, init_trendline_routes
from routes.plugin_routes import plugin_bp, init_plugin_routes
from routes.auth_routes import auth_bp, init_auth_routes
from routes.activity_routes import activity_bp, init_activity_routes
from routes.admin_routes import admin_bp, init_admin_routes

# Import plugin manager
from plugins import PluginManager

# Import auth service for users_db
from services.auth_service import load_users

# ============ CONFIGURATION ============
app = Flask(__name__)
CORS(app)

# Application configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
os.makedirs(DATA_DIR, exist_ok=True)

# ============ INITIALIZE EXTENSIONS ============
cache = setup_cache(app)
limiter = setup_rate_limiter(app)
plugin_manager = PluginManager()

# Load users database
users_db = load_users(DATA_DIR)

# Create token required decorator
token_required_decorator = require_auth(app.config['SECRET_KEY'])

# ============ INITIALIZE ROUTE MODULES ============
# Stock routes (includes frontend serving)
init_stock_routes(FRONTEND_DIR)
app.register_blueprint(stock_bp)

# Analysis routes
app.register_blueprint(analysis_bp)

# Portfolio routes
init_portfolio_routes(DATA_DIR)
app.register_blueprint(portfolio_bp)

# Trendline routes
init_trendline_routes(DATA_DIR)
app.register_blueprint(trendline_bp)

# Plugin routes
init_plugin_routes(plugin_manager, cache, limiter)
app.register_blueprint(plugin_bp)

# Auth routes
init_auth_routes(DATA_DIR, app.config['SECRET_KEY'], limiter, token_required_decorator)
app.register_blueprint(auth_bp)

# Activity routes
init_activity_routes(DATA_DIR, token_required_decorator, users_db)
app.register_blueprint(activity_bp)

# Admin routes
init_admin_routes(DATA_DIR, token_required_decorator, users_db, plugin_manager)
app.register_blueprint(admin_bp)

# ============ SERVER START ============
if __name__ == "__main__":
    print("=" * 60)
    print("Stock Trading Application - Modular Backend Server")
    print("=" * 60)
    print(f"Frontend Directory: {FRONTEND_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Plugins Available: {len(plugin_manager.discover_plugins())}")
    print("=" * 60)
    print("Server starting on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True)
