# Stock Trading Application - Modular Backend

## Overview

This is a fully refactored, modular Flask backend for the Stock Trading Application. The codebase has been restructured from a monolithic 1,289-line file into a clean, maintainable architecture following best practices.

## Quick Start

### Run the Modular Server
```bash
cd C:\StockApp\backend
python api_server_modular.py
```

### Expected Output
```
============================================================
Stock Trading Application - Modular Backend Server
============================================================
Frontend Directory: C:\StockApp\backend\..\frontend
Data Directory: C:\StockApp\backend\data
Plugins Available: 2
============================================================
Server starting on http://127.0.0.1:5000
============================================================
 * Serving Flask app 'api_server_modular'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

## Architecture

### Directory Structure
```
backend/
├── api_server_modular.py       # Main entry point (102 lines)
├── api_server.py               # Original monolithic version (preserved)
│
├── services/                   # Business Logic Layer
│   ├── stock_service.py        # Stock data fetching from Yahoo Finance
│   ├── indicator_service.py    # Technical indicators (SMA, EMA, RSI, MACD, BB, VWAP)
│   ├── pattern_service.py      # Chart patterns (Double Top/Bottom, H&S)
│   ├── prediction_service.py   # ML price predictions (Linear Regression)
│   ├── trade_service.py        # Trade idea generation & strategies
│   ├── portfolio_service.py    # Portfolio management & tracking
│   └── auth_service.py         # Authentication & user management
│
├── routes/                     # API Endpoints (Flask Blueprints)
│   ├── stock_routes.py         # Stock data, volume, indicators
│   ├── analysis_routes.py      # Patterns, volume profile, comparison, prediction
│   ├── portfolio_routes.py     # Portfolio CRUD operations
│   ├── trendline_routes.py     # Trendline storage & chart sharing
│   ├── plugin_routes.py        # Custom indicator plugins
│   ├── auth_routes.py          # User registration, login, profile
│   ├── activity_routes.py      # Activity logging
│   └── admin_routes.py         # Admin statistics & monitoring
│
├── middleware/                 # Cross-Cutting Concerns
│   ├── auth_middleware.py      # JWT token verification decorator
│   ├── rate_limiter.py         # Rate limiting configuration
│   └── caching.py              # Cache setup (filesystem, 5min timeout)
│
├── models/                     # Data Models (ready for future ORM)
│   └── __init__.py
│
├── plugins/                    # Plugin System
│   ├── __init__.py             # PluginManager
│   ├── base_plugin.py          # BasePlugin class
│   ├── wma_plugin.py           # Weighted Moving Average
│   └── atr_plugin.py           # Average True Range
│
└── data/                       # Runtime data storage
    ├── portfolio.json          # User portfolios
    ├── users.json              # User accounts
    ├── activity_log.json       # Activity logs
    ├── lines_*.json            # Trendlines per symbol
    └── shared_*.json           # Shared chart configurations
```

## Features

### API Endpoints

#### Stock Data & Indicators
- `GET /data/<symbol>` - Fetch OHLC data
- `POST /volume` - Calculate average volume
- `POST /indicators` - Calculate technical indicators (SMA, EMA, RSI, MACD, BB, VWAP)

#### Analysis
- `POST /patterns` - Detect chart patterns
- `POST /volume_profile` - Volume by price analysis
- `POST /compare` - Compare multiple symbols
- `POST /predict` - ML-based price predictions
- `POST /trade_ideas` - Auto-generate trade ideas

#### Portfolio
- `GET /portfolio` - View portfolio with P&L
- `POST /portfolio` - Buy/sell stocks
- `DELETE /portfolio` - Reset portfolio

#### Trendlines & Sharing
- `GET /lines/<symbol>` - Get saved trendlines
- `POST /save_line` - Save trendline
- `POST /delete_line` - Delete trendline
- `POST /clear_lines` - Clear all trendlines
- `POST /share_chart` - Generate shareable chart URL
- `GET /get_shared/<id>` - Load shared chart

#### Plugins
- `GET /plugins` - List available plugins
- `GET /plugins/<name>` - Get plugin details
- `POST /plugins/execute` - Execute custom plugin

#### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get JWT token
- `GET /auth/profile` - Get user profile (protected)

#### Activity & Admin
- `POST /activity/log` - Log user activity (protected)
- `GET /activity/logs` - View activity logs (protected)
- `GET /admin/stats` - System statistics (admin only)

### Technical Features

#### Security
- JWT-based authentication
- Password hashing (SHA256)
- Rate limiting (200/hour, 50/minute)
- Token expiration (24 hours)
- Protected routes with `@require_auth` decorator

#### Performance
- Filesystem caching (5-minute default)
- Query-based cache invalidation
- Efficient data structures
- Lazy loading where applicable

#### Extensibility
- Plugin system for custom indicators
- Blueprint-based routing for easy additions
- Service layer for business logic reuse
- Middleware for cross-cutting concerns

## Code Metrics

### Before Refactoring
- **Files:** 1 monolithic file
- **Lines:** 1,289 lines in `api_server.py`
- **Functions:** 30+ mixed concerns
- **Testability:** Difficult
- **Maintainability:** Low

### After Refactoring
- **Files:** 24 modular files
- **Lines in main:** 102 lines in `api_server_modular.py`
- **Total lines:** ~3,095 (including comments and docstrings)
- **Separation:** Services / Routes / Middleware
- **Testability:** Easy (unit test each service)
- **Maintainability:** High

### File Breakdown
```
Services:     7 files  (~800 lines)
Routes:       8 files  (~1200 lines)
Middleware:   3 files  (~100 lines)
Main App:     1 file   (~100 lines)
```

## Dependencies

### Required Packages
```
Flask==3.1.2
Flask-CORS==5.0.0
Flask-Caching==2.3.1
Flask-Limiter==4.0.0
PyJWT==2.10.1
pandas==2.2.3
numpy==2.2.6
yfinance==0.2.51
scipy==1.15.3
scikit-learn==1.7.2
```

### Install All Dependencies
```bash
pip install Flask Flask-CORS Flask-Caching Flask-Limiter PyJWT pandas numpy yfinance scipy scikit-learn
```

## Usage Examples

### 1. Fetch Stock Data
```python
import requests

response = requests.get('http://localhost:5000/data/AAPL?period=1mo')
data = response.json()
print(f"Fetched {len(data)} candles")
```

### 2. Calculate Indicators
```python
payload = {
    "symbol": "AAPL",
    "period": "1mo",
    "indicators": [
        {"type": "SMA", "params": {"period": 20}},
        {"type": "RSI", "params": {"period": 14}}
    ]
}
response = requests.post('http://localhost:5000/indicators', json=payload)
indicators = response.json()
```

### 3. Generate Trade Ideas
```python
payload = {"symbol": "AAPL", "period": "3mo"}
response = requests.post('http://localhost:5000/trade_ideas', json=payload)
ideas = response.json()
print(f"Generated {ideas['total_ideas']} trade ideas")
```

### 4. User Authentication
```python
# Register
register_data = {
    "username": "trader1",
    "password": "secure123",
    "email": "trader@example.com"
}
requests.post('http://localhost:5000/auth/register', json=register_data)

# Login
login_data = {"username": "trader1", "password": "secure123"}
response = requests.post('http://localhost:5000/auth/login', json=login_data)
token = response.json()['token']

# Access protected endpoint
headers = {"Authorization": f"Bearer {token}"}
response = requests.get('http://localhost:5000/auth/profile', headers=headers)
profile = response.json()
```

## Development

### Adding a New Service
1. Create file in `services/` directory
2. Write functions with clear docstrings
3. Use absolute imports: `from services.X import Y`
4. Keep functions pure (no side effects when possible)

### Adding a New Route
1. Create blueprint in `routes/` directory
2. Import required services
3. Define routes with appropriate HTTP methods
4. Register blueprint in `api_server_modular.py`

### Adding Authentication
```python
@my_blueprint.route("/protected")
@token_required_decorator
def protected_route(current_user):
    return jsonify({"user": current_user})
```

### Adding Rate Limiting
```python
@my_blueprint.route("/limited")
@limiter.limit("10 per minute")
def limited_route():
    return jsonify({"message": "Rate limited"})
```

### Adding Caching
```python
@my_blueprint.route("/cached")
@cache.cached(timeout=300)
def cached_route():
    return jsonify(expensive_operation())
```

## Testing

### Manual Testing
Access the application at: `http://localhost:5000`

The frontend will automatically connect to the backend API.

### API Testing with curl
See `DEVELOPER_GUIDE.md` for comprehensive curl examples.

## Documentation

- **REFACTORING_COMPLETE.md** - Detailed refactoring documentation
- **DEVELOPER_GUIDE.md** - Complete developer guide with examples
- **README_MODULAR.md** - This file

## Migration Notes

### From Monolithic to Modular
The original `api_server.py` is **preserved** and still functional. Both versions can run side-by-side:

- **Monolithic:** `python api_server.py`
- **Modular:** `python api_server_modular.py`

All functionality is preserved. The modular version is a drop-in replacement.

### Frontend Compatibility
No changes required to frontend code. All endpoints maintain the same URLs and response formats.

## Production Deployment

### Recommended Setup
1. Use environment variables for secrets
2. Deploy with Gunicorn or Waitress (WSGI server)
3. Use Redis for caching instead of filesystem
4. Use PostgreSQL for data persistence
5. Set up HTTPS with Let's Encrypt
6. Configure Nginx as reverse proxy
7. Enable CORS for specific domains only

### Example Production Command
```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_server_modular:app
```

## Future Enhancements

### Short Term
- [ ] Add comprehensive unit tests
- [ ] Add API documentation (Swagger/OpenAPI)
- [ ] Implement proper logging
- [ ] Add input validation middleware

### Long Term
- [ ] Database integration (SQLAlchemy)
- [ ] WebSocket support for real-time data
- [ ] Celery for background tasks
- [ ] Redis for distributed caching
- [ ] Docker containerization
- [ ] CI/CD pipeline

## Contributing

### Code Style
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all functions
- Keep functions small and focused
- Use meaningful variable names

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Reference issues when applicable
- Keep first line under 50 characters

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or contributions, please refer to:
- **Documentation:** `DEVELOPER_GUIDE.md`
- **Refactoring Details:** `REFACTORING_COMPLETE.md`

## Acknowledgments

Built with:
- Flask - Web framework
- yfinance - Stock data API
- pandas - Data manipulation
- scikit-learn - Machine learning
- scipy - Scientific computing
