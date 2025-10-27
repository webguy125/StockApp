# Developer Guide - Modular Backend

## Quick Start

### Running the Server
```bash
cd C:\StockApp\backend
python api_server_modular.py
```

### Adding a New Endpoint

#### 1. Create the Service Function (if needed)
```python
# services/my_service.py
def calculate_something(data):
    """Business logic here"""
    return result
```

#### 2. Add Route to Appropriate Blueprint
```python
# routes/stock_routes.py (or create new blueprint)
from services.my_service import calculate_something

@stock_bp.route("/my-endpoint", methods=["POST"])
def my_endpoint():
    data = request.get_json()
    result = calculate_something(data)
    return jsonify(result)
```

#### 3. Register Blueprint (if new)
```python
# api_server_modular.py
from routes.my_routes import my_bp

app.register_blueprint(my_bp)
```

### Adding a New Service

```python
# services/analytics_service.py
"""
Analytics Service
Advanced analytics and reporting
"""

import pandas as pd
import numpy as np

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    # Implementation
    return sentiment_score

def calculate_correlation(df1, df2):
    """Calculate correlation between datasets"""
    # Implementation
    return correlation
```

### Adding Authentication to a Route

```python
from routes.my_routes import my_bp

@my_bp.route("/protected-endpoint")
@token_required_decorator  # Add this decorator
def protected_endpoint(current_user):  # current_user passed automatically
    """This endpoint requires authentication"""
    return jsonify({"user": current_user, "data": "secret"})
```

### Adding Rate Limiting

```python
@my_bp.route("/limited-endpoint")
@limiter.limit("10 per minute")  # Custom limit
def limited_endpoint():
    """This endpoint is rate limited"""
    return jsonify({"message": "Limited access"})
```

### Adding Caching

```python
@my_bp.route("/cached-endpoint")
@cache.cached(timeout=300)  # Cache for 5 minutes
def cached_endpoint():
    """This endpoint is cached"""
    expensive_operation()
    return jsonify({"data": result})
```

## Project Structure

```
backend/
├── api_server_modular.py       # Entry point
│
├── services/                   # Business logic
│   ├── stock_service.py        # Stock data operations
│   ├── indicator_service.py    # Technical indicators
│   ├── pattern_service.py      # Pattern detection
│   ├── prediction_service.py   # ML predictions
│   ├── trade_service.py        # Trade ideas
│   ├── portfolio_service.py    # Portfolio management
│   └── auth_service.py         # Authentication
│
├── routes/                     # API endpoints
│   ├── stock_routes.py         # Stock & indicators
│   ├── analysis_routes.py      # Analysis endpoints
│   ├── portfolio_routes.py     # Portfolio CRUD
│   ├── trendline_routes.py     # Trendlines & sharing
│   ├── plugin_routes.py        # Plugin system
│   ├── auth_routes.py          # Auth endpoints
│   ├── activity_routes.py      # Activity logging
│   └── admin_routes.py         # Admin endpoints
│
└── middleware/                 # Cross-cutting concerns
    ├── auth_middleware.py      # JWT authentication
    ├── rate_limiter.py         # Rate limiting
    └── caching.py              # Caching setup
```

## Import Patterns

### Absolute Imports (Recommended)
```python
from services.stock_service import fetch_stock_data
from routes.stock_routes import stock_bp
from middleware.auth_middleware import require_auth
```

### Within Same Package
```python
# In services/trade_service.py
from services.indicator_service import calculate_rsi
```

## Configuration

### Location
All configuration is in `api_server_modular.py`:
```python
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['CACHE_TYPE'] = 'filesystem'
app.config['CACHE_DIR'] = 'cache'
```

### Best Practice
For production, use environment variables:
```python
import os
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key')
```

## Common Tasks

### 1. Add New Technical Indicator

**File:** `services/indicator_service.py`
```python
def calculate_obv(df):
    """Calculate On-Balance Volume"""
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)
```

**Then use in:** `routes/stock_routes.py`
```python
from services.indicator_service import calculate_obv

# In calculate_indicators():
elif ind_type == "OBV":
    obv = calculate_obv(df)
    result["OBV"] = obv.replace({np.nan: None}).tolist()
```

### 2. Add New Pattern Detection

**File:** `services/pattern_service.py`
```python
def detect_triangle(df):
    """Detect triangle pattern"""
    patterns = []
    # Implementation
    return patterns
```

**Use in:** `routes/analysis_routes.py`
```python
from services.pattern_service import detect_triangle

# In detect_patterns():
patterns.extend(detect_triangle(df))
```

### 3. Add New Trade Strategy

**File:** `services/trade_service.py`
```python
def detect_breakout(df, current_price):
    """Detect breakout trading opportunities"""
    ideas = []
    # Implementation
    return ideas
```

**Use in:** `routes/analysis_routes.py`
```python
# Already imported in generate_trade_ideas()
breakout_ideas = detect_breakout(df, current_price)
ideas.extend(breakout_ideas)
```

## Testing

### Manual Testing with curl

**Get stock data:**
```bash
curl http://localhost:5000/data/AAPL?period=1mo
```

**Calculate indicators:**
```bash
curl -X POST http://localhost:5000/indicators \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "period": "1mo",
    "indicators": [
      {"type": "SMA", "params": {"period": 20}},
      {"type": "RSI", "params": {"period": 14}}
    ]
  }'
```

**Register user:**
```bash
curl -X POST http://localhost:5000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123",
    "email": "test@example.com"
  }'
```

**Login:**
```bash
curl -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'
```

**Access protected endpoint:**
```bash
curl http://localhost:5000/auth/profile \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE"
```

## Troubleshooting

### Import Errors
- Ensure you're in the `backend` directory when running
- Check that `__init__.py` files exist in all packages
- Use absolute imports: `from services.X import Y`

### Route Not Found
- Check blueprint is registered in `api_server_modular.py`
- Verify route path matches what you're calling
- Check if route requires authentication

### Database/File Errors
- Ensure `data/` directory exists
- Check file permissions
- Verify JSON files are valid

### Plugin Errors
- Check `plugins/` directory exists
- Verify plugin inherits from `BasePlugin`
- Ensure `get_info()` and `calculate()` methods implemented

## Performance Tips

1. **Use Caching**: Add `@cache.cached()` to expensive operations
2. **Rate Limiting**: Protect endpoints with `@limiter.limit()`
3. **Batch Operations**: Fetch all data at once when possible
4. **Async Operations**: Consider async for I/O-heavy tasks

## Security Best Practices

1. **Always hash passwords**: Use `hash_password()` from auth_service
2. **Validate input**: Check and sanitize all user input
3. **Use HTTPS**: In production, always use SSL/TLS
4. **Environment variables**: Never commit secrets to git
5. **Rate limiting**: Protect all public endpoints
6. **JWT expiration**: Tokens expire after 24 hours

## Next Steps

1. Add database integration (SQLAlchemy)
2. Write unit tests (pytest)
3. Add API documentation (Swagger)
4. Set up logging (structlog)
5. Deploy to production (gunicorn + nginx)
