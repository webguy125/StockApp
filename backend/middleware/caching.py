"""
Caching Middleware
Configuration for Flask-Caching
"""

from flask_caching import Cache


def setup_cache(app):
    """Configure and return cache instance"""
    app.config['CACHE_TYPE'] = 'filesystem'
    app.config['CACHE_DIR'] = 'cache'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes

    cache = Cache(app)

    return cache
