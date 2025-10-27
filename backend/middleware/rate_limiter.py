"""
Rate Limiter Middleware
Configuration for Flask-Limiter
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def setup_rate_limiter(app):
    """Configure and return rate limiter instance"""
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per hour", "50 per minute"],
        storage_uri="memory://"
    )

    return limiter
