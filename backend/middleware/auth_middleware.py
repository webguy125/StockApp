"""
Authentication Middleware
Token verification decorator
"""

from flask import request, jsonify
from functools import wraps
from services.auth_service import verify_token


def require_auth(secret_key):
    """Decorator factory to require JWT token for protected routes"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')

            if not token:
                return jsonify({'error': 'Token is missing'}), 401

            username = verify_token(token, secret_key)

            if not username:
                return jsonify({'error': 'Token is invalid'}), 401

            return f(username, *args, **kwargs)

        return decorated
    return decorator
