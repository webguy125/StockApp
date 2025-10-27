"""
Authentication Routes
Handles user registration, login, and profile
"""

from flask import Blueprint, request, jsonify
from services.auth_service import (
    load_users, create_user, verify_user, generate_token
)

auth_bp = Blueprint('auth', __name__)


def init_auth_routes(data_dir, secret_key, limiter, token_required_decorator):
    """Initialize routes with configuration"""
    users_db = load_users(data_dir)

    @auth_bp.route("/auth/register", methods=["POST"])
    @limiter.limit("10 per hour")
    def register():
        """Register a new user"""
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        email = data.get("email")

        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        try:
            result = create_user(data_dir, users_db, username, password, email)
            return jsonify(result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @auth_bp.route("/auth/login", methods=["POST"])
    @limiter.limit("20 per hour")
    def login():
        """Login and get JWT token"""
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        user = verify_user(users_db, username, password)

        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        # Generate JWT token
        token = generate_token(username, secret_key)

        return jsonify({
            "success": True,
            "token": token,
            "username": username
        })

    @auth_bp.route("/auth/profile", methods=["GET"])
    @token_required_decorator
    def get_profile(current_user):
        """Get user profile (protected route)"""
        user = users_db.get(current_user)

        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "username": current_user,
            "email": user.get("email"),
            "created": user.get("created"),
            "role": user.get("role")
        })
