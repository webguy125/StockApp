"""
Authentication Service Module
JWT token generation/verification, password hashing, user management
"""

import os
import json
import hashlib
import jwt
from datetime import datetime, timedelta


def load_users(data_dir):
    """Load users from file"""
    users_file = os.path.join(data_dir, "users.json")

    if os.path.exists(users_file):
        with open(users_file, "r") as f:
            return json.load(f)
    return {}


def save_users(data_dir, users_db):
    """Save users to file"""
    users_file = os.path.join(data_dir, "users.json")

    with open(users_file, "w") as f:
        json.dump(users_db, f, indent=2)


def hash_password(password):
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(data_dir, users_db, username, password, email=None):
    """Create a new user"""
    if username in users_db:
        raise ValueError("User already exists")

    users_db[username] = {
        "password": hash_password(password),
        "email": email,
        "created": datetime.now().isoformat(),
        "role": "user"
    }

    save_users(data_dir, users_db)

    return {
        "success": True,
        "message": "User registered successfully",
        "username": username
    }


def verify_user(users_db, username, password):
    """Verify user credentials"""
    user = users_db.get(username)

    if not user or user["password"] != hash_password(password):
        return None

    return user


def generate_token(username, secret_key):
    """Generate JWT token"""
    token = jwt.encode({
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, secret_key, algorithm='HS256')

    return token


def verify_token(token, secret_key):
    """Verify JWT token and return username"""
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]

        data = jwt.decode(token, secret_key, algorithms=['HS256'])
        return data['username']
    except:
        return None
