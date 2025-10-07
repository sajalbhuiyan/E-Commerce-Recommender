import os
import hashlib
import secrets
from pathlib import Path
import joblib

USERS_PATH = Path(__file__).resolve().parent / 'users.pkl'


def _load_users():
    if USERS_PATH.exists():
        try:
            data = joblib.load(USERS_PATH)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_users(users: dict):
    joblib.dump(users, USERS_PATH)


def _hash_password(password: str, salt: bytes) -> bytes:
    # Use PBKDF2-HMAC-SHA256
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)


def register_user(username: str, password: str) -> bool:
    """Register a new user. Returns True on success, False if user exists."""
    username = str(username).strip().lower()
    if not username or not password:
        return False
    users = _load_users()
    if username in users:
        return False
    salt = secrets.token_bytes(16)
    pw_hash = _hash_password(password, salt)
    users[username] = {'salt': salt, 'pw': pw_hash}
    _save_users(users)
    return True


def authenticate_user(username: str, password: str) -> bool:
    username = str(username).strip().lower()
    users = _load_users()
    u = users.get(username)
    if not u:
        return False
    salt = u.get('salt')
    pw_hash = u.get('pw')
    if not salt or not pw_hash:
        return False
    test_hash = _hash_password(password, salt)
    return secrets.compare_digest(test_hash, pw_hash)


def list_users():
    return list(_load_users().keys())
