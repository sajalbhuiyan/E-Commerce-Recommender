import csv
import hashlib
import secrets
from pathlib import Path

# CSV-backed user store (username,salt_hex,hash_hex)
USERS_CSV = Path(__file__).resolve().parent / 'users.csv'


def _read_users_from_csv(path: Path = USERS_CSV):
    users = {}
    if not path.exists():
        return users
    try:
        with path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uname = (row.get('username') or '').strip().lower()
                salt = row.get('salt_hex') or row.get('salt') or ''
                h = row.get('hash_hex') or row.get('hash') or ''
                if uname and salt and h:
                    users[uname] = {'salt_hex': salt, 'hash_hex': h}
    except Exception:
        return {}
    return users


def _write_users_to_csv(users: dict, path: Path = USERS_CSV):
    # users: {username: {'salt_hex':..., 'hash_hex':...}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['username', 'salt_hex', 'hash_hex'])
        writer.writeheader()
        for uname, info in users.items():
            writer.writerow({'username': uname, 'salt_hex': info['salt_hex'], 'hash_hex': info['hash_hex']})


def _hash_password(password: str, salt: bytes) -> bytes:
    # PBKDF2-HMAC-SHA256
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)


def register_user(username: str, password: str) -> bool:
    username = str(username or '').strip().lower()
    if not username or not password:
        return False
    users = _read_users_from_csv()
    if username in users:
        return False
    salt = secrets.token_bytes(16)
    h = _hash_password(password, salt)
    users[username] = {'salt_hex': salt.hex(), 'hash_hex': h.hex()}
    _write_users_to_csv(users)
    return True


def authenticate_user(username: str, password: str) -> bool:
    username = str(username or '').strip().lower()
    users = _read_users_from_csv()
    u = users.get(username)
    if not u:
        return False
    try:
        salt = bytes.fromhex(u['salt_hex'])
        expected = bytes.fromhex(u['hash_hex'])
    except Exception:
        return False
    test = _hash_password(password, salt)
    return secrets.compare_digest(test, expected)


def preload_users_from_file(csv_file_path: Path):
    """Overwrite the user store with an uploaded CSV file.

    Expected columns: username,salt_hex,hash_hex OR username,salt,hash (hex strings).
    Returns number of users loaded.
    """
    try:
        users = {}
        with Path(csv_file_path).open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uname = (row.get('username') or '').strip().lower()
                salt = row.get('salt_hex') or row.get('salt') or ''
                h = row.get('hash_hex') or row.get('hash') or ''
                if uname and salt and h:
                    users[uname] = {'salt_hex': salt, 'hash_hex': h}
        if users:
            _write_users_to_csv(users)
            return len(users)
    except Exception:
        return 0
    return 0


def list_users():
    return list(_read_users_from_csv().keys())
