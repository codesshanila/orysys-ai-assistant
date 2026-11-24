"""Authentication: hardcoded demo users (brief Option A) + signed JWT sessions.

Passwords are stored as salted PBKDF2 hashes, never in plain text, and compared in
constant time. The role inside the token is the ONLY source of authorisation for the
request - nothing the user types or the model generates can change it.

Swap-in path: replace `authenticate` with an OIDC token check (e.g. Keycloak) and map
realm roles to `Role`; the rest of the system is unchanged.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt

from app.config import get_settings
from app.security.rbac import Role

_SALT = b"orysys-demo-salt"


def _hash(password: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), _SALT, 100_000).hex()


@dataclass(frozen=True)
class User:
    username: str
    display_name: str
    role: Role
    department: str


# Demo credentials are documented in the README.
_USERS: dict[str, tuple[str, User]] = {
    "viewer": (_hash("viewer123"), User("viewer", "Vimukthi (Viewer)", Role.VIEWER, "branch-operations")),
    "analyst": (_hash("analyst123"), User("analyst", "Anjali (Analyst)", Role.ANALYST, "payments")),
    "admin": (_hash("admin123"), User("admin", "Ashan (Administrator)", Role.ADMIN, "technology")),
}


def authenticate(username: str, password: str) -> User | None:
    record = _USERS.get(username.strip().lower())
    if record is None:
        _hash(password)  # equalise timing for unknown users
        return None
    stored, user = record
    return user if hmac.compare_digest(stored, _hash(password)) else None


def issue_token(user: User) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {"sub": user.username, "role": user.role.value, "iat": now,
               "exp": now + timedelta(minutes=s.jwt_ttl_minutes)}
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> User | None:
    try:
        payload = jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    record = _USERS.get(payload.get("sub", ""))
    if record is None:
        return None
    user = record[1]
    # Role comes from the server-side user record, not just the token claim.
    return user if user.role.value == payload.get("role") else None
