"""Serverseitige Sitzungen, signierte Cookies, CSRF-Token und einfache Rate-Limits.

* ``cv_session``  zufaellige 256-Bit-ID, HttpOnly, SameSite=Lax, Secure (ausser ``COURTVISION_INSECURE_COOKIES=1``).
  Im Store liegt nur ``sha256(id)`` (docs-Art ``session``); 12 h gleitend, 30 d absolut, Logout loescht.
* ``cv_login``    kurzlebiges (10 min), signiertes Cookie auf einen **serverseitig** gehaltenen Code-Flow
  (docs-Art ``loginflow``). Das Codai-``flow`` verlaesst den Server nie und ist von einem anderen Browser aus nicht nutzbar.
* ``cv_csrf``     Double-Submit-Token (lesbar fuer JS, Header ``X-CSRF-Token``), HMAC-signiert und an die Sitzung gebunden.
* ``cv_device``   stabile Geraete-ID fuer Codai (``device``), HttpOnly, 1 Jahr.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from typing import Any, Callable, Optional

from ..store.eventstore import EventStore
from .config import AuthConfig, new_secret

SESSION_COOKIE = "cv_session"
LOGIN_COOKIE = "cv_login"
CSRF_COOKIE = "cv_csrf"
DEVICE_COOKIE = "cv_device"
CSRF_HEADER = "x-csrf-token"
SESSION_KIND = "session"
FLOW_KIND = "loginflow"
TOUCH_EVERY_S = 60


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


class Signer:
    """HMAC-SHA256 ueber ``purpose|value``; der Schluessel steht in ``COURTVISION_SECRET`` oder im Store."""

    def __init__(self, store: EventStore, cfg: AuthConfig) -> None:
        key = cfg.secret
        if not key:
            doc = store.get_doc("secret", "signing")
            if not doc or not doc.get("key"):
                doc = {"key": new_secret()}
                store.put_doc("secret", "signing", doc)
            key = doc["key"]
        self._key = key.encode()

    def mac(self, purpose: str, value: str) -> str:
        return _b64(hmac.new(self._key, f"{purpose}|{value}".encode(), hashlib.sha256).digest())

    def sign(self, purpose: str, value: str) -> str:
        return f"{value}.{self.mac(purpose, value)}"

    def unsign(self, purpose: str, token: Optional[str]) -> Optional[str]:
        if not token or "." not in token:
            return None
        value, _, sig = token.rpartition(".")
        return value if hmac.compare_digest(sig, self.mac(purpose, value)) else None

    # CSRF: nonce.mac(nonce|session_id) -- ohne Sitzung (Login-Formular) ist die session_id leer
    def csrf_token(self, session_id: str = "") -> str:
        nonce = secrets.token_urlsafe(18)
        return f"{nonce}.{self.mac('csrf', nonce + '|' + session_id)}"

    def csrf_valid(self, token: Optional[str], session_id: str = "") -> bool:
        if not token or "." not in token:
            return False
        nonce, _, sig = token.rpartition(".")
        return hmac.compare_digest(sig, self.mac("csrf", nonce + "|" + session_id))


class SessionManager:
    def __init__(self, store: EventStore, cfg: AuthConfig, clock: Callable[[], float] = time.time) -> None:
        self.store, self.cfg, self.clock = store, cfg, clock
        self._last_purge = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _key(sid: str) -> str:
        return hashlib.sha256(sid.encode()).hexdigest()

    def create(self, subject: str, ip: Optional[str] = None, user_agent: Optional[str] = None) -> str:
        sid = secrets.token_urlsafe(32)  # 256 Bit
        now = self.clock()
        self.store.put_doc(SESSION_KIND, self._key(sid), {
            "subject": subject, "created_at": now, "last_seen": now, "active": None,
            "ip": ip, "user_agent": (user_agent or "")[:200]})
        self.purge()
        return sid

    def get(self, sid: Optional[str]) -> Optional[dict[str, Any]]:
        if not sid or len(sid) > 128:
            return None
        key = self._key(sid)
        doc = self.store.get_doc(SESSION_KIND, key)
        if doc is None:
            return None
        now = self.clock()
        if now - doc["last_seen"] > self.cfg.idle_ttl_s or now - doc["created_at"] > self.cfg.abs_ttl_s:
            self.store.delete_doc(SESSION_KIND, key)
            return None
        if now - doc["last_seen"] > TOUCH_EVERY_S:  # gleitendes Ablaufdatum, Schreiblast begrenzt
            doc["last_seen"] = now
            self.store.put_doc(SESSION_KIND, key, doc)
        return doc

    def set_active(self, sid: str, active: Optional[dict[str, Any]]) -> None:
        key = self._key(sid)
        doc = self.store.get_doc(SESSION_KIND, key)
        if doc is not None:
            doc["active"] = active
            self.store.put_doc(SESSION_KIND, key, doc)

    def delete(self, sid: Optional[str]) -> None:
        if sid:
            self.store.delete_doc(SESSION_KIND, self._key(sid))

    def delete_for_subject(self, subject: str) -> int:
        n = 0
        for key, doc in self.store.list_docs(SESSION_KIND).items():
            if doc.get("subject") == subject:
                self.store.delete_doc(SESSION_KIND, key)
                n += 1
        return n

    def purge(self, force: bool = False) -> None:
        """Abgelaufene Sitzungen und Login-Flows entfernen (hoechstens alle 10 min)."""
        now = self.clock()
        with self._lock:
            if not force and now - self._last_purge < 600:
                return
            self._last_purge = now
        for key, doc in self.store.list_docs(SESSION_KIND).items():
            if now - doc["last_seen"] > self.cfg.idle_ttl_s or now - doc["created_at"] > self.cfg.abs_ttl_s:
                self.store.delete_doc(SESSION_KIND, key)
        for key, doc in self.store.list_docs(FLOW_KIND).items():
            if doc.get("created_at", 0) + self.cfg.flow_ttl_s + 60 < now:
                self.store.delete_doc(FLOW_KIND, key)


class RateLimiter:
    """Gleitendes Fenster im Speicher (pro Prozess), ergaenzt die Limits von Codai."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window, self.clock = limit, window_s, clock
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        """Zaehlt einen Versuch; ``False`` = Limit ueberschritten."""
        now = self.clock()
        with self._lock:
            if len(self._hits) > 20000:  # Speicherbremse
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - self.window}
            hits = [t for t in self._hits.get(key, ()) if t > now - self.window]
            if len(hits) >= self.limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True

    def retry_after(self, key: str) -> int:
        with self._lock:
            hits = self._hits.get(key) or [self.clock()]
            return max(1, int(hits[0] + self.window - self.clock()))
