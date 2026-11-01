"""``AuthService``: bündelt Konfiguration, Sitzungen, Benutzerverzeichnis und Codai-Client einer App-Instanz."""
from __future__ import annotations

import hmac
import logging
import re
import uuid
from typing import Any, Mapping, Optional

from starlette.requests import HTTPConnection

from ..store.eventstore import EventStore
from .codai import CodaiClient
from .config import AuthConfig
from .session import (CSRF_COOKIE, DEVICE_COOKIE, LOGIN_COOKIE, SESSION_COOKIE, RateLimiter, SessionManager, Signer)
from .users import FAN, UserDirectory, m_key

log = logging.getLogger("courtvision.auth")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s.]{2,}$")


class AuthService:
    def __init__(self, store: EventStore, cfg: AuthConfig, codai: Optional[CodaiClient] = None) -> None:
        self.store, self.cfg = store, cfg
        self.signer = Signer(store, cfg)
        self.sessions = SessionManager(store, cfg)
        self.users = UserDirectory(store, cfg)
        if codai is None and cfg.service_token and not cfg.is_dev:
            codai = CodaiClient.from_config(cfg)
        self.codai = codai
        w = cfg.limit_window_s
        self.lim_start_ip = RateLimiter(cfg.start_per_ip, w)
        self.lim_start_email = RateLimiter(cfg.start_per_email, w)
        self.lim_verify_ip = RateLimiter(cfg.verify_per_ip, w)
        self.lim_pw_ip = RateLimiter(cfg.password_per_ip, w)
        self.lim_pw_email = RateLimiter(cfg.password_per_email, w)
        if not cfg.is_dev and self.codai is None:
            log.info("Anmeldung per Codai ID nicht konfiguriert (CODAI_SERVICE_TOKEN fehlt): Login-Endpunkte "
                        "antworten 503; Maschinen-Clients mit COURTVISION_SYSTEM_TOKENS funktionieren weiter.")
        if cfg.is_dev:
            log.warning("COURTVISION_AUTH_MODE=dev: X-Role-Header werden akzeptiert -- NUR FUER ENTWICKLUNG.")

    # ------------------------------------------------------------ Anfrage-Infos
    def client_ip(self, conn: HTTPConnection) -> Optional[str]:
        """IP des Nutzers; ``X-Forwarded-For`` nur mit ``COURTVISION_TRUST_PROXY=1`` (von rechts, ``proxy_hops``)."""
        if self.cfg.trust_proxy:
            parts = [p.strip() for p in conn.headers.get("x-forwarded-for", "").split(",") if p.strip()]
            if len(parts) >= self.cfg.proxy_hops:
                return parts[-self.cfg.proxy_hops]
            if parts:
                return parts[0]
        return conn.client.host if conn.client else None

    @staticmethod
    def user_agent(conn: HTTPConnection) -> Optional[str]:
        return conn.headers.get("user-agent")

    def same_origin(self, conn: HTTPConnection) -> bool:
        """Browser-``Origin`` (falls vorhanden) muss zum Host passen."""
        origin = conn.headers.get("origin")
        if origin is None:
            return True
        host = conn.headers.get("host", "")
        if self.cfg.trust_proxy and conn.headers.get("x-forwarded-host"):
            host = conn.headers["x-forwarded-host"].split(",")[-1].strip()
        m = re.match(r"^https?://([^/]+)$", origin)
        return bool(m) and m.group(1).lower() == host.lower()

    # ------------------------------------------------------------ Token / Sitzung
    def system_token_ok(self, presented: str) -> bool:
        ok = False
        for t in self.cfg.system_tokens:  # alle vergleichen, kein Kurzschluss
            ok |= hmac.compare_digest(presented.encode(), t.encode())
        return ok

    def session_for(self, cookies: Mapping[str, str]) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        sid = cookies.get(SESSION_COOKIE)
        doc = self.sessions.get(sid)
        return (sid, doc) if doc else (None, None)

    def resolve_membership(self, user: Optional[dict[str, Any]], session: Optional[dict[str, Any]],
                           role: Optional[str] = None, team_id: Optional[str] = None,
                           player_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Aktive Mitgliedschaft. Explizite Anfrage (Header): muss zu den Mitgliedschaften des Benutzers passen,
        sonst ``None`` -- der Client kann keine fremde Rolle/kein fremdes Team behaupten."""
        mems = self.users.effective_memberships(user)
        if role:
            role = role.strip().lower()
            if role == "fan":
                return dict(FAN)
            for m in mems:
                if (m["role"] == role and (not team_id or m.get("team_id") == team_id)
                        and (not player_id or m.get("player_id") == player_id)):
                    return m
            return None
        act = (session or {}).get("active")
        if act:
            for m in mems:
                if m_key(m) == m_key(act):
                    return m
        return mems[0]

    # ------------------------------------------------------------ Darstellung
    def me(self, user: Optional[dict[str, Any]], active: Optional[dict[str, Any]]) -> dict[str, Any]:
        if user is None:
            return {"mode": self.cfg.mode, "authenticated": False, "role": "fan", "team_id": None, "player_id": None,
                    "memberships": [], "is_admin": False, "account_url": self.cfg.account_url}
        mems = self.users.effective_memberships(user)
        active = active or mems[0]
        return {
            "mode": self.cfg.mode, "authenticated": True, "subject": user["subject"], "username": user.get("username"),
            "email": user.get("email"), "role": active["role"], "team_id": active.get("team_id"),
            "player_id": active.get("player_id"), "label": self.users.label(active),
            "memberships": [{**m, "label": self.users.label(m), "active": m_key(m) == m_key(active)} for m in mems],
            "is_admin": any(m["role"] == "admin" for m in mems), "account_url": self.cfg.account_url,
        }

    # ------------------------------------------------------------ Geraete-ID fuer Codai
    @staticmethod
    def new_device_id() -> str:
        return uuid.uuid4().hex + uuid.uuid4().hex[:8]  # 40 Zeichen [0-9a-f]
