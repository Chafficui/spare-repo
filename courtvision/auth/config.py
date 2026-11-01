"""Konfiguration der Anmeldung (Umgebungsvariablen) -- siehe README, Abschnitt "Anmeldung (Codai ID)"."""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

log = logging.getLogger("courtvision.auth")

MODES = ("codai", "dev")
DEFAULT_CODAI_URL = "https://auth.codai.app"
MIN_SYSTEM_TOKEN_LEN = 16


def _flag(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _csv(v: Optional[str]) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


@dataclass(frozen=True)
class AuthConfig:
    #: ``codai`` (Standard): Sitzungs-Cookie / Bearer-Token. ``dev``: zusaetzlich die Platzhalter-Header X-Role ...
    mode: str = "codai"
    codai_url: str = DEFAULT_CODAI_URL
    #: Oeffentliche Basis-URL fuer Links im Browser (Konto, Passwort vergessen); Standard = ``codai_url``.
    codai_public_url: str = ""
    service_token: str = ""
    client_id: str = ""
    codai_timeout_s: float = 10.0
    trust_proxy: bool = False
    #: Anzahl vertrauenswuerdiger Proxys vor der App (X-Forwarded-For wird von rechts gezaehlt).
    proxy_hops: int = 1
    insecure_cookies: bool = False
    #: Codai-Subjects und/oder E-Mail-Adressen (klein geschrieben) mit Admin-Rolle.
    admins: frozenset = frozenset()
    system_tokens: tuple = ()
    #: Signaturschluessel fuer cv_login/cv_csrf; leer = wird im Store erzeugt und gespeichert.
    secret: str = ""
    idle_ttl_s: int = 12 * 3600
    abs_ttl_s: int = 30 * 86400
    flow_ttl_s: int = 10 * 60
    resend_wait_s: int = 30
    #: Die beim Login getippte Adresse gilt als bestaetigt (der Code ging an sie / das Passwort gehoert zu ihr),
    #: wenn Codai keine Adresse zurueckgibt (EMAIL_CLAIM_SERVICES).
    trust_flow_email: bool = True
    #: Limits pro Fenster (Sekunden) -- zusaetzlich zu denen von Codai.
    limit_window_s: int = 15 * 60
    start_per_ip: int = 10
    start_per_email: int = 5
    verify_per_ip: int = 30
    password_per_ip: int = 20
    password_per_email: int = 10
    demo: bool = False
    #: Demo: (E-Mail, Beschreibung) -- die Anmeldeseite zeigt sie als Schnellauswahl.
    demo_accounts: tuple = ()

    @property
    def public_url(self) -> str:
        return (self.codai_public_url or self.codai_url).rstrip("/")

    @property
    def account_url(self) -> str:
        return self.public_url + "/account"

    @property
    def password_reset_url(self) -> str:
        return self.public_url + "/password?reset=1"

    @property
    def is_dev(self) -> bool:
        return self.mode == "dev"

    def with_(self, **kw) -> "AuthConfig":
        return replace(self, **kw)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, **overrides) -> "AuthConfig":
        e = os.environ if env is None else env
        mode = (e.get("COURTVISION_AUTH_MODE") or "codai").strip().lower()
        if mode not in MODES:
            raise ValueError(f"COURTVISION_AUTH_MODE muss {' oder '.join(MODES)} sein, nicht '{mode}'")
        tokens = []
        for t in _csv(e.get("COURTVISION_SYSTEM_TOKENS")):
            if len(t) < MIN_SYSTEM_TOKEN_LEN:
                log.warning("COURTVISION_SYSTEM_TOKENS: Token mit weniger als %d Zeichen wird ignoriert",
                            MIN_SYSTEM_TOKEN_LEN)
                continue
            tokens.append(t)
        try:
            hops = max(1, int(e.get("COURTVISION_PROXY_HOPS") or 1))
        except ValueError:
            hops = 1
        cfg = cls(
            mode=mode,
            codai_url=(e.get("CODAI_AUTH_URL") or DEFAULT_CODAI_URL).rstrip("/"),
            service_token=(e.get("CODAI_SERVICE_TOKEN") or "").strip(),
            client_id=(e.get("CODAI_CLIENT_ID") or "").strip(),
            trust_proxy=_flag(e.get("COURTVISION_TRUST_PROXY")),
            proxy_hops=hops,
            insecure_cookies=_flag(e.get("COURTVISION_INSECURE_COOKIES")),
            admins=frozenset(a.lower() for a in _csv(e.get("COURTVISION_ADMINS"))),
            system_tokens=tuple(tokens),
            secret=(e.get("COURTVISION_SECRET") or "").strip(),
            trust_flow_email=(e.get("COURTVISION_TRUST_TYPED_EMAIL", "1").strip() != "0"),
        )
        return replace(cfg, **overrides) if overrides else cfg


def new_secret() -> str:
    return secrets.token_urlsafe(48)
