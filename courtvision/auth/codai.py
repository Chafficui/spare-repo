"""Server-zu-Server-Client fuer die "Headless sign-in"-API von Codai ID (``/service/login/*``).

Der Browser spricht **nie** direkt mit Codai und sieht das Service-Token nie. Codai beantwortet nur die Frage
*wer ist das?* (``subject`` = Codai-User-ID, ``username``, optional ``email``); welche Rechte jemand in CourtVision hat,
entscheidet CourtVision (:mod:`courtvision.auth.users`).

Vertrag (Codai ``src/headless.ts`` / ``src/service-api.ts``)::

    Authorization: Bearer <Service-Token>        (mind. 32 Zeichen; der Dienst muss in HEADLESS_LOGIN_SERVICES stehen)
    POST /service/login/code/start   {email, locale?, client?, device?, ip?, userAgent?} -> {flow, expiresAt, codeLength}
    POST /service/login/code/verify  {flow, code, device?, ip?, userAgent?}             -> {subject, created, username, email?}
    POST /service/login/password     {email, password, locale?, client?, device?, ...}   -> {subject, created, username, email?}
                                                                                           | {stepUp: true, flow, expiresAt, codeLength}
    GET  /service/users/<subject>                                                       -> {subject, username, ...} | 404

Fehler (JSON ``{"error": "<code>"}``): 400 bad_email / bad_code(+attemptsLeft) / bad_device, 401 bad_credentials
(bzw. ``unauthorized`` bei falschem Service-Token), 403 forbidden, 404 unknown, 410 expired, 429 rate_limited,
502 mail_failed, 503 busy / unavailable.

``ip`` und ``userAgent`` des Nutzers werden immer weitergereicht: Codai zaehlt seine Limits pro Nutzer-IP; ohne IP
teilen sich alle Nutzer eines Dienstes einen Eimer.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Optional, Union

import httpx

from .config import AuthConfig


# ------------------------------------------------------------------ Ergebnisse
@dataclass(frozen=True)
class CodeStarted:
    flow: str
    expires_at: int  # ms seit Epoche (Codai)
    code_length: int = 6


@dataclass(frozen=True)
class StepUp:
    """Passwort war richtig, aber das Geraet ist neu: Codai hat einen Bestaetigungs-Code gemailt (``verify_code``)."""
    flow: str
    expires_at: int
    code_length: int = 6


@dataclass(frozen=True)
class Identity:
    subject: str
    username: Optional[str]
    created: bool = False
    email: Optional[str] = None  # nur mit EMAIL_CLAIM_SERVICES


@dataclass(frozen=True)
class CodaiUser:
    subject: str
    username: Optional[str]


# ------------------------------------------------------------------ Fehler
class CodaiError(Exception):
    """Basis. ``code`` = Codai-Fehlercode, ``status`` = HTTP-Status von Codai."""
    code = "codai_error"
    status = 502

    def __init__(self, message: str = "", *, status: Optional[int] = None, code: Optional[str] = None) -> None:
        super().__init__(message or self.code)
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code


class BadEmail(CodaiError):
    code, status = "bad_email", 400


class BadCode(CodaiError):
    code, status = "bad_code", 400

    def __init__(self, message: str = "", *, attempts_left: Optional[int] = None) -> None:
        super().__init__(message)
        self.attempts_left = attempts_left


class BadDevice(CodaiError):
    code, status = "bad_device", 400


class Expired(CodaiError):
    code, status = "expired", 410


class BadCredentials(CodaiError):
    code, status = "bad_credentials", 401


class RateLimited(CodaiError):
    code, status = "rate_limited", 429


class MailFailed(CodaiError):
    code, status = "mail_failed", 502


class Busy(CodaiError):
    code, status = "busy", 503


class Unavailable(CodaiError):
    """503 unavailable, Netzwerkfehler, Zeitueberschreitung, unerwartete Antworten."""
    code, status = "unavailable", 503


class Misconfigured(CodaiError):
    """401 unauthorized (Service-Token falsch) oder 403 forbidden (Dienst nicht in HEADLESS_LOGIN_SERVICES)."""
    code, status = "misconfigured", 502


class UnknownUser(CodaiError):
    code, status = "unknown", 404


_BY_CODE = {"bad_email": BadEmail, "bad_device": BadDevice, "expired": Expired, "bad_credentials": BadCredentials,
            "rate_limited": RateLimited, "mail_failed": MailFailed, "busy": Busy, "unavailable": Unavailable,
            "unknown": UnknownUser}


def _valid_ip(ip: Optional[str]) -> Optional[str]:
    if not ip:
        return None
    try:
        return str(ipaddress.ip_address(ip.strip()))
    except ValueError:
        return None


class CodaiClient:
    """``CodaiClient(base_url, service_token, timeout)``.

    ``http`` darf ein fertiger ``httpx.Client`` sein (Tests/Demo: ``fastapi.testclient.TestClient`` auf die
    Fake-Codai-App, ohne Netzwerk)."""

    def __init__(self, base_url: str, service_token: str, timeout: float = 10.0, *, client_id: str = "",
                 http: Optional[httpx.Client] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self._token = service_token
        self._http = http if http is not None else httpx.Client(timeout=timeout)
        self._timeout = timeout

    @classmethod
    def from_config(cls, cfg: AuthConfig, http: Optional[httpx.Client] = None) -> "CodaiClient":
        return cls(cfg.codai_url, cfg.service_token, cfg.codai_timeout_s, client_id=cfg.client_id, http=http)

    def __repr__(self) -> str:  # Token nie ins Log
        return f"CodaiClient({self.base_url!r}, token=***)"

    # --------------------------------------------------------------- Transport
    def _origin(self, ip: Optional[str], user_agent: Optional[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if _valid_ip(ip):
            out["ip"] = _valid_ip(ip)
        if user_agent:
            out["userAgent"] = user_agent[:512]
        return out

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict[str, Any]:
        try:
            r = self._http.request(method, self.base_url + path, json=body,
                                   headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"})
        except httpx.HTTPError as e:
            raise Unavailable(f"Codai nicht erreichbar: {type(e).__name__}") from e
        try:
            data = r.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            data = {}
        if r.status_code == 200:
            return data
        raise self._error(r.status_code, data)

    @staticmethod
    def _error(status: int, data: dict[str, Any]) -> CodaiError:
        code = str(data.get("error") or "")
        if status == 401 and code in ("unauthorized", ""):
            return Misconfigured("Codai lehnt das Service-Token ab (401 unauthorized)", status=401, code="unauthorized")
        if status == 403:
            return Misconfigured("Der Dienst darf Codai nicht headless nutzen (403 forbidden): HEADLESS_LOGIN_SERVICES?",
                                 status=403, code="forbidden")
        if code == "bad_code":
            left = data.get("attemptsLeft")
            return BadCode(attempts_left=left if isinstance(left, int) else None)
        cls = _BY_CODE.get(code)
        if cls is not None:
            return cls(status=status)
        if status >= 500:
            return Unavailable(f"Codai antwortet {status} {code}".strip(), status=status)
        return CodaiError(f"Unerwartete Codai-Antwort {status} {code}".strip(), status=status, code=code or "codai_error")

    @staticmethod
    def _flow(data: dict[str, Any], cls):
        try:
            return cls(flow=str(data["flow"]), expires_at=int(data.get("expiresAt") or 0),
                       code_length=int(data.get("codeLength") or 6))
        except (KeyError, TypeError, ValueError) as e:
            raise Unavailable("Unerwartete Codai-Antwort (flow fehlt)") from e

    @staticmethod
    def _identity(data: dict[str, Any]) -> Identity:
        sub = data.get("subject")
        if not isinstance(sub, str) or not sub:
            raise Unavailable("Unerwartete Codai-Antwort (subject fehlt)")
        un = data.get("username")
        em = data.get("email")
        return Identity(subject=sub.lower(), username=un if isinstance(un, str) and un else None,
                        created=bool(data.get("created")), email=em.strip().lower() if isinstance(em, str) and em else None)

    # --------------------------------------------------------------- API
    def start_code(self, email: str, locale: str = "de", ip: Optional[str] = None,
                   user_agent: Optional[str] = None, device: Optional[str] = None) -> CodeStarted:
        body: dict[str, Any] = {"email": email, "locale": locale if locale in ("de", "en") else "de",
                                **self._origin(ip, user_agent)}
        if self.client_id:
            body["client"] = self.client_id
        if device:
            body["device"] = device
        return self._flow(self._request("POST", "/service/login/code/start", body), CodeStarted)

    def verify_code(self, flow: str, code: str, ip: Optional[str] = None, user_agent: Optional[str] = None,
                    device: Optional[str] = None) -> Identity:
        body: dict[str, Any] = {"flow": flow, "code": code, **self._origin(ip, user_agent)}
        if device:
            body["device"] = device
        return self._identity(self._request("POST", "/service/login/code/verify", body))

    def password_login(self, email: str, password: str, locale: str = "de", ip: Optional[str] = None,
                       user_agent: Optional[str] = None, device: Optional[str] = None) -> Union[Identity, StepUp]:
        body: dict[str, Any] = {"email": email, "password": password, "locale": locale if locale in ("de", "en") else "de",
                                **self._origin(ip, user_agent)}
        if self.client_id:
            body["client"] = self.client_id
        if device:
            body["device"] = device
        data = self._request("POST", "/service/login/password", body)
        if data.get("stepUp"):
            return self._flow(data, StepUp)
        return self._identity(data)

    def get_user(self, subject: str) -> CodaiUser:
        data = self._request("GET", f"/service/users/{subject}")
        return CodaiUser(subject=str(data.get("subject") or subject).lower(),
                         username=data.get("username") if isinstance(data.get("username"), str) else None)
