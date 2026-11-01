"""Fake-Codai: bildet die Headless-Endpunkte von Codai ID samt Fehlersemantik nach (Entwicklung, Demo, Tests).

Nachgebildet nach Codai ``src/headless.ts``: Bearer-Token (mind. 32 Zeichen) + Gate ``HEADLESS_LOGIN_SERVICES``, Codes
6-stellig / 10 Minuten / einmalig / 5 Fehlversuche, Limits (5 Codes je Adresse und 15 min, 20 je Dienst+IP und 15 min),
Passwort-Login mit Geraete-Pruefung (``stepUp``), E-Mail-Claim (``EMAIL_CLAIM_SERVICES``).

**Nur fuer Entwicklung.** Die Codes stehen im Log (und in ``FakeCodai.outbox``); nicht im offenen Netz betreiben.

Einbindung::

    fake = FakeCodai(...)
    app.mount("/_fake_codai", fake.app)                 # im Prozess (siehe courtvision.cli serve --demo)
    python -m courtvision.auth.fake_codai --port 4460   # eigener Prozess (CODAI_AUTH_URL=http://127.0.0.1:4460)
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

log = logging.getLogger("courtvision.fake_codai")

NS = uuid.UUID("5b0a6f64-7d59-4e2b-9a3b-0c0a1d1d0001")
DEFAULT_SERVICE = "courtvision"
DEFAULT_TOKEN = "fake-codai-service-token-0123456789abcdef"  # >= 32 Zeichen
CODE_TTL_S = 600
MAX_ATTEMPTS = 5
START_PER_ADDRESS = 5
START_PER_SERVICE_IP = 20
PASSWORD_PER_IP = 20
PASSWORD_PER_ADDRESS = 10
WINDOW_S = 15 * 60
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DEVICE = re.compile(r"^[A-Za-z0-9._~:+=-]{16,128}$")


def normalise_email(raw: str) -> str:
    return raw.strip().lower()


@dataclass
class _Flow:
    email: str
    service: str
    code: str
    expires: float
    attempts: int = 0
    used: bool = False
    step_up_device: Optional[str] = None
    locale: str = "de"


class FakeCodai:
    def __init__(self, *, tokens: Optional[dict[str, str]] = None, headless: Optional[set[str]] = None,
                 email_claim: Optional[set[str]] = None, fixed_code: Optional[str] = None,
                 passwords: Optional[dict[str, str]] = None, usernames: Optional[dict[str, str]] = None,
                 auto_usernames: bool = False, mail_fail: Optional[set[str]] = None,
                 clock: Callable[[], float] = time.time, announce: Optional[Callable[[str], None]] = None) -> None:
        self.tokens = dict(tokens if tokens is not None else {DEFAULT_SERVICE: DEFAULT_TOKEN})
        self.headless = set(headless if headless is not None else self.tokens)
        self.email_claim = set(email_claim or ())
        self.fixed_code = fixed_code
        self.passwords = {normalise_email(k): v for k, v in (passwords or {}).items()}
        self.usernames = {normalise_email(k): v for k, v in (usernames or {}).items()}
        self.auto_usernames = auto_usernames
        self.mail_fail = {normalise_email(m) for m in (mail_fail or ())}
        self.clock = clock
        self.announce = announce or (lambda line: log.warning(line))
        self.outbox: list[dict[str, Any]] = []  # {to, code, flow, kind}
        self.accounts: dict[str, str] = {}  # email -> subject (nach dem ersten erfolgreichen Login)
        self.known_devices: dict[str, set[str]] = {}
        self.sign_ins: list[dict[str, Any]] = []  # Protokoll: method, ip, userAgent (fuer Tests)
        self.requests: list[dict[str, Any]] = []  # Protokoll aller Service-Aufrufe (ohne Passwoerter)
        self._flows: dict[str, _Flow] = {}
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.RLock()
        self.app = self._build()

    # ------------------------------------------------------------ Hilfen
    @staticmethod
    def subject_for(email: str) -> str:
        return str(uuid.uuid5(NS, normalise_email(email)))

    def username_for(self, email: str) -> Optional[str]:
        email = normalise_email(email)
        if email in self.usernames:
            return self.usernames[email]
        if self.auto_usernames:
            return re.sub(r"[^a-z0-9_.-]", "", email.split("@")[0]) or None
        return None

    def last_code(self, email: str) -> Optional[str]:
        """Test-Hook: letzter gemailter Code an ``email``."""
        for m in reversed(self.outbox):
            if m["to"] == normalise_email(email):
                return m["code"]
        return None

    def _take(self, key: str, limit: int) -> bool:
        now = self.clock()
        hits = [t for t in self._hits.get(key, []) if t > now - WINDOW_S]
        if len(hits) >= limit:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True

    def _count(self, key: str) -> int:
        now = self.clock()
        return len([t for t in self._hits.get(key, []) if t > now - WINDOW_S])

    def _new_code(self) -> str:
        return self.fixed_code or f"{secrets.randbelow(10**6):06d}"

    def _mail_code(self, email: str, service: str, kind: str, locale: str, device: Optional[str] = None) -> tuple[str, _Flow]:
        code = self._new_code()
        flow_id = secrets.token_urlsafe(32)[:43]
        fl = _Flow(email, service, code, self.clock() + CODE_TTL_S, step_up_device=device, locale=locale)
        self._flows[flow_id] = fl
        self.outbox.append({"to": email, "code": code, "flow": flow_id, "kind": kind, "service": service})
        self.announce(f"[fake-codai] {'Bestaetigungs-' if kind == 'stepup' else 'Anmelde-'}Code fuer {email}: {code}")
        return flow_id, fl

    @staticmethod
    def _json(status: int, body: dict) -> JSONResponse:
        return JSONResponse(body, status_code=status, headers={"cache-control": "no-store"})

    # ------------------------------------------------------------ App
    def _build(self) -> FastAPI:
        app = FastAPI(title="Fake Codai ID (nur Entwicklung)", docs_url=None, redoc_url=None, openapi_url=None)

        def service_of(request: Request) -> Optional[str]:
            m = re.match(r"^Bearer\s+(\S{32,})$", request.headers.get("authorization", ""))
            if not m:
                return None
            found = None
            for name, tok in self.tokens.items():  # alle vergleichen (konstante Zeit)
                if hmac.compare_digest(m.group(1).encode(), tok.encode()):
                    found = name
            return found

        async def body_of(request: Request) -> Optional[dict]:
            try:
                raw = await request.body()
                if len(raw) > 8192:
                    return None
                data = __import__("json").loads(raw)
                return data if isinstance(data, dict) else None
            except ValueError:
                return None

        def origin_ip(fields: dict) -> str:
            ip = fields.get("ip")
            return ip.strip() if isinstance(ip, str) and ip.strip() else "-"

        def device_of(fields: dict, ip: str) -> tuple[Optional[str], bool]:
            """(device-Hash, gueltig). Ohne Geraete-ID steht der (ip, userAgent)-Fingerabdruck ein."""
            d = fields.get("device")
            if d is None:
                fp = hashlib.sha256(f"{ip}|{fields.get('userAgent') or ''}".encode()).hexdigest()
                return fp, True
            if not isinstance(d, str) or not _DEVICE.match(d):
                return None, False
            return hashlib.sha256(d.encode()).hexdigest(), True

        def establish(email: str, service: str, method: str, fields: dict, device: str) -> dict:
            created = email not in self.accounts
            sub = self.accounts.setdefault(email, self.subject_for(email))
            self.known_devices.setdefault(sub, set()).add(device)
            self.sign_ins.append({"subject": sub, "method": method, "ip": origin_ip(fields),
                                  "userAgent": fields.get("userAgent"), "service": service})
            out = {"subject": sub, "created": created, "username": self.username_for(email)}
            if service in self.email_claim:
                out["email"] = email
            return out

        @app.api_route("/service/{rest:path}", methods=["GET", "POST"])
        async def service(rest: str, request: Request):
            path = "/service/" + rest
            svc = service_of(request)
            if svc is None:
                return self._json(401, {"error": "unauthorized"})
            if svc not in self.headless:
                return self._json(403, {"error": "forbidden"})
            fields = await body_of(request) if request.method == "POST" else None
            with self._lock:
                if request.method == "GET":
                    m = re.fullmatch(r"/service/users/([^/]+)", path)
                    if not m or not re.fullmatch(r"[0-9a-f-]{36}", m.group(1), re.I):
                        return self._json(404, {"error": "unknown" if m else "not found"})
                    sub = m.group(1).lower()
                    for em, s in self.accounts.items():
                        if s == sub:
                            return self._json(200, {"subject": sub, "username": self.username_for(em),
                                                    "usernameChangedAt": None, "nextChangeAt": None})
                    return self._json(404, {"error": "unknown"})
                if fields is None:
                    return self._json(400, {"error": "body must be a JSON object"})
                self.requests.append({"path": path, "service": svc,
                                      **{k: v for k, v in fields.items() if k not in ("password", "code")}})
                handler = {"/service/login/code/start": self._start, "/service/login/code/verify": self._verify,
                           "/service/login/password": self._password}.get(path)
                if handler is None:
                    return self._json(404, {"error": "not found"})
                return handler(svc, fields, origin_ip, device_of, establish)

        @app.get("/account", response_class=HTMLResponse)
        def account():
            return _STUB.format(title="Codai ID - Konto (Fake)", text="Hier verwaltet man sein Codai-ID-Konto.")

        @app.get("/password", response_class=HTMLResponse)
        def password():
            return _STUB.format(title="Codai ID - Passwort (Fake)", text="Hier setzt man das Passwort zurueck.")

        return app

    # ------------------------------------------------------------ Endpunkte
    def _start(self, svc, fields, origin_ip, device_of, establish):
        em = fields.get("email")
        email = normalise_email(em) if isinstance(em, str) else ""
        if not _EMAIL.match(email):
            return self._json(400, {"error": "bad_email"})
        ip = origin_ip(fields)
        _, ok = device_of(fields, ip)
        if not ok:
            return self._json(400, {"error": "bad_device"})
        if not self._take(f"ip|{svc}|{ip}", START_PER_SERVICE_IP):
            return self._json(429, {"error": "rate_limited"})
        if not self._take(f"addr|{email}", START_PER_ADDRESS):
            return self._json(429, {"error": "rate_limited"})
        if email in self.mail_fail:
            self._hits[f"addr|{email}"].pop()  # Mail nie versendet: Kontingent nicht verbraucht
            return self._json(502, {"error": "mail_failed"})
        loc = fields.get("locale") if fields.get("locale") in ("de", "en") else "en"
        flow_id, fl = self._mail_code(email, svc, "code", loc)
        return self._json(200, {"flow": flow_id, "expiresAt": int(fl.expires * 1000), "codeLength": 6})

    def _verify(self, svc, fields, origin_ip, device_of, establish):
        flow_id, code = fields.get("flow"), fields.get("code")
        fl = self._flows.get(flow_id) if isinstance(flow_id, str) else None
        ip = origin_ip(fields)
        device, ok = device_of(fields, ip)
        if not ok:
            return self._json(400, {"error": "bad_device"})
        if (fl is None or fl.used or fl.service != svc or fl.expires < self.clock() or fl.attempts >= MAX_ATTEMPTS
                or (fl.step_up_device is not None and fl.step_up_device != device)):
            return self._json(410, {"error": "expired"})
        if not isinstance(code, str) or not hmac.compare_digest(code.encode(), fl.code.encode()):
            fl.attempts += 1
            return self._json(400, {"error": "bad_code", "attemptsLeft": MAX_ATTEMPTS - fl.attempts})
        fl.used = True
        method = "password" if fl.step_up_device is not None else "email_code"
        return self._json(200, establish(fl.email, svc, method, fields, device))

    def _password(self, svc, fields, origin_ip, device_of, establish):
        em, pw = fields.get("email"), fields.get("password")
        email = normalise_email(em) if isinstance(em, str) else ""
        if not _EMAIL.match(email) or not isinstance(pw, str):
            return self._json(401, {"error": "bad_credentials"})
        ip = origin_ip(fields)
        device, ok = device_of(fields, ip)
        if not ok:
            return self._json(400, {"error": "bad_device"})
        if not self._take(f"pw-ip|{ip}", PASSWORD_PER_IP):
            return self._json(429, {"error": "rate_limited"})
        if self._count(f"pw-addr|{email}") >= PASSWORD_PER_ADDRESS:
            return self._json(401, {"error": "bad_credentials"})  # gesperrt: auch richtiges Passwort wird abgelehnt
        stored = self.passwords.get(email)
        if stored is None or not hmac.compare_digest(pw.encode(), stored.encode()):
            self._take(f"pw-addr|{email}", 10**9)
            return self._json(401, {"error": "bad_credentials"})
        sub = self.accounts.get(email) or self.subject_for(email)
        if device not in self.known_devices.get(sub, set()):
            if not self._take(f"ip|{svc}|{ip}", START_PER_SERVICE_IP) or not self._take(f"addr|{email}", START_PER_ADDRESS):
                return self._json(429, {"error": "rate_limited"})
            flow_id, fl = self._mail_code(email, svc, "stepup", "de", device=device)
            return self._json(200, {"stepUp": True, "flow": flow_id, "expiresAt": int(fl.expires * 1000), "codeLength": 6})
        return self._json(200, establish(email, svc, "password", fields, device))


_STUB = ("<!doctype html><meta charset=utf-8><title>{title}</title><body style='font:16px system-ui;margin:3rem'>"
         "<h1>{title}</h1><p>{text}</p><p>Dies ist die Fake-Codai-Seite der CourtVision-Demo.</p>")


def main(argv=None) -> int:  # pragma: no cover - Handstart
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Fake Codai ID (nur Entwicklung)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4460)
    ap.add_argument("--code", default=None, help="fester 6-stelliger Code (sonst zufaellig, im Log)")
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--email-claim", action="store_true")
    a = ap.parse_args(argv)
    fake = FakeCodai(tokens={DEFAULT_SERVICE: a.token}, fixed_code=a.code, auto_usernames=True,
                     email_claim={DEFAULT_SERVICE} if a.email_claim else None, announce=print)
    print(f"Fake-Codai auf http://{a.host}:{a.port} - Service-Token fuer CourtVision: CODAI_SERVICE_TOKEN={a.token}")
    uvicorn.run(fake.app, host=a.host, port=a.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
