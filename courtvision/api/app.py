"""CourtVision-API (FastAPI).

Start::

    python -m uvicorn courtvision.api.app:app --port 8765       # DB: Umgebungsvariable COURTVISION_DB (Standard :memory:)

**Zugriff:** Anmeldung ueber Codai ID (Sitzungs-Cookie, ``/auth/*``) bzw. Bearer-Token fuer Maschinen-Clients, siehe
:mod:`courtvision.api.auth` und :mod:`courtvision.api.routes_auth`. Die Platzhalter-Header ``X-Role`` ... gelten nur mit
``COURTVISION_AUTH_MODE=dev`` (Tests, lokale Entwicklung).

**Status-Lebenszyklus eines Spiels:** ``scheduled`` --POST /start--> ``live`` --POST /end--> ``processing``
--POST /processed--> ``review`` --POST /finalize--> ``final``. Pro Live-Spiel haelt der Server eine
:class:`courtvision.live.GameSession` (Tafel- und Beobachtungs-Ingest, Perioden-Events); ``/end`` ruft
``session.finish()``. Die Korrektur-Endpunkte (``/review``, ``/fouls``, ``/scoresheet``, ``/finalize``) arbeiten in
``processing`` und ``review``; ``final`` ist gesperrt (nur Admin). Kalibrierung, Medien-URLs, Einwilligungs-Audit
und DSFA-Status liegen als Dokumente im Store (``put_doc``) und ueberleben einen Neustart.

Die statische Web-UI (``courtvision/web``) liegt unter ``/``; alle API-Routen haben Vorrang.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..auth.codai import CodaiClient
from ..auth.config import AuthConfig
from ..auth.service import AuthService
from ..auth.session import CSRF_COOKIE, CSRF_HEADER, SESSION_COOKIE
from ..store.eventstore import EventStore
from . import routes_auth, routes_demo, routes_live, routes_players, routes_review, routes_setup
from .auth import Principal, resolve_principal
from .state import AppState

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


UNSAFE = frozenset({"POST", "PUT", "PATCH", "DELETE"})
#: Seiten ohne Inline-Skripte/-Styles bekommen eine strenge CSP (die aelteren Seiten haben noch Inline-Skripte).
STRICT_CSP_PAGES = frozenset({"/login.html", "/admin.html"})
STRICT_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
              "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def create_app(store: Optional[EventStore] = None, *, scoreboard_n_stable: int = 3,
               auth_mode: Optional[str] = None, auth_config: Optional[AuthConfig] = None,
               codai_client: Optional[CodaiClient] = None, fake_codai=None) -> FastAPI:
    """App-Factory. ``store`` ist der Event-Store (Standard: in-memory SQLite).

    ``scoreboard_n_stable``: Entprellung des Plausibilitaetsfilters (identische Tafel-Messungen,
    bis ein Wert uebernommen wird).

    Anmeldung: ``auth_config`` (Standard: aus den Umgebungsvariablen, :meth:`AuthConfig.from_env`); ``auth_mode``
    ueberschreibt nur den Modus (``"codai"`` | ``"dev"``). ``codai_client`` ersetzt den HTTP-Client zu Codai;
    ``fake_codai`` (:class:`courtvision.auth.fake_codai.FakeCodai`) wird unter ``/_fake_codai`` eingehaengt und, falls
    kein Client angegeben ist, ohne Netzwerk im Prozess angesprochen (Demo).
    """
    store = store if store is not None else EventStore(":memory:")
    cfg = auth_config if auth_config is not None else AuthConfig.from_env()
    if auth_mode is not None:
        cfg = cfg.with_(mode=auth_mode)
    if fake_codai is not None and codai_client is None:
        from fastapi.testclient import TestClient

        from ..auth.fake_codai import DEFAULT_SERVICE

        http = TestClient(fake_codai.app, base_url="http://fake-codai")
        codai_client = CodaiClient("http://fake-codai", fake_codai.tokens[DEFAULT_SERVICE], cfg.codai_timeout_s,
                                   client_id=cfg.client_id, http=http)
    state = AppState(store, scoreboard_n_stable=scoreboard_n_stable)
    auth = AuthService(store, cfg, codai_client)
    app = FastAPI(title="CourtVision API", version="1.0",
                  description="Rollenbasierte API; Anmeldung ueber Codai ID (Sitzungs-Cookie) oder Bearer-Token.")
    app.state.cv = state
    app.state.store = store
    app.state.auth = auth
    for mod in (routes_setup, routes_live, routes_review, routes_players, routes_demo):
        app.include_router(mod.build(state))
    app.include_router(routes_auth.build(state, auth))

    def deny(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse({"detail": {"code": code, "message": message}}, status_code=status)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """CSRF (Double-Submit, an die Sitzung gebunden) fuer zustandsaendernde Browser-Anfragen + Sicherheits-Header."""
        if request.method in UNSAFE:
            bearer = request.headers.get("authorization", "")[:7].lower() == "bearer "
            # Ohne Cookies gibt es keine "ambient authority", die ein fremder Link ausnutzen koennte; /auth/* immer.
            ambient = SESSION_COOKIE in request.cookies or CSRF_COOKIE in request.cookies
            if not bearer and (request.url.path.startswith("/auth/") or ambient) and (not cfg.is_dev or ambient):
                if not auth.same_origin(request):
                    return deny(403, "csrf", "Anfrage von fremder Herkunft abgelehnt.")
                sid, _ = auth.session_for(request.cookies)
                cookie = request.cookies.get(CSRF_COOKIE)
                header = request.headers.get(CSRF_HEADER)
                if not (cookie and header and hmac.compare_digest(cookie, header)
                        and auth.signer.csrf_valid(header, sid or "")):
                    return deny(403, "csrf", "Sicherheitsprüfung fehlgeschlagen (CSRF). Bitte lade die Seite neu.")
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if request.url.path in STRICT_CSP_PAGES:
            response.headers["Content-Security-Policy"] = STRICT_CSP
            response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request: Request, exc: StarletteHTTPException):
        # Nicht angemeldet + 403 -> 401 (die Oberflaeche leitet dann zur Anmeldung)
        if exc.status_code == 403:
            try:
                p = resolve_principal(request)
            except StarletteHTTPException:
                p = None
            if isinstance(p, Principal) and p.source == "anon":
                exc = StarletteHTTPException(401, exc.detail)
        return await http_exception_handler(request, exc)

    if fake_codai is not None:
        app.mount("/_fake_codai", fake_codai.app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    if WEB_DIR.is_dir():  # zuletzt: API-Routen haben Vorrang
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


app = create_app(EventStore(os.environ.get("COURTVISION_DB", ":memory:")))
