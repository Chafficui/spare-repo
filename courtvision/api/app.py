"""CourtVision-API (FastAPI).

Start::

    python -m uvicorn courtvision.api.app:app --port 8765       # DB: Umgebungsvariable COURTVISION_DB (Standard :memory:)

**Zugriff:** Rolle per Header ``X-Role`` (+ optional ``X-Team-Id``, ``X-Player-Id``) --
Platzhalter ohne echte Authentifizierung, siehe :mod:`courtvision.api.auth`. Vor dem
Produktivbetrieb durch echtes Auth ersetzen.

Die statische Web-UI (``courtvision/web``) liegt unter ``/``; alle API-Routen haben Vorrang.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..store.eventstore import EventStore
from . import routes_live, routes_players, routes_review, routes_setup
from .state import AppState

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(store: Optional[EventStore] = None, *, scoreboard_n_stable: int = 3) -> FastAPI:
    """App-Factory. ``store`` ist der Event-Store (Standard: in-memory SQLite).

    ``scoreboard_n_stable``: Entprellung des Plausibilitaetsfilters (identische Tafel-Messungen,
    bis ein Wert uebernommen wird).
    """
    store = store if store is not None else EventStore(":memory:")
    state = AppState(store, scoreboard_n_stable=scoreboard_n_stable)
    app = FastAPI(title="CourtVision API", version="1.0",
                  description="Rollenbasierte API (X-Role-Header ist ein PLATZHALTER fuer echtes Auth).")
    app.state.cv = state
    app.state.store = store
    for mod in (routes_setup, routes_live, routes_review, routes_players):
        app.include_router(mod.build(state))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    if WEB_DIR.is_dir():  # zuletzt: API-Routen haben Vorrang
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


app = create_app(EventStore(os.environ.get("COURTVISION_DB", ":memory:")))
