"""Rollenmodell und Identitaet der API.

Die Identitaet kommt (in dieser Reihenfolge) aus

1. ``Authorization: Bearer <Token>`` -- Maschinen-Clients (Edge/CV-Pipeline) mit einem Token aus
   ``COURTVISION_SYSTEM_TOKENS``; ergibt die Rolle ``system``;
2. dem Sitzungs-Cookie ``cv_session`` (Login ueber Codai ID, siehe :mod:`courtvision.auth`); die aktive Rolle ist eine
   der **Mitgliedschaften** des Benutzers (Sitzung oder Header ``X-Active-Role`` [+ ``X-Active-Team``/``X-Active-Player``],
   serverseitig geprueft);
3. **nur mit** ``COURTVISION_AUTH_MODE=dev``: den Platzhalter-Headern ``X-Role``/``X-Team-Id``/``X-Player-Id``
   (jeder Client kann sich jede Rolle geben -- nie im Betrieb verwenden).

Ohne Anmeldung ist man oeffentlicher ``fan``; geschuetzte Endpunkte antworten dann mit 401 (angemeldet, aber ohne
Berechtigung: 403). Alle Endpunkte beziehen die Identitaet ueber :func:`current_principal`.

Rollen
------
``hallenwart``   Setup (Stammdaten, Kalibrierung, Start/Ende)
``kampfgericht`` Live-Konsole (Review, Fouls, Spielbericht, Freigabe, Events)
``trainer``      Dashboard; sieht alles ueber das *eigene* Team (Mitgliedschaft mit ``team_id``)
``spieler``      Spieler/Eltern; sieht die eigenen Daten (Mitgliedschaft mit ``player_id``)
``fan``          oeffentlich (Standard)
``verband``      Liga-Auswertungen/Export (oeffentliche Sicht auf Einzeldaten)
``admin``        darf alles
``system``       interne Dienste (Edge/Cloud-Pipeline), schreibt Events
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request
from starlette.requests import HTTPConnection

ROLES = ("hallenwart", "kampfgericht", "trainer", "spieler", "fan", "verband", "admin", "system")
#: Rollen mit betrieblichem Vollzugriff auf Spieldaten (Setup, Konsole, Pipeline).
OPERATIONAL = frozenset({"admin", "hallenwart", "kampfgericht", "system"})


@dataclass(frozen=True)
class Principal:
    role: str = "fan"
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    #: Codai-Subject (angemeldete Benutzer), ``system`` fuer Token-Clients.
    subject: Optional[str] = None
    username: Optional[str] = None
    #: ``session`` | ``token`` | ``dev`` (Platzhalter-Header) | ``anon`` (nicht angemeldet, Codai-Modus)
    source: str = "dev"

    @property
    def authenticated(self) -> bool:
        return self.source in ("session", "token")

    @property
    def operational(self) -> bool:
        return self.role in OPERATIONAL

    @property
    def label(self) -> str:
        extra = self.team_id or self.player_id
        return f"{self.role}:{extra}" if extra else self.role


ANON = Principal(source="anon")


def principal_from(role: Optional[str], team_id: Optional[str] = None,
                   player_id: Optional[str] = None) -> Principal:
    """Platzhalter-Header (Dev-Modus)."""
    r = (role or "fan").strip().lower()
    if r not in ROLES:
        raise HTTPException(401, f"Unbekannte Rolle '{role}'. Erlaubt: {', '.join(ROLES)}")
    return Principal(r, (team_id or None), (player_id or None), source="dev")


def _bearer(conn: HTTPConnection) -> Optional[str]:
    h = conn.headers.get("authorization", "")
    return h[7:].strip() if h[:7].lower() == "bearer " else None


def resolve_principal(conn: HTTPConnection, *, allow_headers: bool = True) -> Principal:
    """Identitaet einer HTTP-/WebSocket-Verbindung. Wirft 401/403 bei ungueltigem Token bzw. unzulaessiger Rolle."""
    auth = getattr(conn.app.state, "auth", None)
    if auth is None:  # App ohne Auth-Dienst: Dev-Verhalten
        h = conn.headers
        return principal_from(h.get("x-role"), h.get("x-team-id"), h.get("x-player-id"))
    tok = _bearer(conn)
    if tok is not None:
        if auth.system_token_ok(tok):
            return Principal("system", subject="system", username="system", source="token")
        raise HTTPException(401, "Ungültiges Token")
    sid, session = auth.session_for(conn.cookies)
    if session is not None:
        user = auth.users.get(session["subject"])
        if user is not None:
            h = conn.headers
            req_role = h.get("x-active-role")
            m = auth.resolve_membership(user, session, req_role, h.get("x-active-team"), h.get("x-active-player"))
            if m is None:
                raise HTTPException(403, f"Die Rolle '{req_role}' gehört nicht zu deinen Mitgliedschaften")
            return Principal(m["role"], m.get("team_id"), m.get("player_id"), user["subject"],
                             user.get("username"), "session")
    if auth.cfg.is_dev and allow_headers:
        h = conn.headers
        return principal_from(h.get("x-role"), h.get("x-team-id"), h.get("x-player-id"))
    return ANON


def current_principal(request: Request) -> Principal:
    p = getattr(request.state, "principal", None)
    if p is None:
        p = resolve_principal(request)
        request.state.principal = p
    return p


def require(*roles: str):
    """FastAPI-Dependency: Rolle muss in ``roles`` sein (``admin`` darf immer)."""
    allowed = set(roles) | {"admin"}

    def dep(p: Principal = Depends(current_principal)) -> Principal:
        if p.role not in allowed:
            if p.source == "anon":
                raise HTTPException(401, "Bitte anmelden")
            raise HTTPException(
                403, f"Rolle '{p.role}' ist hierfuer nicht berechtigt (erforderlich: {', '.join(sorted(allowed))})")
        return p

    return dep
