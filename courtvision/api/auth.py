"""Rollenmodell der API -- **PLATZHALTER, kein echtes Auth**.

Die Rolle kommt aus dem Header ``X-Role`` (optional ``X-Team-Id``,
``X-Player-Id``). Jeder Client kann diese Header beliebig setzen. Vor dem
Produktivbetrieb MUSS dieses Modul durch echte Authentifizierung ersetzt werden
(z. B. OIDC/JWT mit Rollen- und Team-Claims, Eltern-/Spieler-Verknuepfung). Alle
Endpunkte beziehen die Identitaet ausschliesslich ueber
:func:`current_principal`; nur diese Funktion (und :func:`principal_from`)
muessen ersetzt werden.

Rollen
------
``hallenwart``   Setup (Stammdaten, Kalibrierung, Start/Ende)
``kampfgericht`` Live-Konsole (Review, Fouls, Spielbericht, Freigabe, Events)
``trainer``      Dashboard; sieht alles ueber das *eigene* Team (``X-Team-Id``)
``spieler``      Spieler/Eltern; sieht die eigenen Daten (``X-Player-Id``)
``fan``          oeffentlich (Standard, wenn kein Header gesetzt ist)
``verband``      Liga-Auswertungen/Export (oeffentliche Sicht auf Einzeldaten)
``admin``        darf alles
``system``       interne Dienste (Edge/Cloud-Pipeline), schreibt Events
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException

ROLES = ("hallenwart", "kampfgericht", "trainer", "spieler", "fan", "verband", "admin", "system")
#: Rollen mit betrieblichem Vollzugriff auf Spieldaten (Setup, Konsole, Pipeline).
OPERATIONAL = frozenset({"admin", "hallenwart", "kampfgericht", "system"})


@dataclass(frozen=True)
class Principal:
    role: str = "fan"
    team_id: Optional[str] = None
    player_id: Optional[str] = None

    @property
    def operational(self) -> bool:
        return self.role in OPERATIONAL

    @property
    def label(self) -> str:
        extra = self.team_id or self.player_id
        return f"{self.role}:{extra}" if extra else self.role


def principal_from(role: Optional[str], team_id: Optional[str] = None,
                   player_id: Optional[str] = None) -> Principal:
    r = (role or "fan").strip().lower()
    if r not in ROLES:
        raise HTTPException(401, f"Unbekannte Rolle '{role}'. Erlaubt: {', '.join(ROLES)}")
    return Principal(r, (team_id or None), (player_id or None))


def current_principal(
    x_role: Optional[str] = Header(None),
    x_team_id: Optional[str] = Header(None),
    x_player_id: Optional[str] = Header(None),
) -> Principal:
    return principal_from(x_role, x_team_id, x_player_id)


def require(*roles: str):
    """FastAPI-Dependency: Rolle muss in ``roles`` sein (``admin`` darf immer)."""
    allowed = set(roles) | {"admin"}

    def dep(p: Principal = Depends(current_principal)) -> Principal:
        if p.role not in allowed:
            raise HTTPException(
                403, f"Rolle '{p.role}' ist hierfuer nicht berechtigt (erforderlich: {', '.join(sorted(allowed))})")
        return p

    return dep
