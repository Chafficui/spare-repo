"""Einwilligungsverwaltung je Spieler mit Audit-Trail.

Regeln (Produktplan "Recht & Datenschutz"):

* Erwachsene: berechtigtes Interesse (Standard ``Consent.adult_default()``) mit Widerspruch (Opt-out).
* Minderjaehrige: ohne Elterneinwilligung gilt ``OBJECTED`` (Overlay ohne Namen, keine Highlights,
  Stats nur als Team-Wert). Ab ``child_age`` (Standard 14; unbekanntes Alter wird vorsichtig als
  "ab Schwelle" behandelt) ist zusaetzlich die eigene Einwilligung des Kindes noetig (``child_consent``).
* Jede Aenderung wird als :class:`ConsentRecord` protokolliert.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

from pydantic import BaseModel

from ..domain.models import Consent, Game, Player

DEFAULT_CHILD_AGE = 14


class ConsentRecord(BaseModel):
    ts: float
    player_id: str
    actor: str  # "player" | "parent" | "club" | "system"
    action: str  # grant | revoke | object | withdraw_objection | parental_consent | child_consent | ...
    level: str = ""
    before: int
    after: int  # wirksame Einwilligung (nach Regeln)
    note: str = ""


class _Profile(BaseModel):
    age: Optional[int] = None
    granted: int = 0  # gewaehrte Stufen (Minderjaehrige starten bei 0, Erwachsene bei adult_default)
    objected: bool = False
    parental_consent: bool = False
    child_consent: bool = False


class ConsentManager:
    def __init__(self, players: Iterable[Player] = (), child_age: int = DEFAULT_CHILD_AGE,
                 clock: Callable[[], float] = time.time) -> None:
        self.child_age = child_age
        self._clock = clock
        self.players: dict[str, Player] = {}
        self._profiles: dict[str, _Profile] = {}
        self._log: list[ConsentRecord] = []
        for p in players:
            self.register(p)

    # ---------------------------------------------------------------- Setup
    def register(self, player: Player, age: Optional[int] = None) -> None:
        self.players[player.id] = player
        prof = _Profile(age=age)
        if player.is_minor:
            prof.granted = 0
        else:
            prof.granted = int(Consent.adult_default())
        self._profiles[player.id] = prof
        self._sync(player.id)

    def set_age(self, player_id: str, age: int, actor: str = "club") -> None:
        before = self.effective(player_id)
        self._profiles[player_id].age = age
        self._sync(player_id)
        self._record(player_id, actor, "set_age", f"age={age}", before)

    # ------------------------------------------------------------ Regeln
    def needs_child_consent(self, player_id: str) -> bool:
        p, prof = self.players[player_id], self._profiles[player_id]
        return p.is_minor and (prof.age is None or prof.age >= self.child_age)

    def effective(self, player_id: str) -> Consent:
        p, prof = self.players[player_id], self._profiles[player_id]
        if prof.objected:
            return Consent.OBJECTED
        if p.is_minor:
            if not prof.parental_consent or (self.needs_child_consent(player_id) and not prof.child_consent):
                return Consent.OBJECTED
        return Consent(prof.granted)

    def _sync(self, player_id: str) -> None:
        self.players[player_id].consent = int(self.effective(player_id))

    def _record(self, player_id: str, actor: str, action: str, level: str, before: Consent,
                note: str = "") -> ConsentRecord:
        rec = ConsentRecord(ts=self._clock(), player_id=player_id, actor=actor, action=action, level=level,
                            before=int(before), after=int(self.effective(player_id)), note=note)
        self._log.append(rec)
        return rec

    def _change(self, player_id: str, actor: str, action: str, level: str, fn: Callable[[_Profile], None],
                note: str = "") -> ConsentRecord:
        before = self.effective(player_id)
        fn(self._profiles[player_id])
        self._sync(player_id)
        return self._record(player_id, actor, action, level, before, note)

    # --------------------------------------------------------------- Aktionen
    def grant(self, player_id: str, level: Consent, actor: str = "player", note: str = "") -> ConsentRecord:
        def f(pr: _Profile) -> None:
            pr.granted |= int(level)
        return self._change(player_id, actor, "grant", level.name or str(int(level)), f, note)

    def revoke(self, player_id: str, level: Consent, actor: str = "player", note: str = "") -> ConsentRecord:
        def f(pr: _Profile) -> None:
            pr.granted &= ~int(level)
        return self._change(player_id, actor, "revoke", level.name or str(int(level)), f, note)

    def object_(self, player_id: str, actor: str = "player", note: str = "") -> ConsentRecord:
        """Widerspruch (Opt-out)."""
        def f(pr: _Profile) -> None:
            pr.objected = True
        return self._change(player_id, actor, "object", "OBJECTED", f, note)

    def withdraw_objection(self, player_id: str, actor: str = "player", note: str = "") -> ConsentRecord:
        def f(pr: _Profile) -> None:
            pr.objected = False
        return self._change(player_id, actor, "withdraw_objection", "OBJECTED", f, note)

    def grant_parental(self, player_id: str, levels: Consent, note: str = "") -> ConsentRecord:
        """Elterneinwilligung (Minderjaehrige) inklusive der gewaehrten Stufen."""
        def f(pr: _Profile) -> None:
            pr.parental_consent = True
            pr.granted |= int(levels)
        return self._change(player_id, "parent", "parental_consent", str(levels), f, note)

    def revoke_parental(self, player_id: str, note: str = "") -> ConsentRecord:
        def f(pr: _Profile) -> None:
            pr.parental_consent = False
        return self._change(player_id, "parent", "parental_consent_revoked", "", f, note)

    def grant_child(self, player_id: str, note: str = "") -> ConsentRecord:
        """Eigene Einwilligung des Kindes (ab Verstandesreife / ``child_age``)."""
        def f(pr: _Profile) -> None:
            pr.child_consent = True
        return self._change(player_id, "player", "child_consent", "", f, note)

    def revoke_child(self, player_id: str, note: str = "") -> ConsentRecord:
        def f(pr: _Profile) -> None:
            pr.child_consent = False
        return self._change(player_id, "player", "child_consent_revoked", "", f, note)

    # ---------------------------------------------------------------- Audit
    def load_audit(self, player_id: str, records: Iterable[ConsentRecord]) -> None:
        """Gespeicherte Audit-Eintraege eines Spielers wiederherstellen (ersetzt vorhandene)."""
        self._log = [r for r in self._log if r.player_id != player_id] + list(records)

    def audit_trail(self, player_id: Optional[str] = None) -> list[ConsentRecord]:
        return [r for r in self._log if player_id is None or r.player_id == player_id]


# --------------------------------------------------------- Sichtbarkeitsregeln
def can_show_stats_public(player: Player) -> bool:
    return player.has(Consent.STATS_PUBLIC)


def team_only_stats(player: Player) -> bool:
    """True: nur als Team-Wert ausweisen (Widerspruch, oder nicht einmal interne Stats erlaubt)."""
    return not player.has(Consent.STATS_INTERNAL)


def public_ratings_allowed(player: Player) -> bool:
    """Ratings werden nie fuer Minderjaehrige oeffentlich; sonst nur mit oeffentlicher Stats-Einwilligung."""
    return (not player.is_minor) and player.has(Consent.STATS_PUBLIC)


def livestream_allowed(game: Game, players: Iterable[Player]) -> bool:
    """Livestream nur, wenn alle Spieler beider Teams des Spiels LIVESTREAM-Einwilligung haben."""
    teams = {game.home_team_id, game.away_team_id}
    involved = [p for p in players if p.team_id in teams]
    return bool(involved) and all(p.has(Consent.LIVESTREAM) for p in involved)


def livestream_blockers(game: Game, players: Iterable[Player]) -> list[str]:
    teams = {game.home_team_id, game.away_team_id}
    return [p.id for p in players if p.team_id in teams and not p.has(Consent.LIVESTREAM)]
