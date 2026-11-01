"""Aufbewahrungsfristen fuer Rohvideo und fluechtige biometrische Merkmale."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable

from ..domain.models import Consent, Player


@dataclass(frozen=True)
class RetentionPolicy:
    default_days: int = 90
    #: Frist, wenn ALLE Spieler des Spiels TRAINING_DATA eingewilligt haben
    training_days: int = 730


@dataclass
class Recording:
    id: str
    game_id: str
    recorded_at: datetime
    players: list[Player] = field(default_factory=list)  # alle Spieler beider Teams


def retention_days(rec: Recording, policy: RetentionPolicy = RetentionPolicy()) -> int:
    """90 Tage; laenger nur bei TRAINING_DATA-Einwilligung aller Spieler (wird bei Widerruf wieder 90)."""
    if rec.players and all(p.has(Consent.TRAINING_DATA) for p in rec.players):
        return max(policy.default_days, policy.training_days)
    return policy.default_days


def due_for_deletion(recordings: Iterable[Recording], now: datetime,
                     policy: RetentionPolicy = RetentionPolicy()) -> list[Recording]:
    return [r for r in recordings if now - r.recorded_at >= timedelta(days=retention_days(r, policy))]


# ------------------------------------------------- fluechtige Biometrie (Re-ID)
_biometric_stores: list[Callable[[], None]] = []


def register_biometric_store(purge: Callable[[], None]) -> None:
    """Registriert einen Speicher fluechtiger Re-ID-Merkmale (z. B. Embedding-Galerie).

    Re-ID-Merkmale duerfen nur im Arbeitsspeicher und nur fuer die Dauer eines Spiels existieren
    (Art. 9 DSGVO; nie ueber Spiele hinweg gespeichert, nie persistiert). Die CV-Pipeline
    registriert hier ihre Galerien; ``purge_transient_biometrics`` wird bei Spielende
    (GAME_END/GAME_FINALIZED) und bei Abbruch/Neustart aufgerufen.
    """
    _biometric_stores.append(purge)


def purge_transient_biometrics() -> int:
    """Leert alle registrierten Re-ID-Speicher; liefert die Zahl der geleerten Speicher."""
    n = 0
    for purge in list(_biometric_stores):
        purge()
        n += 1
    return n
