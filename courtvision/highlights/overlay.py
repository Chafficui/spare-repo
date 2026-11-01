"""Overlay-Spezifikation (Name, Nummer, Spielstand, Uhr, Logo) mit Einwilligungslogik."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ..domain.models import Consent, Player

#: Kennzeichnung KI-generierter Inhalte (EU AI Act Art. 50).
AI_LABEL = "KI-generiert"


class OverlaySpec(BaseModel):
    player_id: Optional[str] = None
    number: Optional[int] = None
    name: Optional[str] = None  # None => nur Nummer
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    clock: Optional[str] = None
    period: Optional[int] = None
    logo: Optional[str] = None  # Pfad/URL des Vereinslogos
    ai_label: str = AI_LABEL


def name_allowed(player: Player) -> bool:
    """Name nur bei Einwilligung (oeffentliche Stats + Highlights) und ohne Widerspruch."""
    return player.has(Consent.HIGHLIGHTS) and player.has(Consent.STATS_PUBLIC)


def format_clock(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def build_overlay(player: Optional[Player], score: Optional[tuple[int, int]] = None,
                  clock: Optional[float] = None, period: Optional[int] = None,
                  logo: Optional[str] = None) -> OverlaySpec:
    return OverlaySpec(
        player_id=player.id if player else None,
        number=player.number if player else None,
        name=player.name if player is not None and name_allowed(player) else None,
        score_home=score[0] if score else None, score_away=score[1] if score else None,
        clock=format_clock(clock), period=period, logo=logo,
    )


def apply_consent(spec: OverlaySpec, player: Optional[Player]) -> OverlaySpec:
    """Entfernt den Namen, falls der Spieler (inzwischen) keine Einwilligung hat."""
    if spec.name is not None and (player is None or not name_allowed(player)):
        return spec.model_copy(update={"name": None})
    return spec


def overlay_text_lines(spec: OverlaySpec) -> list[str]:
    """Textzeilen fuers Rendern; die KI-Kennzeichnung ist immer enthalten."""
    lines = [spec.ai_label]
    if spec.number is not None:
        lines.append(f"#{spec.number} {spec.name}" if spec.name else f"#{spec.number}")
    if spec.score_home is not None and spec.score_away is not None:
        lines.append(f"{spec.score_home}:{spec.score_away}")
    if spec.clock is not None:
        lines.append(f"Q{spec.period} {spec.clock}" if spec.period else spec.clock)
    return lines
