"""Spielernoten: PIR, EFF, Game Score, Bayes-Shrinkage, Perzentil-Note 1,0-10,0.

Pipeline der Spielnote (siehe Produktplan, "Eigene Spielnote"):

1. Rohwert: Game Score pro 40 Minuten, Bayes-Shrinkage zur Liga-Mitte::

       per40 = (GmSc + liga_mittel_per40 * prior_min / 40) / ((min + prior_min) / 40)

   Entspricht einem Beobachtungsprior von ``prior_min`` Minuten mit Ligamittel.
   Bei 0 Minuten ergibt das genau das Ligamittel, bei vielen Minuten die
   normale Rate GmSc * 40 / min.
2. Normierung: Perzentil in der Verteilung (Liga + Altersklasse, 12 Monate),
   abgebildet stückweise linear: p=0 -> 1,0; p=0,5 -> 6,0; p=1 -> 10,0.
3. Kontext: kleiner Zu-/Abschlag (Plus/Minus, Wurfschwierigkeit), gedeckelt
   auf +-0,5 Notenpunkte, auf die Note addiert; Ergebnis auf [1, 10] begrenzt.
4. Konfidenz: ``provisional`` markiert Noten mit unbestätigten Events.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from typing import Iterable, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from .boxscore import BoxScore, PlayerLine

#: Fallback bei dünner Datenlage (< MIN_SAMPLES): Normalverteilung von GmSc/40.
FALLBACK_MEAN_PER40 = 12.0
FALLBACK_SD_PER40 = 8.0
MIN_SAMPLES = 20
CONTEXT_CAP = 0.5


# ---------------------------------------------------------------- Rohformeln
def pir(line: PlayerLine) -> int:
    """PIR = (PTS+REB+AST+STL+BLK+FD) - (FGA-FGM) - (FTA-FTM) - TO - BLKA - PF."""
    return (
        line.pts + line.reb + line.ast + line.stl + line.blk + line.fd
        - (line.fga - line.fgm) - (line.fta - line.ftm) - line.tov - line.blka - line.pf
    )


def eff(line: PlayerLine) -> int:
    """Klassisches EFF = PTS+REB+AST+STL+BLK - (FGA-FGM) - (FTA-FTM) - TO."""
    return (
        line.pts + line.reb + line.ast + line.stl + line.blk
        - (line.fga - line.fgm) - (line.fta - line.ftm) - line.tov
    )


def game_score(line: PlayerLine) -> float:
    """Hollinger Game Score.

    GmSc = PTS + 0.4 FGM - 0.7 FGA - 0.4 (FTA-FTM) + 0.7 ORB + 0.3 DRB
           + STL + 0.7 AST + 0.7 BLK - 0.4 PF - TO
    """
    return (
        line.pts + 0.4 * line.fgm - 0.7 * line.fga - 0.4 * (line.fta - line.ftm)
        + 0.7 * line.orb + 0.3 * line.drb + line.stl + 0.7 * line.ast + 0.7 * line.blk
        - 0.4 * line.pf - line.tov
    )


def gmsc_per40(line: PlayerLine, league_mean_per40: float, prior_minutes: float = 8.0) -> float:
    """GmSc pro 40 Minuten mit Bayes-Shrinkage (Formel im Modul-Docstring)."""
    minutes = line.seconds_played / 60.0
    return (game_score(line) + league_mean_per40 * prior_minutes / 40.0) / ((minutes + prior_minutes) / 40.0)


def context_adjustment(
    plus_minus_per40: float,
    shot_difficulty: Optional[float] = None,
    *,
    pm_weight: float = 0.02,
    difficulty_weight: float = 0.6,
    cap: float = CONTEXT_CAP,
) -> float:
    """Zu-/Abschlag in Notenpunkten, begrenzt auf +-``cap`` (Standard 0,5).

    ``pm_weight`` Notenpunkte je Plus/Minus-Punkt pro 40 Minuten (+25 -> +0,5).
    ``shot_difficulty`` in [0, 1] (0,5 = ligatypisch, höher = schwerere Würfe):
    ``difficulty_weight * (d - 0.5)``; None = kein Beitrag.
    """
    adj = pm_weight * plus_minus_per40
    if shot_difficulty is not None:
        adj += difficulty_weight * (min(1.0, max(0.0, shot_difficulty)) - 0.5)
    return max(-cap, min(cap, adj))


# ------------------------------------------------------------------ Verteilung
class LeagueDistribution(BaseModel):
    """GmSc/40-Werte einer Liga + Altersklasse als (Datum, Wert)-Paare."""

    samples: list[tuple[date, float]] = Field(default_factory=list)
    window_days: int = 365

    def add(self, day: date, value: float) -> None:
        self.samples.append((day, value))

    def values(self, as_of: Optional[date] = None) -> list[float]:
        """Sortierte Werte der letzten ``window_days`` bis ``as_of`` (Standard: jüngstes Sample)."""
        if not self.samples:
            return []
        end = as_of or max(d for d, _ in self.samples)
        start = end - timedelta(days=self.window_days)
        return sorted(v for d, v in self.samples if start < d <= end)

    def size(self, as_of: Optional[date] = None) -> int:
        return len(self.values(as_of))

    def mean(self, as_of: Optional[date] = None) -> Optional[float]:
        vals = self.values(as_of)
        return sum(vals) / len(vals) if vals else None

    def percentile(self, value: float, as_of: Optional[date] = None) -> float:
        """Perzentil in [0, 1] (Mid-Rank bei Gleichstand); 0,5 bei leerer Verteilung."""
        vals = self.values(as_of)
        if not vals:
            return 0.5
        lo, hi = bisect_left(vals, value), bisect_right(vals, value)
        return (lo + 0.5 * (hi - lo)) / len(vals)


def normal_percentile(value: float, mean: float = FALLBACK_MEAN_PER40, sd: float = FALLBACK_SD_PER40) -> float:
    return 0.5 * (1.0 + math.erf((value - mean) / (sd * math.sqrt(2.0))))


def to_grade(p: float) -> float:
    """Perzentil -> Note: p=0 -> 1,0; p=0,5 -> 6,0; p=1 -> 10,0 (linear je Hälfte), 1 Nachkommastelle."""
    p = min(1.0, max(0.0, p))
    g = 1.0 + 10.0 * p if p <= 0.5 else 6.0 + 8.0 * (p - 0.5)
    return round(g, 1)


# ------------------------------------------------------------------------ Note
class Grade(BaseModel):
    value: float = Field(ge=1.0, le=10.0)
    raw_gmsc: float
    per40: float  # geshrinkte GmSc/40
    pir: int
    eff: int
    provisional: bool = False
    percentile: float = 0.5
    adjustment: float = 0.0
    minutes: float = 0.0


def grade_player(
    line: PlayerLine,
    distribution: Optional[LeagueDistribution] = None,
    *,
    provisional: bool = False,
    plus_minus_per40: Optional[float] = None,
    shot_difficulty: Optional[float] = None,
    as_of: Optional[date] = None,
    prior_minutes: float = 8.0,
) -> Grade:
    """Note 1,0-10,0 für eine Spielerzeile.

    Hat die Verteilung weniger als ``MIN_SAMPLES`` (20) Werte, gelten Fallback-
    Ligamittel 12 und eine Normalverteilung (sd 8) für GmSc/40. Plus/Minus pro
    40 Minuten wird aus der Zeile abgeleitet, falls nicht übergeben.
    """
    dist = distribution or LeagueDistribution()
    enough = dist.size(as_of) >= MIN_SAMPLES
    mean = dist.mean(as_of) if enough else FALLBACK_MEAN_PER40
    per40 = gmsc_per40(line, mean, prior_minutes)
    p = dist.percentile(per40, as_of) if enough else normal_percentile(per40)
    if plus_minus_per40 is None:
        plus_minus_per40 = line.plus_minus * 2400.0 / line.seconds_played if line.seconds_played > 0 else 0.0
    adj = context_adjustment(plus_minus_per40, shot_difficulty)
    value = round(min(10.0, max(1.0, to_grade(p) + adj)), 1)
    return Grade(
        value=value, raw_gmsc=game_score(line), per40=per40, pir=pir(line), eff=eff(line),
        provisional=provisional, percentile=p, adjustment=adj, minutes=line.seconds_played / 60.0,
    )


def grade_boxscore(
    box: BoxScore,
    distributions: LeagueDistribution | Mapping[str, LeagueDistribution] | None = None,
    provisional_threshold: int = 1,
) -> dict[str, Grade]:
    """Noten aller Spieler mit Einsatzzeit.

    ``distributions``: eine Verteilung für alle, oder Mapping mit Schlüsseln
    "home"/"away" bzw. "default". "vorläufig", wenn ``provisional_events >=
    provisional_threshold`` (Standard 1) oder der Box-Score provisorisch ist.
    """
    out: dict[str, Grade] = {}
    for pid, line in box.players.items():
        if line.seconds_played <= 0:
            continue
        if isinstance(distributions, Mapping):
            dist = distributions.get(line.team) or distributions.get("default")
        else:
            dist = distributions
        prov = line.provisional_events >= provisional_threshold or box.status == "provisional"
        out[pid] = grade_player(line, dist, provisional=prov)
    return out


# --------------------------------------------------------------------- Profil
#: Skalierung der Radar-Achsen (Wert am Cap = 100, linear ab 0).
RADAR_CAPS = {
    "scoring": 30.0,        # Punkte pro 40 min
    "rebounding": 14.0,     # Rebounds pro 40 min
    "playmaking": 10.0,     # Assists pro 40 min
    "defense": 5.0,         # (STL+BLK) pro 40 min
    "ball_security": 3.0,   # AST/TOV-Verhältnis
}
TS_RANGE = (0.35, 0.70)  # TS% 35 % -> 0, 70 % -> 100


class Profile(BaseModel):
    player_id: Optional[str] = None
    games: int = 0
    minutes: float = 0.0
    average_grade: float = 0.0
    form: list[float] = Field(default_factory=list)  # letzte 5 Noten, ältestes zuerst
    bests: dict[str, float] = Field(default_factory=dict)
    radar: dict[str, float] = Field(default_factory=dict)


def _scale(x: float, cap: float) -> float:
    return round(max(0.0, min(100.0, 100.0 * x / cap)), 1)


def season_profile(lines: Sequence[PlayerLine], grades: Sequence[float], player_id: Optional[str] = None) -> Profile:
    """Saisonprofil: Durchschnittsnote, Form (letzte 5), Bestwerte, Radar (0-100).

    Radar aus aufsummierten Zeilen: Scoring = PTS/40 (Cap 30); Effizienz = TS%
    linear zwischen 35 % und 70 %; Rebounding = REB/40 (Cap 14); Playmaking =
    AST/40 (Cap 10); Defense = (STL+BLK)/40 (Cap 5); Ballsicherheit = AST/TOV
    (Cap 3; TOV=0 -> 100 bei AST>0, sonst 50).
    """
    lines, grades = list(lines), list(grades)
    sec = sum(l.seconds_played for l in lines)
    tot = lambda f: sum(getattr(l, f) for l in lines)  # noqa: E731
    per40 = (lambda f: tot(f) * 2400.0 / sec) if sec > 0 else (lambda f: 0.0)
    fga, fta = tot("fga"), tot("fta")
    den = 2.0 * (fga + 0.44 * fta)
    ts = tot("pts") / den if den else None
    lo, hi = TS_RANGE
    eff_axis = 0.0 if ts is None else round(max(0.0, min(100.0, 100.0 * (ts - lo) / (hi - lo))), 1)
    ast, tov = tot("ast"), tot("tov")
    if tov:
        sec_axis = _scale(ast / tov, RADAR_CAPS["ball_security"])
    else:
        sec_axis = 100.0 if ast > 0 else (50.0 if lines else 0.0)
    radar = {
        "scoring": _scale(per40("pts"), RADAR_CAPS["scoring"]),
        "efficiency": eff_axis,
        "rebounding": _scale(per40("reb"), RADAR_CAPS["rebounding"]),
        "playmaking": _scale(per40("ast"), RADAR_CAPS["playmaking"]),
        "defense": _scale((tot("stl") + tot("blk")) * 2400.0 / sec if sec > 0 else 0.0, RADAR_CAPS["defense"]),
        "ball_security": sec_axis,
    }
    bests = {
        "pts": float(max((l.pts for l in lines), default=0)),
        "reb": float(max((l.reb for l in lines), default=0)),
        "ast": float(max((l.ast for l in lines), default=0)),
        "grade": float(max(grades, default=0.0)),
    }
    return Profile(
        player_id=player_id, games=len(lines), minutes=sec / 60.0,
        average_grade=round(sum(grades) / len(grades), 2) if grades else 0.0,
        form=grades[-5:], bests=bests, radar=radar,
    )


def leaderboard(
    profiles: Iterable[Profile],
    metric: str = "average_grade",
    *,
    minors: set[str],
    limit: Optional[int] = None,
    min_games: int = 1,
) -> list[Profile]:
    """Liga-Bestenliste, absteigend. Minderjährige (``minors``) sind ausgeschlossen.

    ``metric``: "average_grade", ein Radar-Achsenname oder ein Schlüssel aus ``bests``.
    Profile ohne ``player_id`` werden ausgelassen.
    """
    def key(p: Profile) -> float:
        if metric == "average_grade":
            return p.average_grade
        if metric in p.radar:
            return p.radar[metric]
        if metric in p.bests:
            return p.bests[metric]
        raise KeyError(metric)

    rows = [p for p in profiles if p.player_id and p.player_id not in minors and p.games >= min_games]
    rows.sort(key=key, reverse=True)
    return rows[:limit] if limit else rows
