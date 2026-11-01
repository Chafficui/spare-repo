"""Fiktive Demo-Stammdaten (Venue, Teams, Spieler, Spiel) fuer Demos und Tests."""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import Consent, Game, Player, Team, Venue

_HOME = [
    (4, "Jonas Berger"), (5, "Luca Hartmann"), (7, "Felix Brandt"), (8, "Tim Kessler"),
    (9, "Marek Vogel"), (10, "Paul Lindner"), (11, "Niklas Roth"), (12, "Ben Albrecht"),
    (13, "Samuel Krueger"), (14, "Eric Sommer"), (15, "Jan Winkler"), (23, "Moritz Falk"),
]
_AWAY = [
    (3, "Leon Schaefer"), (6, "Max Reinhardt"), (7, "David Neumann"), (8, "Kilian Busch"),
    (9, "Tobias Engel"), (10, "Robin Voigt"), (11, "Oskar Lange"), (12, "Henri Kaiser"),
    (14, "Dominik Wolff"), (15, "Fabian Metz"), (21, "Nico Arnold"), (24, "Julian Stein"),
]


@dataclass
class DemoSetup:
    venue: Venue
    home: Team
    away: Team
    players: dict[str, list[Player]]  # team_id -> roster
    game: Game

    @property
    def rosters(self) -> dict[str, list[Player]]:
        """Rosters keyed by side ("home"/"away") for ``simulate_game``."""
        return {"home": self.players[self.home.id], "away": self.players[self.away.id]}

    def all_players(self) -> list[Player]:
        return [p for r in self.players.values() for p in r]


def _roster(team_id: str, prefix: str, data: list[tuple[int, str]], minor_idx: int | None) -> list[Player]:
    out = []
    for i, (num, name) in enumerate(data):
        kw = {}
        if i == minor_idx:
            # minderjaehrig, Eltern haben nur interne Statistik erlaubt
            kw = dict(is_minor=True, consent=int(Consent.STATS_INTERNAL))
        out.append(Player(id=f"{prefix}{num}", team_id=team_id, number=num, name=name, **kw))
    return out


def demo_setup(store_like=None) -> DemoSetup:
    venue = Venue(id="venue-dessau", name="Anhalt-Arena Dessau", city="Dessau-Rosslau", scoreboard_source="second_camera")
    home = Team(id="team-dessau", name="TV Dessau Herren 1", club="TV Dessau", league="Oberliga", jersey_color="#1f4fa3")
    away = Team(id="team-halle", name="BC Halle Herren 2", club="BC Halle", league="Oberliga", jersey_color="#e8e8e8")
    players = {
        home.id: _roster(home.id, "dsx", _HOME, minor_idx=10),  # Nr. 15, jugendlicher Spieler
        away.id: _roster(away.id, "hal", _AWAY, None),
    }
    game = Game(id="demo-game-1", venue_id=venue.id, home_team_id=home.id, away_team_id=away.id,
                league="Oberliga", start_time="2026-03-14T18:00:00+01:00")
    setup = DemoSetup(venue, home, away, players, game)
    if store_like is not None:
        load_demo(store_like, setup)
    return setup


def load_demo(store, setup: DemoSetup | None = None) -> DemoSetup:
    """Schreibt die Demo-Stammdaten in einen EventStore."""
    setup = setup or demo_setup()
    store.put_venue(setup.venue)
    store.put_team(setup.home)
    store.put_team(setup.away)
    for p in setup.all_players():
        store.put_player(p)
    store.put_game(setup.game)
    return setup
