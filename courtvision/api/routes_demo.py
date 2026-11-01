"""Demo: ein simuliertes Spiel durch die komplette Live-Kette schicken (nur Admin).

``POST /games/{id}/simulate`` fuehrt :func:`courtvision.sim.simulate_game` fuer die Kader des Spiels aus und
speist Beobachtungen + Tafelstrom in die ``GameSession`` des Spiels. Die Wahrheits-Fouls (nicht optisch
beobachtbar) werden als CONSOLE-FOUL-Events zu ihren Zeitstempeln eingestreut, damit Fouls/Freiwuerfe im
Box-Score erscheinen. ``speed=null``: synchron, danach ``finish()`` (Status processing). ``speed>0``: Wiedergabe in einem
Hintergrund-Thread mit diesem Echtzeitfaktor -- die Live-Oberflaechen laufen per WebSocket mit.
"""
from __future__ import annotations

import heapq
import threading
import time
from typing import Any, Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..domain.events import Event, EventSource, EventStatus, EventType
from ..domain.models import GameStatus
from ..sim import simulate_game
from .auth import Principal, require
from .state import AppState, LiveSlot

#: Der Simulator liefert die Tafel mit 1 Hz; mit 3 gleichen Lesungen gingen aufeinanderfolgende Freiwuerfe
#: als Sprung durch (siehe ``courtvision.live.session``). Die Demo-Session entprellt daher mit 2.
DEMO_N_STABLE = 2


class SimulateIn(BaseModel):
    seed: int = 1
    periods: int = Field(4, ge=1, le=8)
    period_length_s: float = Field(600.0, ge=30.0, le=1200.0)
    include_frames: bool = True  # Positionen (10 Hz) nötig für Wechsel / Minuten
    speed: Optional[float] = Field(None, gt=0.0, le=1000.0)  # None: synchron; sonst Echtzeitfaktor


def _timeline(sim) -> Iterator[tuple[float, str, Any]]:
    """Beobachtungen, Tafelwerte und Konsolen-Fouls nach Zeit gemischt (bei Gleichstand: Tafel, Beobachtung, Foul)."""
    sb = ((ts, 0, i, ("sb", (ts, d))) for i, (ts, d) in enumerate(sim.scoreboard))
    ob = ((o.video_ts, 1, i, ("obs", o)) for i, o in enumerate(sim.observations))
    fouls = sorted((e for e in sim.truth if e.type == EventType.FOUL), key=lambda e: e.video_ts)
    fl = ((e.video_ts, 2, i, ("foul", e)) for i, e in enumerate(fouls))
    for ts, _, _, (kind, item) in heapq.merge(sb, ob, fl, key=lambda t: t[:3]):
        yield ts, kind, item


def _feed(st: AppState, slot: LiveSlot, kind: str, item: Any) -> None:
    with slot.lock:
        if kind == "sb":
            slot.session.feed_scoreboard(item)
        elif kind == "obs":
            slot.session.feed_observation(item)
        else:
            e: Event = item
            st.store.append(e.derive(supersedes=None, source=EventSource.CONSOLE,
                                     status=EventStatus.CONFIRMED, confidence=1.0))


def build(st: AppState) -> APIRouter:
    r = APIRouter()

    def run(slot: LiveSlot, sim, speed: Optional[float], state: dict[str, Any]) -> None:
        try:
            t0, start = time.monotonic(), None
            for ts, kind, item in _timeline(sim):
                if speed is not None:
                    start = ts if start is None else start
                    wait = t0 + (ts - start) / speed - time.monotonic()
                    if wait > 0.002:
                        time.sleep(wait)
                    if state.get("cancel"):
                        break
                _feed(st, slot, kind, item)
                state["video_ts"] = ts
            with slot.lock:
                slot.session.finish()
        except Exception as e:  # noqa: BLE001 (Fehler im Hintergrund-Thread sichtbar machen)
            state["error"] = f"{type(e).__name__}: {e}"
        finally:
            state["running"] = False

    @r.post("/games/{game_id}/simulate")
    def simulate(game_id: str, body: SimulateIn, _: Principal = Depends(require())):
        g = st.game(game_id)
        if g.status != GameStatus.SCHEDULED:
            raise HTTPException(409, f"Nur geplante Spiele können simuliert werden (Status '{g.status.value}')")
        rosters = st.rosters(g)
        for side in ("home", "away"):
            if len(rosters[side]) < 5:
                raise HTTPException(400, f"Kader '{side}' hat weniger als 5 Spieler")
        g = g.model_copy(update={"periods": body.periods, "period_length_s": body.period_length_s})
        st.store.put_game(g)
        sim = simulate_game(body.seed, g, rosters, periods=body.periods, period_length_s=body.period_length_s,
                            include_frames=body.include_frames)
        starters = {side: next(e for e in sim.truth if e.type == EventType.STARTERS and e.team == side).payload["players"]
                    for side in ("home", "away")}
        # Wechselzonen: gespeicherte Kalibrierung hat Vorrang, sonst die des Simulators
        cal = st.calibration(game_id)
        st.start_game(g, starters, explicit_start=False, n_stable=DEMO_N_STABLE,
                      calibration=None if cal and cal.get("zones") else sim.zones)
        slot = st.slot(game_id)
        state: dict[str, Any] = {"running": True, "speed": body.speed, "duration_s": sim.duration,
                                 "video_ts": 0.0, "final_score": sim.final_score}
        slot.simulation = state
        info = {"game_id": game_id, "seed": body.seed, "periods_played": sim.periods_played,
                "duration_s": sim.duration, "final_score": sim.final_score, "speed": body.speed}
        if body.speed is None:
            run(slot, sim, None, state)
            if state.get("error"):
                raise HTTPException(500, f"Simulation fehlgeschlagen: {state['error']}")
            return {**info, "mode": "sync", "game": st.game(game_id).model_dump(mode="json"),
                    "n_events": len(st.store.events(game_id))}
        threading.Thread(target=run, args=(slot, sim, body.speed, state), daemon=True,
                         name=f"simulate-{game_id}").start()
        return {**info, "mode": "replay", "game": st.game(game_id).model_dump(mode="json"),
                "message": f"Wiedergabe mit Faktor {body.speed:g} läuft (ca. {sim.duration / body.speed:.0f} s)"}

    @r.get("/games/{game_id}/simulate")
    def simulate_status(game_id: str, _: Principal = Depends(require())):
        st.game(game_id)
        slot = st._slots.get(game_id)  # noqa: SLF001
        return (slot.simulation if slot and slot.simulation else {"running": False})

    return r
