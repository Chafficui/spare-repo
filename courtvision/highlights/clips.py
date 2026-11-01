"""Szenen, Clip-Plaene (Live, Recap, Spieler-Reel, Hochformat, Saison-Reel) und ffmpeg-Builder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, Field

from ..domain.events import Event, EventType, effective_events
from ..domain.models import Consent, Game, Player
from .camera import CropRect
from .overlay import AI_LABEL, OverlaySpec, apply_consent, build_overlay, overlay_text_lines
from .scoring import (
    CONF_THRESHOLD, History, HighlightContext, build_contexts, detect_and_ones, detect_fastbreaks,
    highlight_score, publishable_confidence,
)

POST_ROLL_S = 2.0
DEFAULT_LEAD_S = 6.0
MAX_LEAD_S = 20.0

Aspect = Literal["16:9", "9:16"]
Players = Union[Mapping[str, Player], Iterable[Player], None]


# --------------------------------------------------------------------- Modelle
class Segment(BaseModel):
    game_id: str
    start: float
    end: float
    event_ids: list[str] = Field(default_factory=list)
    actor_id: Optional[str] = None
    team: Optional[str] = None
    camera: Literal["ball", "actor"] = "ball"
    overlay: Optional[OverlaySpec] = None
    score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


class ClipPlan(BaseModel):
    kind: Literal["live", "recap", "player_reel", "vertical", "season_reel"]
    segments: list[Segment]
    aspect: Aspect = "16:9"
    #: Textueberlagerungen, die auf jedem Clip erscheinen (KI-Kennzeichnung, Art. 50 AI Act)
    overlays: list[str] = Field(default_factory=lambda: [AI_LABEL])
    ai_generated: bool = True

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)


@dataclass
class Scene:
    game_id: str
    event_ids: list[str]
    primary_id: str
    start: float
    end: float
    actor_id: Optional[str]
    team: Optional[str]
    score: float
    kind: str
    featured: list[str] = field(default_factory=list)
    period: int = 0
    clock: Optional[float] = None
    score_after: tuple[int, int] = (0, 0)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _players_map(players: Players) -> dict[str, Player]:
    if players is None:
        return {}
    if isinstance(players, Mapping):
        return dict(players)
    return {p.id: p for p in players}


# ------------------------------------------------------------------ Szenenbau
class SceneBuilder:
    def __init__(self, events: Iterable[Event], game: Optional[Game] = None,
                 history: Optional[History] = None, threshold: float = CONF_THRESHOLD) -> None:
        self.eff = effective_events(events)
        self.by_id = {e.id: e for e in self.eff}
        self.threshold = threshold
        self.ctx = build_contexts(self.eff, game, history)
        self.score = {e.id: highlight_score(e, self.ctx[e.id]) for e in self.eff}
        self.and_ones = detect_and_ones(self.eff)
        self.fast = detect_fastbreaks(self.eff)  # shot -> steal
        self.fast_by_steal = {v: k for k, v in self.fast.items()}
        self.assists = {e.payload.get("shot_id"): e for e in self.eff if e.type == EventType.ASSIST}
        self.possessions = [e for e in self.eff if e.type == EventType.POSSESSION]

    def eligible(self, e: Event) -> bool:
        return (not self.ctx[e.id].injury and publishable_confidence(e, self.threshold)
                and self.score[e.id] > 0)

    def _possession_start(self, e: Event) -> float:
        cands = [p for p in self.possessions
                 if p.payload.get("start_ts", 1e18) <= e.video_ts <= p.payload.get("end_ts", -1) + 0.5]
        if cands:
            start = max(float(p.payload["start_ts"]) for p in cands)
        elif e.clip_start is not None:
            start = e.clip_start
        else:
            start = e.video_ts - DEFAULT_LEAD_S
        return max(0.0, max(start, e.video_ts - MAX_LEAD_S))

    def scene_for_event(self, e: Event) -> Optional[Scene]:
        """Szene, in der ``e`` die Hauptrolle spielt (Wurf inkl. Assist/And-One/Fastbreak)."""
        ids: list[str] = [e.id]
        primary, kind = e, e.type.value.lower()
        actor = e.player_id
        start: Optional[float] = None
        if e.type == EventType.ASSIST:
            shot = self.by_id.get(e.payload.get("shot_id"))
            if shot is None:
                return None
            primary, ids, kind = shot, [shot.id, e.id], "assist"
            actor = e.player_id
        if primary.type == EventType.SHOT:
            if not primary.payload.get("made"):
                return None
            kind = "three" if int(primary.payload.get("value", 2)) == 3 else "two"
            a = self.assists.get(primary.id)
            if a is not None and a.id not in ids:
                ids.append(a.id)
            if primary.id in self.and_ones:
                ao = self.and_ones[primary.id]
                ids += [ao["foul"]] + ao["fts"]
                kind = "and_one"
            if primary.id in self.fast:
                st = self.by_id[self.fast[primary.id]]
                ids.insert(0, st.id)
                start = st.video_ts
                kind = "fastbreak"
        elif primary.type == EventType.STEAL and primary.id in self.fast_by_steal:
            shot = self.by_id[self.fast_by_steal[primary.id]]
            return self.scene_for_event(shot)
        evs = [self.by_id[i] for i in dict.fromkeys(ids)]
        if start is None:
            start = self._possession_start(primary)
        end = max(x.video_ts for x in evs) + POST_ROLL_S
        start = min(start, primary.video_ts)
        if actor is None:
            actor = primary.player_id
        featured = list(dict.fromkeys(x.player_id for x in evs
                                      if x.player_id and x.type in (EventType.SHOT, EventType.ASSIST,
                                                                    EventType.BLOCK, EventType.STEAL)))
        c = self.ctx[primary.id]
        return Scene(
            game_id=primary.game_id, event_ids=list(dict.fromkeys(i for i in ids)), primary_id=primary.id,
            start=start, end=end, actor_id=actor, team=primary.team,
            score=sum(self.score[x.id] for x in evs if x.id in self.score),
            kind=kind, featured=featured, period=primary.period, clock=primary.clock,
            score_after=c.score_after,
        )

    def all_scenes(self) -> list[Scene]:
        scenes: dict[tuple, Scene] = {}
        for e in self.eff:
            if e.type not in (EventType.SHOT, EventType.BLOCK, EventType.STEAL):
                continue
            if not self.eligible(e):
                continue
            if e.type == EventType.STEAL and e.id in self.fast_by_steal:
                continue  # Teil der Fastbreak-Szene des Wurfs
            sc = self.scene_for_event(e)
            if sc and sc.score > 0:
                scenes[tuple(sc.event_ids)] = sc
        return sorted(scenes.values(), key=lambda s: s.start)


def build_scenes(events: Iterable[Event], game: Optional[Game] = None, history: Optional[History] = None,
                 threshold: float = CONF_THRESHOLD) -> list[Scene]:
    """Alle publizierbaren Szenen (chronologisch): ohne Verletzungsszenen und unsichere Events."""
    return SceneBuilder(events, game, history, threshold).all_scenes()


# ------------------------------------------------------------------- Planbau
def _fit(start: float, end: float, lo: float, hi: float) -> tuple[float, float]:
    """Dauer in [lo, hi]: zu lang -> Anfang kuerzen (Abschluss bleibt), zu kurz -> Vorlauf verlaengern."""
    dur = end - start
    if dur > hi:
        return end - hi, end
    if dur < lo:
        start -= lo - dur
        if start < 0:
            end -= start
            start = 0.0
    return start, end


def _overlay_for(scene_actor: Optional[str], score_after: tuple[int, int], clock: Optional[float],
                 period: int, pmap: Mapping[str, Player], logo: Optional[str]) -> OverlaySpec:
    return build_overlay(pmap.get(scene_actor) if scene_actor else None, score_after, clock, period or None, logo)


def _segment(scene: Scene, start: float, end: float, pmap: Mapping[str, Player],
             camera: str = "ball", logo: Optional[str] = None, actor: Optional[str] = None) -> Segment:
    actor = actor or scene.actor_id
    return Segment(
        game_id=scene.game_id, start=start, end=end, event_ids=list(scene.event_ids), actor_id=actor,
        team=scene.team, camera=camera, score=scene.score,
        overlay=_overlay_for(actor, scene.score_after, scene.clock, scene.period, pmap, logo),
    )


def live_clip(scenes: Sequence[Scene], players: Players = None, logo: Optional[str] = None,
              min_s: float = 8.0, max_s: float = 15.0) -> Optional[ClipPlan]:
    """Einzelnes Top-Ereignis, 8-15 s."""
    if not scenes:
        return None
    sc = max(scenes, key=lambda s: s.score)
    a, b = _fit(sc.start, sc.end, min_s, max_s)
    return ClipPlan(kind="live", segments=[_segment(sc, a, b, _players_map(players), logo=logo)])


def _overlaps(a: Scene, b: Scene) -> bool:
    return a.game_id == b.game_id and a.start < b.end and b.start < a.end


def recap(scenes: Sequence[Scene], players: Players = None, logo: Optional[str] = None,
          min_scenes: int = 8, max_scenes: int = 12, min_total: float = 60.0, max_total: float = 120.0,
          per_team_reserve: int = 3, scene_min: float = 6.0, scene_max: float = 14.0) -> Optional[ClipPlan]:
    """Top-Szenen beider Teams (8-12), Gesamtlaenge 60-120 s, chronologisch."""
    if not scenes:
        return None
    pmap = _players_map(players)
    ranked = sorted(scenes, key=lambda s: -s.score)
    chosen: list[Scene] = []

    def take(sc: Scene) -> None:
        if len(chosen) < max_scenes and sc not in chosen and not any(_overlaps(sc, c) for c in chosen):
            chosen.append(sc)

    for team in ("home", "away"):  # beide Teams vertreten
        for sc in [s for s in ranked if s.team == team][:per_team_reserve]:
            take(sc)
    for sc in ranked:
        take(sc)

    def dur(sc: Scene) -> float:
        a, b = _fit(sc.start, sc.end, scene_min, scene_max)
        return b - a

    # zu lang: schwaechste Szenen entfernen (mindestens min_scenes behalten)
    while sum(dur(s) for s in chosen) > max_total and len(chosen) > min_scenes:
        chosen.remove(min(chosen, key=lambda s: s.score))
    durs = {id(s): dur(s) for s in chosen}
    total = sum(durs.values())
    if total > max_total:  # immer noch zu lang: gleichmaessig kuerzen
        f = max_total / total
        durs = {k: v * f for k, v in durs.items()}
    elif total < min_total and chosen:  # zu kurz: Vorlauf gleichmaessig verlaengern
        extra = (min_total + 1.0 - total) / len(chosen)
        durs = {k: v + extra for k, v in durs.items()}
    segs = []
    for sc in sorted(chosen, key=lambda s: (s.game_id, s.start)):
        a, b = _fit(sc.end - durs[id(sc)], sc.end, durs[id(sc)], durs[id(sc)])
        segs.append(_segment(sc, a, b, pmap, logo=logo))
    return ClipPlan(kind="recap", segments=segs)


def player_reel(events: Iterable[Event], player_id: str, game: Optional[Game] = None,
                history: Optional[History] = None, players: Players = None, logo: Optional[str] = None,
                threshold: float = CONF_THRESHOLD, merge_gap_s: float = 0.5) -> Optional[ClipPlan]:
    """Alle Treffer, Assists, Blocks und Steals eines Spielers (variable Laenge, chronologisch)."""
    b = SceneBuilder(events, game, history, threshold)
    pmap = _players_map(players)
    scenes: dict[tuple, Scene] = {}
    for e in b.eff:
        if e.player_id != player_id:
            continue
        if e.type == EventType.SHOT:
            if not e.payload.get("made"):
                continue
        elif e.type not in (EventType.ASSIST, EventType.BLOCK, EventType.STEAL):
            continue
        if b.ctx[e.id].injury or not publishable_confidence(e, threshold):
            continue
        sc = b.scene_for_event(e)
        if sc is None:
            continue
        sc.actor_id = player_id
        scenes[tuple(sc.event_ids)] = sc
    ordered = sorted(scenes.values(), key=lambda s: s.start)
    segs: list[Segment] = []
    for sc in ordered:
        seg = _segment(sc, sc.start, sc.end, pmap, logo=logo, actor=player_id)
        if segs and seg.game_id == segs[-1].game_id and seg.start - segs[-1].end <= merge_gap_s:
            prev = segs[-1]
            prev.end = max(prev.end, seg.end)
            prev.event_ids += [i for i in seg.event_ids if i not in prev.event_ids]
            prev.score += seg.score
            prev.overlay = seg.overlay
        else:
            segs.append(seg)
    return ClipPlan(kind="player_reel", segments=segs) if segs else None


def vertical_clip(scene: Scene, players: Players = None, logo: Optional[str] = None,
                  min_s: float = 10.0, max_s: float = 30.0) -> ClipPlan:
    """9:16-Ausschnitt (10-30 s), Kamera folgt dem Hauptakteur."""
    a, b = _fit(scene.start, scene.end, min_s, max_s)
    return ClipPlan(kind="vertical", aspect="9:16",
                    segments=[_segment(scene, a, b, _players_map(players), camera="actor", logo=logo)])


def season_reel(scenes_by_game: Mapping[str, Sequence[Scene]], players: Players = None,
                logo: Optional[str] = None, min_total: float = 120.0, max_total: float = 180.0,
                target_total: float = 150.0, scene_min: float = 6.0, scene_max: float = 10.0,
                max_per_game: Optional[int] = None) -> Optional[ClipPlan]:
    """Beste Szenen ueber alle Spiele, 2-3 min."""
    pmap = _players_map(players)
    pool: list[Scene] = []
    for gid, lst in scenes_by_game.items():
        top = sorted(lst, key=lambda s: -s.score)
        pool += top[:max_per_game] if max_per_game else top
    pool.sort(key=lambda s: -s.score)
    chosen: list[Scene] = []
    total = 0.0
    for sc in pool:
        a, b = _fit(sc.start, sc.end, scene_min, scene_max)
        if total + (b - a) > max_total or any(_overlaps(sc, c) for c in chosen):
            continue
        chosen.append(sc)
        total += b - a
        if total >= target_total:
            break
    if not chosen:
        return None
    segs = []
    for sc in sorted(chosen, key=lambda s: (s.game_id, s.start)):
        a, b = _fit(sc.start, sc.end, scene_min, scene_max)
        segs.append(_segment(sc, a, b, pmap, logo=logo))
    return ClipPlan(kind="season_reel", segments=segs)


# --------------------------------------------------------------- Freigabefilter
#: Einwilligung je Ausspielkanal (Minderjaehrige zusaetzlich siehe MINOR_EXTRA)
CHANNEL_REQUIREMENTS: dict[str, Consent] = {
    "team": Consent.HIGHLIGHTS,
    "club": Consent.HIGHLIGHTS,
    "public": Consent.HIGHLIGHTS,
    "livestream": Consent.HIGHLIGHTS | Consent.LIVESTREAM,
}
MINOR_EXTRA: dict[str, Consent] = {
    "public": Consent.STATS_PUBLIC,  # oeffentliche Kanaele: nur bei ausdruecklicher Freigabe
    "livestream": Consent.STATS_PUBLIC,
}


def player_allowed(player: Optional[Player], channel: str) -> bool:
    if player is None:
        return False
    req = CHANNEL_REQUIREMENTS.get(channel, CHANNEL_REQUIREMENTS["public"])
    if player.is_minor:
        req = req | MINOR_EXTRA.get(channel, Consent.NONE)
    if Consent.OBJECTED in Consent(player.consent):
        return False
    return all(player.has(f) for f in Consent if f in req and f != Consent.NONE)


def filter_publishable(plans: Iterable[ClipPlan], players: Players, channel: str = "public") -> list[ClipPlan]:
    """Entfernt Szenen, deren Hauptakteur nicht (kanalgenau) eingewilligt hat; leere Plaene entfallen.

    Spieler mit Widerspruch oder ohne HIGHLIGHTS-Einwilligung erscheinen nie als
    Hauptakteur; bei Minderjaehrigen gelten zusaetzlich die Kanalregeln. In den
    verbleibenden Overlays wird der Name ohne Einwilligung entfernt.
    """
    pmap = _players_map(players)
    out: list[ClipPlan] = []
    for plan in plans:
        segs = []
        for seg in plan.segments:
            if seg.actor_id is None or not player_allowed(pmap.get(seg.actor_id), channel):
                continue
            ov = apply_consent(seg.overlay, pmap.get(seg.actor_id)) if seg.overlay else None
            segs.append(seg.model_copy(update={"overlay": ov}))
        if segs:
            out.append(plan.model_copy(update={"segments": segs}))
    return out


# ----------------------------------------------------------------- ffmpeg
def _esc_text(s: str) -> str:
    return (s.replace("\\", "").replace("'", "").replace(":", "\\:").replace("%", "\\%"))


def _lerp_expr(vals: Sequence[float], times: Sequence[float]) -> str:
    """Stueckweise lineare Interpolation als ffmpeg-Ausdruck in ``t``."""
    expr = f"{vals[-1]:.1f}"
    for i in range(len(vals) - 2, -1, -1):
        t0, t1, v0, v1 = times[i], times[i + 1], vals[i], vals[i + 1]
        seg = f"{v0:.1f}+({v1 - v0:.1f})*(t-{t0:.3f})/{max(t1 - t0, 1e-3):.3f}"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return f"if(lt(t,{times[0]:.3f}),{vals[0]:.1f},{expr})"


def _crop_filter(track: Optional[Sequence[CropRect]], fps: float, vertical: bool,
                 src_w: int, src_h: int, max_keys: int = 24) -> str:
    if not track:
        w, h = (int(src_h * 9 / 16) // 2 * 2, src_h) if vertical else (src_w, src_h)
        return f"crop={w}:{h}:{(src_w - w) // 2}:{(src_h - h) // 2}"
    n = len(track)
    idx = sorted({round(i * (n - 1) / max(1, min(max_keys, n) - 1)) for i in range(min(max_keys, n))})
    times = [i / fps for i in idx]
    w, h = track[0].w, track[0].h
    if len(idx) == 1:
        xe, ye = f"{track[0].x:.1f}", f"{track[0].y:.1f}"
    else:
        xe = _lerp_expr([track[i].x for i in idx], times)
        ye = _lerp_expr([track[i].y for i in idx], times)
    return f"crop=w={int(w)}:h={int(h)}:x='{xe}':y='{ye}'"


def render_command(plan: ClipPlan, source_path: Union[str, Mapping[str, str]], out_path: str,
                   crop_track: Optional[Sequence[Optional[Sequence[CropRect]]]] = None, *,
                   fps: float = 25.0, ffmpeg: str = "ffmpeg", with_audio: bool = True,
                   src_size: tuple[int, int] = (3840, 2160)) -> list[str]:
    """ffmpeg-argv zum Rendern des Plans (nicht ausgefuehrt).

    ``source_path``: ein Pfad oder ``{game_id: pfad}``. ``crop_track[i]`` ist die
    Crop-Liste (``virtual_camera``-Ausgabe, ``fps`` Bilder/s ab Segmentbeginn) von
    Segment i; ``None`` = statischer Mittelausschnitt. Jedes Segment bekommt die
    KI-Kennzeichnung ("KI-generiert") sowie Overlay-Texte per ``drawtext``.
    """
    sources = {None: source_path} if isinstance(source_path, str) else dict(source_path)
    inputs = list(dict.fromkeys(sources.values()))
    vertical = plan.aspect == "9:16"
    out_w, out_h = (1080, 1920) if vertical else (1920, 1080)
    parts: list[str] = []
    labels: list[str] = []
    for i, seg in enumerate(plan.segments):
        src = sources.get(seg.game_id, sources.get(None))
        if src is None:
            raise KeyError(f"keine Quelle fuer Spiel {seg.game_id}")
        k = inputs.index(src)
        track = crop_track[i] if crop_track is not None and i < len(crop_track) else None
        chain = [f"[{k}:v]trim=start={seg.start:.3f}:end={seg.end:.3f}", "setpts=PTS-STARTPTS",
                 _crop_filter(track, fps, vertical, *src_size), f"scale={out_w}:{out_h}", "setsar=1"]
        lines = list(plan.overlays)
        if seg.overlay is not None:
            lines += [l for l in overlay_text_lines(seg.overlay) if l not in lines]
        for j, text in enumerate(lines):
            chain.append(f"drawtext=text='{_esc_text(text)}':x=24:y={24 + 44 * j}:fontsize=36:"
                         f"fontcolor=white:box=1:boxcolor=black@0.5")
        parts.append(",".join(chain) + f"[v{i}]")
        labels.append(f"[v{i}]")
        if with_audio:
            parts.append(f"[{k}:a]atrim=start={seg.start:.3f}:end={seg.end:.3f},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[a{i}]")
    n = len(plan.segments)
    if with_audio:
        joined = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{joined}concat=n={n}:v=1:a=1[outv][outa]")
    else:
        parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]")
    cmd = [ffmpeg, "-y", "-hide_banner"]
    for path in inputs:
        cmd += ["-i", path]
    cmd += ["-filter_complex", ";".join(parts), "-map", "[outv]"]
    if with_audio:
        cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-metadata", f"comment={AI_LABEL}", out_path]
    return cmd
