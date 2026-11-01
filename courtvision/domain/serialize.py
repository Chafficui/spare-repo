"""JSON-(De-)Serialisierung der CV-Beobachtungen (Edge-Pipeline -> API).

Format: ``{"kind": "<Klassenname>", <Felder>...}``; ``FrameState.players`` ist eine Liste von
``PlayerState``-Dicts. Namenskollision: das Feld ``Formation.kind`` (``free_throw``/``jump_ball``) heisst im
JSON ``formation_kind``, weil ``kind`` den Beobachtungstyp traegt. ``observation_from_dict`` validiert streng (unbekannte ``kind``/Felder,
fehlende Pflichtfelder, Seiten ``home``/``away``, Koerbe ``left``/``right``) und wirft
``ValueError`` -- die API macht daraus ein 422.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, get_args, get_origin, get_type_hints

from . import observations as _obs
from .observations import (BallTouch, ClockState, Formation, FrameState, Observation, OutOfBounds,
                           PlayerState, Possession, RimMake, ShotRelease)

OBSERVATION_TYPES: dict[str, type] = {c.__name__: c for c in (
    FrameState, Possession, ShotRelease, RimMake, BallTouch, OutOfBounds, Formation, ClockState)}

_SIDES = {"home", "away"}
_SIDE_FIELDS = {"team", "shooter_team", "last_touch_team"}
_BASKETS = {"left", "right"}
_FORMATION_KINDS = {"free_throw", "jump_ball"}
#: Feldname im Dataclass -> Schluessel im JSON (``kind`` ist der Typ-Diskriminator)
_RENAME = {"Formation": {"kind": "formation_kind"}}


def observation_to_dict(o: Observation) -> dict[str, Any]:
    """Beobachtung -> JSON-faehiges Dict mit ``kind``."""
    name = type(o).__name__
    if name not in OBSERVATION_TYPES:
        raise TypeError(f"Keine Beobachtung: {name}")
    d: dict[str, Any] = {"kind": name}
    for f in dataclasses.fields(o):
        v = getattr(o, f.name)
        if f.name == "players":
            v = [dataclasses.asdict(p) for p in v]
        d[_RENAME.get(name, {}).get(f.name, f.name)] = v
    return d


def _is_float(hint: Any) -> bool:
    return hint is float or (get_origin(hint) is not None and float in get_args(hint))


def _is_int(hint: Any) -> bool:
    return hint is int or (get_origin(hint) is not None and int in get_args(hint))


def _build(cls: type, data: Mapping[str, Any], where: str) -> dict[str, Any]:
    hints = get_type_hints(cls, vars(_obs))
    fields = {f.name: f for f in dataclasses.fields(cls)}
    ren = _RENAME.get(cls.__name__, {})
    data = {next((k for k, v in ren.items() if v == key), key): val for key, val in data.items()}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unbekannte Felder {unknown}")
    kw: dict[str, Any] = {}
    for name, f in fields.items():
        if name not in data:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ValueError(f"{where}: Pflichtfeld '{name}' fehlt")
            continue
        v, hint = data[name], hints[name]
        if name == "players":
            continue
        if v is not None:
            if isinstance(v, bool) and not (hint is bool or bool in get_args(hint)) and name != "running":
                raise ValueError(f"{where}: '{name}' darf kein bool sein")
            if _is_float(hint) and isinstance(v, (int, float)) and not isinstance(v, bool):
                v = float(v)
            elif _is_int(hint) and not _is_float(hint) and (isinstance(v, bool) or not isinstance(v, int)):
                raise ValueError(f"{where}: '{name}' muss eine ganze Zahl sein")
            elif _is_float(hint) and not isinstance(v, float):
                raise ValueError(f"{where}: '{name}' muss eine Zahl sein")
        if v is None and f.default is dataclasses.MISSING:
            if not (get_origin(hint) is not None and type(None) in get_args(hint)):
                raise ValueError(f"{where}: '{name}' darf nicht null sein")
        if name in _SIDE_FIELDS and v is not None and v not in _SIDES:
            raise ValueError(f"{where}: '{name}' muss 'home' oder 'away' sein")
        if name == "basket" and v is not None and v not in _BASKETS:
            raise ValueError(f"{where}: 'basket' muss 'left' oder 'right' sein")
        if name == "kind" and cls is Formation and v not in _FORMATION_KINDS:
            raise ValueError(f"{where}: Formation.kind muss 'free_throw' oder 'jump_ball' sein")
        kw[name] = v
    return kw


def player_state_from_dict(d: Mapping[str, Any]) -> PlayerState:
    if not isinstance(d, Mapping):
        raise ValueError("PlayerState muss ein Objekt sein")
    return PlayerState(**_build(PlayerState, d, "PlayerState"))


def observation_from_dict(d: Mapping[str, Any]) -> Observation:
    """JSON-Dict (mit ``kind``) -> Beobachtung. ``ValueError`` bei ungueltigen Eingaben."""
    if not isinstance(d, Mapping):
        raise ValueError("Beobachtung muss ein Objekt sein")
    kind = d.get("kind")
    cls = OBSERVATION_TYPES.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise ValueError(f"Unbekannte kind '{kind}' (erlaubt: {', '.join(OBSERVATION_TYPES)})")
    data = {k: v for k, v in d.items() if k != "kind"}
    kw = _build(cls, data, kind)
    if cls is FrameState:
        players = data.get("players", [])
        if not isinstance(players, (list, tuple)):
            raise ValueError("FrameState.players muss eine Liste sein")
        kw["players"] = tuple(player_state_from_dict(p) for p in players)
    return cls(**kw)
