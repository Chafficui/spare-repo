"""Spielfeldlinien als JSON (Meter, Ursprung Feldmitte) zum Zeichnen im Browser."""
from __future__ import annotations

import math

from ..geometry.court import (BASKET_X, FT_LINE_FROM_BASELINE, HALF_L, HALF_W, LANDMARKS, LANE_WIDTH,
                              RESTRICTED_RADIUS, THREE_CORNER_Y, THREE_RADIUS, three_point_line)

CENTER_CIRCLE_R = 1.80
FT_CIRCLE_R = 1.80
BASKET_R = 0.225
BACKBOARD_HALF = 0.90
BACKBOARD_FROM_BASELINE = 1.20


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40) -> list[list[float]]:
    return [[round(cx + r * math.cos(a0 + (a1 - a0) * i / n), 4),
             round(cy + r * math.sin(a0 + (a1 - a0) * i / n), 4)] for i in range(n + 1)]


def court_geometry() -> dict:
    lines: dict[str, list] = {
        "outline": [[-HALF_L, -HALF_W], [HALF_L, -HALF_W], [HALF_L, HALF_W], [-HALF_L, HALF_W], [-HALF_L, -HALF_W]],
        "midline": [[0, -HALF_W], [0, HALF_W]],
        "center_circle": _arc(0, 0, CENTER_CIRCLE_R, 0, 2 * math.pi, 64),
    }
    three: dict[str, list] = {}
    baskets: dict[str, list] = {}
    for name, s in (("left", -1), ("right", 1)):
        ftx = s * (HALF_L - FT_LINE_FROM_BASELINE)
        w = LANE_WIDTH / 2
        lines[f"lane_{name}"] = [[s * HALF_L, -w], [ftx, -w], [ftx, w], [s * HALF_L, w]]
        toward = math.pi if s == 1 else 0.0  # Richtung Feldmitte
        lines[f"ft_circle_solid_{name}"] = _arc(ftx, 0, FT_CIRCLE_R, toward - math.pi / 2, toward + math.pi / 2)
        lines[f"ft_circle_dashed_{name}"] = _arc(ftx, 0, FT_CIRCLE_R, toward + math.pi / 2, toward + 3 * math.pi / 2)
        bx = s * BASKET_X
        lines[f"restricted_{name}"] = _arc(bx, 0, RESTRICTED_RADIUS, toward - math.pi / 2, toward + math.pi / 2)
        lines[f"basket_{name}"] = _arc(bx, 0, BASKET_R, 0, 2 * math.pi, 24)
        bbx = s * (HALF_L - BACKBOARD_FROM_BASELINE)
        lines[f"backboard_{name}"] = [[bbx, -BACKBOARD_HALF], [bbx, BACKBOARD_HALF]]
        three[name] = [[round(x, 4), round(y, 4)] for x, y in three_point_line(name)]
        baskets[name] = [bx, 0.0]
    return {
        "units": "m", "origin": "center", "length": 2 * HALF_L, "width": 2 * HALF_W,
        "x_range": [-HALF_L, HALF_L], "y_range": [-HALF_W, HALF_W],
        "three_radius": THREE_RADIUS, "three_corner_y": THREE_CORNER_Y,
        "lines": lines, "three_point": three, "baskets": baskets,
        "landmarks": {k: [v[0], v[1]] for k, v in LANDMARKS.items()},
    }
