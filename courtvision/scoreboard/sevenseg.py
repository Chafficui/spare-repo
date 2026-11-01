"""7-Segment-OCR in reinem numpy.

Bilder sind 2D-Arrays (Graustufen, float 0..1 oder uint8); RGB wird gemittelt.
Annahme: helle Ziffern auf dunklem Grund (``FieldLayout.invert`` dreht um).

Segmentbezeichnung::

     aaa
    f   b
     ggg
    e   c
     ddd
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from .state import ScoreboardReading

SEGMENTS = "abcdefg"
DIGIT_SEGMENTS: dict[int, str] = {
    0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg",
    5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
}
#: Musterzeilen: Index 0..9 = Ziffern, Index 10 = leer.
PATTERNS = np.array(
    [[1 if s in DIGIT_SEGMENTS[d] else 0 for s in SEGMENTS] for d in range(10)]
    + [[0] * 7],
    dtype=np.int8,
)
BLANK = 10

DEFAULT_SIZE = (40, 24)  # (Höhe, Breite) eines Ziffernausschnitts


# --------------------------------------------------------------------------
# Synthetische Ziffern
# --------------------------------------------------------------------------
def _geometry(h: int, w: int, t_frac: float) -> dict[str, tuple[int, int, int, int]]:
    """Rechtecke (y0, y1, x0, x1) der sieben Segmente."""
    mx = max(1, round(0.1 * w))
    my = max(1, round(0.06 * h))
    t = max(2, round(t_frac * w))
    x0, x1, y0, y1, ym = mx, w - mx, my, h - my, h // 2
    gy = ym - t // 2
    return {
        "a": (y0, y0 + t, x0, x1),
        "d": (y1 - t, y1, x0, x1),
        "g": (gy, gy + t, x0, x1),
        "f": (y0, ym, x0, x0 + t),
        "b": (y0, ym, x1 - t, x1),
        "e": (ym, y1, x0, x0 + t),
        "c": (ym, y1, x1 - t, x1),
    }


def render_digit(
    d: Optional[int],
    size: tuple[int, int] = DEFAULT_SIZE,
    segment_thickness: float = 0.18,
    noise: float = 0.0,
    blur: float = 0.0,
    on_level: float = 1.0,
    off_level: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    flicker: float = 0.0,
    background: Optional[float] = None,
    dim_prob: float = 0.0,
) -> np.ndarray:
    """Synthetisches 7-Segment-Ziffernbild (float32, 0..1).

    d: 0..9, ``None`` = leer. segment_thickness: Anteil der Breite.
    off_level: Helligkeit unbeleuchteter Segmente (Geisterbild), background:
    Grundhelligkeit (Default = off_level). flicker: beleuchtete Segmente werden
    mit U(1-flicker, 1) skaliert (LED-Flimmern); dim_prob: Wahrscheinlichkeit,
    dass ein beleuchtetes Segment nur teilweise (30-70 %) leuchtet.
    """
    rng = rng or np.random.default_rng()
    h, w = size
    img = np.full((h, w), off_level if background is None else background, dtype=np.float32)
    geo = _geometry(h, w, segment_thickness)
    lit = DIGIT_SEGMENTS.get(d, "") if d is not None else ""
    for s in SEGMENTS:
        y0, y1, x0, x1 = geo[s]
        if s in lit:
            level = on_level
            if flicker > 0:
                level *= rng.uniform(1 - flicker, 1.0)
            if dim_prob > 0 and rng.random() < dim_prob:
                level = off_level + (on_level - off_level) * rng.uniform(0.3, 0.7)
        else:
            level = off_level
        img[y0:y1, x0:x1] = level
    if blur > 0:
        img = ndimage.gaussian_filter(img, blur)
    if noise > 0:
        img = img + rng.normal(0.0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return a


# --------------------------------------------------------------------------
# Klassifikatoren
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DigitResult:
    digit: Optional[int]  # None = leer
    confidence: float


def otsu_threshold(values: np.ndarray) -> tuple[float, float]:
    """Otsu über wenige Werte. Rückgabe: (Schwelle, Kontrast = mean_hi - mean_lo)."""
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = v.size
    if n < 2 or v[-1] - v[0] < 1e-9:
        return float(v.mean()) if n else 0.5, 0.0
    best, best_i = -1.0, 1
    for i in range(1, n):
        m0, m1 = v[:i].mean(), v[i:].mean()
        var = (i / n) * ((n - i) / n) * (m0 - m1) ** 2
        if var > best:
            best, best_i = var, i
    thr = 0.5 * (v[best_i - 1] + v[best_i])
    return float(thr), float(v[best_i:].mean() - v[:best_i].mean())


class SegmentClassifier:
    """Mittelt die Helligkeit der 7 Segmentbereiche, schwellt adaptiv (Otsu) und
    ordnet das Segmentmuster per kleinster Hamming-Distanz einer Ziffer zu."""

    def __init__(self, thickness: float = 0.16, min_contrast: float = 0.2,
                 abs_threshold: float = 0.5):
        self.thickness = thickness
        self.min_contrast = min_contrast
        self.abs_threshold = abs_threshold

    def segment_means(self, crop: np.ndarray) -> np.ndarray:
        g = _to_gray(crop)
        h, w = g.shape
        t = max(2.0, self.thickness * w)
        mx, my = max(1, round(0.1 * w)), max(1, round(0.06 * h))
        x0, x1, y0, y1, ym = mx, w - mx, my, h - my, h / 2
        hx, hy = 0.2 * w, max(1.0, 0.04 * h)  # horizontale Segmente
        vx, vy = max(1.0, 0.3 * t), 0.12 * h  # vertikale Segmente
        centers = {
            "a": (w / 2, y0 + t / 2, hx, hy), "d": (w / 2, y1 - t / 2, hx, hy),
            "g": (w / 2, ym, hx, hy),
            "f": (x0 + t / 2, (y0 + ym) / 2, vx, vy), "b": (x1 - t / 2, (y0 + ym) / 2, vx, vy),
            "e": (x0 + t / 2, (ym + y1) / 2, vx, vy), "c": (x1 - t / 2, (ym + y1) / 2, vx, vy),
        }
        out = np.empty(7, dtype=np.float64)
        for i, s in enumerate(SEGMENTS):
            cx, cy, ax, ay = centers[s]
            xa, xb = int(max(0, np.floor(cx - ax))), int(min(w, np.ceil(cx + ax)))
            ya, yb = int(max(0, np.floor(cy - ay))), int(min(h, np.ceil(cy + ay)))
            out[i] = g[ya:max(yb, ya + 1), xa:max(xb, xa + 1)].mean()
        return out

    def classify_many(self, crops: Sequence[np.ndarray]) -> list[DigitResult]:
        """Alle Ziffern eines Feldes mit gemeinsamer Schwelle (stabiler als einzeln)."""
        means = np.stack([self.segment_means(c) for c in crops])
        thr, contrast = otsu_threshold(means)
        if contrast < self.min_contrast:  # nur dunkel oder nur hell: absolute Schwelle
            thr, contrast = self.abs_threshold, 1.0
        return [self._match(m, thr, contrast) for m in means]

    def classify(self, crop: np.ndarray) -> DigitResult:
        return self.classify_many([crop])[0]

    @staticmethod
    def _match(means: np.ndarray, thr: float, contrast: float) -> DigitResult:
        bits = (means > thr).astype(np.int8)
        dist = (PATTERNS != bits).sum(axis=1)
        order = np.argsort(dist, kind="stable")
        d1, d2 = int(dist[order[0]]), int(dist[order[1]])
        soft = min(1.0, 2.0 * float(np.min(np.abs(means - thr))) / max(contrast, 1e-6))
        conf = min(1.0, (d2 - d1) / 2.0 + (0.5 if d1 == 0 else 0.0)) * soft
        if d1 > 0:
            conf *= 0.5 / (1 + d1)  # kein gültiges Muster
        best = int(order[0])
        return DigitResult(None if best == BLANK else best, float(conf))


class TemplateClassifier:
    """Kleiner, synthetisch trainierter Klassifikator (Softmax-Regression, numpy).

    Der Ausschnitt wird auf ``shape`` skaliert; 11 Klassen (0-9, leer).
    """

    def __init__(self, shape: tuple[int, int] = (20, 12)):
        self.shape = shape
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def _features(self, crop: np.ndarray) -> np.ndarray:
        g = _to_gray(crop)
        z = (self.shape[0] / g.shape[0], self.shape[1] / g.shape[1])
        return ndimage.zoom(g, z, order=1).ravel()[: self.shape[0] * self.shape[1]]

    @classmethod
    def train_synthetic(cls, n_per_class: int = 300, seed: int = 0, iters: int = 400,
                        lr: float = 0.5, l2: float = 1e-3,
                        shape: tuple[int, int] = (20, 12)) -> "TemplateClassifier":
        self = cls(shape)
        rng = np.random.default_rng(seed)
        X, y = [], []
        for cls_id in range(11):
            for _ in range(n_per_class):
                hgt = int(rng.integers(30, 61))
                size = (hgt, int(round(hgt * rng.uniform(0.5, 0.7))))
                on = rng.uniform(0.6, 1.0)
                off = rng.uniform(0.0, 0.3)
                img = render_digit(
                    None if cls_id == BLANK else cls_id, size,
                    segment_thickness=rng.uniform(0.12, 0.25),
                    noise=rng.uniform(0, 0.3), blur=rng.uniform(0, 1.0),
                    on_level=on, off_level=off, rng=rng, flicker=rng.uniform(0, 0.3),
                    background=rng.uniform(0.0, off + 0.05), dim_prob=0.1,
                )
                X.append(self._features(img))
                y.append(cls_id)
        X, y = np.array(X), np.array(y)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-3
        Xs = (X - self.mu) / self.sd
        n, dim = Xs.shape
        W = np.zeros((dim, 11))
        b = np.zeros(11)
        Y = np.eye(11)[y]
        for _ in range(iters):
            p = _softmax(Xs @ W + b)
            gW = Xs.T @ (p - Y) / n + l2 * W
            W -= lr * gW
            b -= lr * (p - Y).mean(0)
        self.W, self.b = W, b
        return self

    def classify(self, crop: np.ndarray) -> DigitResult:
        assert self.W is not None, "nicht trainiert"
        x = (self._features(crop) - self.mu) / self.sd
        p = _softmax((x @ self.W + self.b)[None, :])[0]
        k = int(np.argmax(p))
        return DigitResult(None if k == BLANK else k, float(p[k]))

    def classify_many(self, crops: Sequence[np.ndarray]) -> list[DigitResult]:
        return [self.classify(c) for c in crops]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Felder und Layout
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldLayout:
    """Ein Feld: feste Region (x, y, w, h) in Pixeln mit n Ziffernplätzen.

    fmt: "int" (n Ziffern, führende Leerstellen erlaubt), "mm:ss" (4 Ziffern),
    "ss.t" (3 Ziffern, Zehntel). slot_fracs: optionale (x0, x1)-Anteile je Platz
    (z. B. um den Doppelpunkt auszusparen); Default: gleichmässig.
    """

    roi: tuple[int, int, int, int]
    n_slots: int
    fmt: str = "int"
    slot_fracs: Optional[tuple[tuple[float, float], ...]] = None
    invert: bool = False

    def slots(self) -> list[tuple[int, int, int, int]]:
        """Pixelrechtecke (y0, y1, x0, x1) der Ziffernplätze im Gesamtbild."""
        x, y, w, h = self.roi
        fr = self.slot_fracs or tuple(
            (i / self.n_slots, (i + 1) / self.n_slots) for i in range(self.n_slots))
        return [(y, y + h, x + int(round(a * w)), x + int(round(b * w))) for a, b in fr]


@dataclass
class FieldResult:
    value: Optional[float]
    confidence: float
    digits: list[DigitResult] = field(default_factory=list)


class FieldReader:
    """Liest ein Feld; mittelt die letzten k Frames gegen LED-Flimmern."""

    def __init__(self, layout: FieldLayout, classifier=None, k: int = 3, mode: str = "mean"):
        self.layout = layout
        self.classifier = classifier or SegmentClassifier()
        self.mode = mode
        self.buf: deque[np.ndarray] = deque(maxlen=max(1, k))

    def reset(self) -> None:
        self.buf.clear()

    def push(self, image: np.ndarray) -> None:
        g = _to_gray(image)
        x, y, w, h = self.layout.roi
        crop = g[y:y + h, x:x + w]
        self.buf.append(1.0 - crop if self.layout.invert else crop)

    def _combined(self) -> np.ndarray:
        stack = np.stack(list(self.buf))
        if self.mode == "max":
            return stack.max(axis=0)
        if self.mode == "median":
            return np.median(stack, axis=0)
        return stack.mean(axis=0)

    def read(self, image: Optional[np.ndarray] = None) -> FieldResult:
        if image is not None:
            self.push(image)
        if not self.buf:
            return FieldResult(None, 0.0)
        comb = self._combined()
        x, y, w, h = self.layout.roi
        crops = [comb[:, int(round(a * w)):max(int(round(b * w)), int(round(a * w)) + 1)]
                 for a, b in self._fracs()]
        digs = self.classifier.classify_many(crops)
        return self._decode(digs)

    def _fracs(self):
        n = self.layout.n_slots
        return self.layout.slot_fracs or tuple((i / n, (i + 1) / n) for i in range(n))

    def _decode(self, digs: list[DigitResult]) -> FieldResult:
        conf = min(d.confidence for d in digs)
        vals = [d.digit for d in digs]
        fail = FieldResult(None, conf * 0.0, digs)
        fmt = self.layout.fmt
        if fmt == "int":
            idx = next((i for i, v in enumerate(vals) if v is not None), None)
            if idx is None or any(v is None for v in vals[idx:]):
                return fail
            return FieldResult(float(int("".join(str(v) for v in vals[idx:]))), conf, digs)
        if fmt == "mm:ss" and len(vals) == 4:
            m10, m1, s10, s1 = vals
            if s10 is None or s1 is None or s10 > 5:
                return fail
            if m1 is None and m10 is not None:
                return fail
            return FieldResult(float((m10 or 0) * 600 + (m1 or 0) * 60 + s10 * 10 + s1), conf, digs)
        if fmt == "ss.t" and len(vals) == 3:
            s10, s1, t = vals
            if s1 is None or t is None or (s10 is not None and s10 > 5):
                return fail
            return FieldResult((s10 or 0) * 10 + s1 + t / 10.0, conf, digs)
        return fail


#: Anteil gesättigter Pixel im Feld, ab dem Blendung/Spiegelung angenommen wird.
GLARE_RATIO = 0.6
POLARIZER_HINT = ("Starke Blendung/Spiegelung auf der Anzeigetafel erkannt: "
                  "Polfilter vor der Kamera drehen oder Kamerawinkel/Position ändern.")


@dataclass
class ScoreboardLayout:
    """Zuordnung Feldname -> FieldLayout. Namen: home, away, clock, period,
    fouls_home, fouls_away; optional ``clock_alt`` (z. B. "ss.t" unter 1 min)."""

    fields: dict[str, FieldLayout]


class ScoreboardOCR:
    """Liest ein ganzes Tafelbild in ein :class:`ScoreboardReading`."""

    def __init__(self, layout: ScoreboardLayout, classifier=None, k: int = 3,
                 mode: str = "mean", stopped_after_s: float = 1.5, source: str = "ocr"):
        self.layout = layout
        self.readers = {n: FieldReader(fl, classifier, k, mode) for n, fl in layout.fields.items()}
        self.stopped_after_s = stopped_after_s
        self.source = source
        self._prev_clock: Optional[float] = None
        self._change_ts = 0.0
        self._running = False
        self.hints: list[str] = []

    def reset(self) -> None:
        for r in self.readers.values():
            r.reset()
        self._prev_clock = None

    def glare_ratio(self, image: np.ndarray) -> float:
        g = _to_gray(image)
        worst = 0.0
        for fl in self.layout.fields.values():
            x, y, w, h = fl.roi
            c = g[y:y + h, x:x + w]
            worst = max(worst, float((c >= 0.98).mean()))
        return worst

    def setup_hints(self, image: np.ndarray) -> list[str]:
        return [POLARIZER_HINT] if self.glare_ratio(image) > GLARE_RATIO else []

    def read(self, image: np.ndarray, video_ts: float = 0.0) -> ScoreboardReading:
        self.hints = self.setup_hints(image)
        res = {n: r.read(image) for n, r in self.readers.items()}
        vals: dict = {}
        conf: dict[str, float] = {}
        for n in ("home", "away", "period", "fouls_home", "fouls_away"):
            r = res.get(n)
            if r is not None and r.value is not None:
                vals[n] = int(r.value)
                conf[n] = r.confidence
        main, alt = res.get("clock"), res.get("clock_alt")
        pick = None
        for c in (main, alt):
            if c is not None and c.value is not None and (pick is None or c.confidence > pick.confidence + 1e-9):
                pick = c
        if pick is not None:
            vals["clock"] = float(pick.value)
            conf["clock"] = pick.confidence
            self._update_running(vals["clock"], video_ts)
        return ScoreboardReading(video_ts=video_ts, running=self._running if pick else None,
                                 conf=conf, source=self.source, **vals)

    def _update_running(self, clock: float, ts: float) -> None:
        if self._prev_clock is None:
            self._change_ts = ts
        elif clock != self._prev_clock:
            self._running = clock < self._prev_clock
            self._change_ts = ts
        elif ts - self._change_ts > self.stopped_after_s:
            self._running = False
        self._prev_clock = clock


def render_scoreboard_image(layout: ScoreboardLayout, texts: dict[str, str],
                            shape: tuple[int, int], rng: Optional[np.random.Generator] = None,
                            noise: float = 0.05, blur: float = 0.0, flicker: float = 0.0,
                            on_level: float = 1.0, off_level: float = 0.05,
                            glare: bool = False, segment_thickness: float = 0.18) -> np.ndarray:
    """Synthetisches Tafelbild für Tests/Demos. ``texts[field]`` hat genau n_slots
    Zeichen: Ziffer oder Leerzeichen (leer)."""
    rng = rng or np.random.default_rng()
    img = np.full(shape, 0.02, dtype=np.float32)
    for name, txt in texts.items():
        fl = layout.fields[name]
        assert len(txt) == fl.n_slots, (name, txt)
        for ch, (y0, y1, x0, x1) in zip(txt, fl.slots()):
            d = None if ch == " " else int(ch)
            img[y0:y1, x0:x1] = render_digit(
                d, (y1 - y0, x1 - x0), segment_thickness, noise=0.0, blur=blur,
                on_level=on_level, off_level=off_level, rng=rng, flicker=flicker)
    if glare:
        img[:, :] = np.maximum(img, 0.99)
    if noise > 0:
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)
