"""Team-Clustering aus Erscheinungs-Embeddings (nur transient, nur ein Spiel).

Produktplan: SigLIP-Embeddings -> UMAP -> KMeans (k=2). Hier: PCA (numpy) statt
UMAP (UMAP waere ein optionaler Austausch, s. ``reduce_fn``) und ein eigenes
KMeans. Startwerte kommen aus den Trikotfarben der Spielanlage (``seed_colors``,
RGB); Schiedsrichter sind eine eigene Klasse (Detektionsklasse "referee" oder ein
Ausreisser-Cluster weit von beiden Teams). Es werden keine Gesichter betrachtet.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ..domain.models import Side

RGB = tuple[float, float, float]

_NAMED = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (220, 30, 30), "blue": (30, 60, 200),
          "green": (30, 160, 60), "yellow": (240, 220, 40), "orange": (250, 130, 20),
          "purple": (130, 50, 170), "grey": (128, 128, 128), "gray": (128, 128, 128),
          "navy": (10, 20, 100), "pink": (240, 120, 180)}


def parse_color(c: str) -> RGB:
    """Hex ("#rrggbb"/"rgb") oder einfacher Farbname -> RGB (0..255)."""
    s = c.strip().lower()
    if s in _NAMED:
        return tuple(float(v) for v in _NAMED[s])  # type: ignore[return-value]
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Farbe nicht erkannt: {c!r}")
    return tuple(float(int(s[i:i + 2], 16)) for i in (0, 2, 4))  # type: ignore[return-value]


# -- Embedding-Fallback ------------------------------------------------------
def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-9
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h / 6.0, s, mx], axis=-1)


def color_histogram_embedding(crop: np.ndarray, hue_bins: int = 12,
                              torso: tuple[float, float, float, float] = (0.2, 0.6, 0.2, 0.8)) -> np.ndarray:
    """HSV-Histogramm der Torso-Region eines Spieler-Crops (H, W, 3, RGB uint8).

    Farbige Pixel (S, V hoch) wandern gewichtet in Hue-Bins, unbunte in drei
    Helligkeitsbins (dunkel/grau/hell). Ergebnis L1-normiert, Dimension hue_bins+3.
    """
    h, w = crop.shape[:2]
    r0, r1, c0, c1 = torso
    roi = crop[int(h * r0):max(int(h * r1), int(h * r0) + 1), int(w * c0):max(int(w * c1), int(w * c0) + 1)]
    hsv = _rgb_to_hsv(roi.reshape(-1, 3))
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    colored = (sat > 0.25) & (val > 0.2)
    hist = np.zeros(hue_bins + 3)
    if colored.any():
        idx = ((hue[colored] + 0.5 / hue_bins) * hue_bins).astype(int) % hue_bins  # Rot liegt mittig in Bin 0
        np.add.at(hist, idx, sat[colored] * val[colored])
    ach = ~colored
    if ach.any():
        v = val[ach]
        hist[hue_bins] += np.sum(v < 0.3)
        hist[hue_bins + 1] += np.sum((v >= 0.3) & (v < 0.7))
        hist[hue_bins + 2] += np.sum(v >= 0.7)
    s = hist.sum()
    return hist / s if s > 0 else hist


def flat_patch_embedding(rgb: RGB, size: tuple[int, int] = (40, 20)) -> np.ndarray:
    """Embedding einer einfarbigen Flaeche (zum Einbetten der Seed-Farben)."""
    patch = np.empty((*size, 3), np.uint8)
    patch[:] = np.clip(rgb, 0, 255).astype(np.uint8)
    return color_histogram_embedding(patch)


def rgb_identity_embedding(rgb: RGB) -> np.ndarray:
    """Seed-Einbettung, wenn die Track-Embeddings selbst RGB-artige Merkmale sind."""
    return np.asarray(rgb, float)


# -- PCA / KMeans ------------------------------------------------------------
class PCA:
    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray = np.zeros(0)
        self.components_: np.ndarray = np.zeros((0, 0))

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, float)
        self.mean_ = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        k = max(1, min(self.n_components, vt.shape[0], X.shape[1]))
        self.components_ = vt[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mean_) @ self.components_.T


def kmeans(X: np.ndarray, k: int = 2, init: Optional[np.ndarray] = None, n_iter: int = 100,
           seed: int = 0, n_init: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd-KMeans. Rueckgabe (labels, centroids, inertia). Ohne ``init``: k-means++."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)

    def run(c: np.ndarray):
        c = c.copy()
        for _ in range(n_iter):
            d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            new = c.copy()
            for j in range(len(c)):
                m = lab == j
                new[j] = X[m].mean(0) if m.any() else X[d.min(1).argmax()]
            if np.allclose(new, c):
                c = new
                break
            c = new
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        return lab, c, float(d.min(1).sum())

    if init is not None:
        return run(np.asarray(init, float))
    best = None
    for _ in range(n_init):
        cs = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d2 = np.min([((X - c) ** 2).sum(1) for c in cs], axis=0)
            p = d2 / d2.sum() if d2.sum() > 0 else None
            cs.append(X[rng.choice(len(X), p=p)])
        r = run(np.array(cs))
        if best is None or r[2] < best[2]:
            best = r
    return best  # type: ignore[return-value]


class TeamAssigner:
    """Sammelt Embeddings je Track, clustert in zwei Teams und ordnet home/away zu.

    ``seed_colors={"home": rgb, "away": rgb}`` (aus der Spielanlage) legt die
    Startzentren und die Zuordnung Cluster -> Seite fest; ohne Seeds ist die
    Zuordnung willkuerlich (aber ueber Refits stabil) und ``seeded`` False.
    ``seed_embed`` bettet eine RGB-Farbe in den Embedding-Raum ein (Default:
    HSV-Histogramm einer einfarbigen Flaeche, passend zu ``color_histogram_embedding``;
    fuer Roh-RGB-Merkmale ``rgb_identity_embedding``).
    """

    def __init__(self, seed_colors: Optional[Mapping[Side, RGB]] = None,
                 seed_embed: Callable[[RGB], np.ndarray] = flat_patch_embedding,
                 n_components: int = 3, min_tracks: int = 4, outlier_k: float = 4.0,
                 reduce_fn: Optional[Callable[[np.ndarray], tuple[np.ndarray, Callable]]] = None) -> None:
        self.seed_colors = dict(seed_colors) if seed_colors else None
        self.seed_embed = seed_embed
        self.n_components, self.min_tracks, self.outlier_k = n_components, min_tracks, outlier_k
        self.reduce_fn = reduce_fn  # optional: z. B. UMAP; liefert (Z, transform)
        self._emb: dict[int, np.ndarray] = {}
        self._n: dict[int, int] = {}
        self._ref: set[int] = set()
        self._transform: Optional[Callable] = None
        self.centroids: Optional[np.ndarray] = None  # (2, d) im reduzierten Raum, Reihenfolge (home, away)
        self._ref_thresh: float = np.inf
        self.fitted = False

    @property
    def seeded(self) -> bool:
        return self.seed_colors is not None

    def observe(self, track_id: int, embedding: np.ndarray, is_referee: bool = False) -> None:
        e = np.asarray(embedding, float)
        n = self._n.get(track_id, 0)
        self._emb[track_id] = e if n == 0 else (self._emb[track_id] * n + e) / (n + 1)
        self._n[track_id] = n + 1
        if is_referee:
            self._ref.add(track_id)

    def clear(self) -> None:
        """Spielende: Merkmale und Modell verwerfen."""
        self._emb.clear(); self._n.clear(); self._ref.clear()
        self._transform, self.centroids, self.fitted = None, None, False

    def fit(self) -> bool:
        ids = [t for t in self._emb if t not in self._ref]
        if len(ids) < max(2, self.min_tracks):
            return False
        X = np.array([self._emb[t] for t in ids])
        if self.reduce_fn is not None:
            Z, tf = self.reduce_fn(X)
        else:
            pca = PCA(self.n_components).fit(X)
            Z, tf = pca.transform(X), pca.transform
        seeds = None
        if self.seed_colors:
            seeds = np.array([tf(np.asarray(self.seed_embed(self.seed_colors[s]), float)[None])[0] for s in ("home", "away")])
        keep = np.ones(len(ids), bool)
        cen = seeds if seeds is not None else None
        for _ in range(3):  # trimmed KMeans: Ausreisser (Schiris) mit robusten Zentren erkennen
            lab, cen, _ = kmeans(Z[keep], 2, init=seeds) if seeds is not None else kmeans(Z[keep], 2)
            full_lab = np.linalg.norm(Z[:, None, :] - cen[None], axis=2).argmin(1)
            rob = np.array([np.median(Z[full_lab == j], axis=0) if (full_lab == j).any() else cen[j] for j in range(2)])
            if seeds is None and self.centroids is not None:
                # Label-Stabilitaet ueber Refits
                if np.linalg.norm(rob[0] - self.centroids[1]) + np.linalg.norm(rob[1] - self.centroids[0]) < \
                        np.linalg.norm(rob[0] - self.centroids[0]) + np.linalg.norm(rob[1] - self.centroids[1]):
                    cen, rob = cen[::-1], rob[::-1]
            if seeds is not None:  # Zuordnung ueber Seed-Naehe
                straight = np.linalg.norm(cen[0] - seeds[0]) + np.linalg.norm(cen[1] - seeds[1])
                swapped = np.linalg.norm(cen[0] - seeds[1]) + np.linalg.norm(cen[1] - seeds[0])
                if swapped < straight:
                    cen, rob = cen[::-1], rob[::-1]
            d = np.linalg.norm(Z[:, None, :] - rob[None], axis=2).min(1)
            inter = np.linalg.norm(rob[0] - rob[1])
            med = np.median(d[keep])
            mad = 1.4826 * np.median(np.abs(d[keep] - med))
            thr = max(med + self.outlier_k * max(mad, 1e-6), 0.5 * inter)
            new_keep = d <= thr
            if new_keep.sum() < 2 or (new_keep == keep).all():
                break
            keep = new_keep
        _, cen, _ = kmeans(Z[keep], 2, init=cen)
        d = np.linalg.norm(Z[:, None, :] - cen[None], axis=2).min(1)
        inter = np.linalg.norm(cen[0] - cen[1])
        med = np.median(d[keep])
        mad = 1.4826 * np.median(np.abs(d[keep] - med))
        self._ref_thresh = max(med + self.outlier_k * max(mad, 1e-6), 0.5 * inter)
        self.centroids, self._transform, self.fitted = cen, tf, True
        return True

    def team_of(self, track_id: int) -> Optional[Side]:
        """"home"/"away"; None fuer Schiris/Ausreisser/unbekannt/noch ungefittet."""
        if not self.fitted or track_id in self._ref or track_id not in self._emb:
            return None
        z = self._transform(self._emb[track_id][None])[0]  # type: ignore[misc]
        d = np.linalg.norm(self.centroids - z, axis=1)  # type: ignore[operator]
        if d.min() > self._ref_thresh:
            return None
        return "home" if d[0] <= d[1] else "away"

    def is_referee_like(self, track_id: int) -> bool:
        if track_id in self._ref:
            return True
        return self.fitted and track_id in self._emb and self.team_of(track_id) is None
