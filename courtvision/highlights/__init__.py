"""Highlight-Engine: Bewertung, Szenen/Clips, virtuelle Kamera, Overlay, Freigabefilter."""
from .camera import CameraConfig, CropRect, virtual_camera
from .clips import (
    ClipPlan, Scene, Segment, build_scenes, filter_publishable, live_clip, player_reel,
    recap, render_command, season_reel, vertical_clip,
)
from .overlay import AI_LABEL, OverlaySpec, build_overlay, overlay_text_lines
from .scoring import (
    HighlightContext, ScoredEvent, build_contexts, highlight_score, injury_windows, score_events,
)

__all__ = [
    "AI_LABEL", "CameraConfig", "ClipPlan", "CropRect", "HighlightContext", "OverlaySpec", "Scene",
    "ScoredEvent", "Segment", "build_contexts", "build_overlay", "build_scenes", "filter_publishable",
    "highlight_score", "injury_windows", "live_clip", "overlay_text_lines", "player_reel", "recap",
    "render_command", "score_events", "season_reel", "vertical_clip", "virtual_camera",
]
