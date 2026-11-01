"""Event-Erkennung: Zustandsautomat (Observationen -> Events) und Wechselerkennung."""
from .engine import EngineConfig, EventEngine
from .substitutions import SubstitutionConfig, SubstitutionDetector, SubstitutionZones

__all__ = [
    "EngineConfig",
    "EventEngine",
    "SubstitutionConfig",
    "SubstitutionDetector",
    "SubstitutionZones",
]
