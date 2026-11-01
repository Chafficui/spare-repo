"""CourtVision CV-Pipeline: Detektion, Tracking, Teams, Trikotnummern, Ball, Besitz.

Schwere ML-Abhaengigkeiten (rfdetr, torch, rtmlib) sind optional und werden nur
in den Adaptern lazy importiert.
"""
from .ball import (BallHeatmapModel, BallSample, Parabola, RimMakeDetector, ShotDetector, fit_parabola,
                   heatmap_peak)
from .detection import (CropRegion, Detection, Detector, DetectorUnavailableError, RFDETRDetector, crop_regions,
                        downscale, rim_roi)
from .jersey import (Assignment, JerseyNumberReader, JerseyReading, NumberVoter, RosterAssigner, RosterEntry,
                     TrackVotes)
from .pipeline import CVPipeline, Mode, PipelineConfig, RimConfig
from .pose import PoseEstimator, foot_point, hand_points
from .possession import FormationDetector, LastTouch, OutOfBoundsDetector, PossessionEstimator
from .teams import TeamAssigner, color_histogram_embedding, kmeans, parse_color
from .tracking import Track, Tracker, TrackerConfig

__all__ = [n for n in dir() if not n.startswith("_")]
