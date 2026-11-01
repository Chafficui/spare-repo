"""Edge-Rechner: lokale Aufzeichnung (GStreamer), Segment-Verwaltung und resumable Upload.

Plan (Ausfallsicherheit): Das Rohvideo wird immer lokal aufgezeichnet; faellt das
Netz aus, wird nachgeholt. Waehrend des Spiels hochgeladene Segmente machen den
vollstaendigen Box-Score <= 15 min nach Abpfiff moeglich.
"""
from .recorder import (EncoderUnavailableError, LocalRecorder, RecorderConfig, Segment,
                       build_gstreamer_pipeline, gst_launch_command)
from .uploader import (LteFallback, SegmentState, StreamFeasibility, Transport, TransportError,
                       UploadQueue, Uploader, realtime_upload_possible, stream_feasible,
                       upload_backlog_eta_s)

__all__ = [n for n in dir() if not n.startswith("_")]
