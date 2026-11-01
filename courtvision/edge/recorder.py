"""GStreamer-Pipeline des Edge-Rechners und lokale Segmentverwaltung.

Pipeline (Plan: Latenzbudget / Technik-Stack)::

    rtspsrc (PoE-Kamera, H.264/H.265) -> depay -> parse -> tee
        |- (a) splitmuxsink: segmentierte lokale Aufzeichnung, IMMER an (Rohvideo)
        |- (b) Livestream-Ingest per RTMP: Passthrough ohne Re-Encode;
        |       nur mit ``hw_encoder`` Decode -> (Overlay/Ausschnitt) -> HW-Encode
        '- (c) appsink: dekodierte, herunterskalierte Frames fuer die Analyse

Entscheidungen
--------------
* Passthrough ist der Default: Der Edge-Rechner re-enkodiert nicht. Overlay und
  Ausschnitt brauchen einen Hardware-Encoder (``nvv4l2h264enc`` auf Orin NX/AGX).
  Der Orin Nano hat keinen NVENC: ``hw_encoder`` und ``overlay`` werden dort
  abgelehnt (:class:`EncoderUnavailableError`), statt auf CPU zu enkodieren.
* Klassisches RTMP/FLV kann kein H.265. Ein H.265-Kamerastrom geht deshalb nur
  mit ``hw_encoder`` (H.264-Transcode) oder ``enhanced_rtmp=True`` (Ingest versteht
  Enhanced RTMP) zum Ingest; sonst Fehler.
* Segmente: Matroska (robust bei Stromausfall) ist Standard; MP4 nutzt
  fragmentierte Ausgabe (``fragment-duration``), damit ein Abbruch nicht alles zerstoert.
* Die Pipeline wird hier nur als String gebaut; getestet wird ohne GStreamer.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

#: Geraete mit Hardware-Encoder (NVENC). Der Orin Nano hat keinen.
HW_ENCODER_DEVICES = frozenset({"orin_nx", "orin_agx"})
JETSON_DEVICES = frozenset({"orin_nano", "orin_nx", "orin_agx"})
_HW_ENCODERS = {"nvv4l2h264enc": "h264", "nvv4l2h265enc": "h265"}


class EncoderUnavailableError(ValueError):
    """Gewuenschte Re-Encode-/Overlay-Funktion ist auf diesem Geraet nicht moeglich."""


@dataclass
class RecorderConfig:
    camera_url: str = "rtsp://192.168.1.64:554/Streaming/Channels/101"
    codec: str = "h264"  # Kamerastrom: "h264" | "h265"
    segment_dir: str = "/data/courtvision/raw"
    segment_s: int = 300  # 5-min-Segmente
    container: str = "mkv"  # "mkv" | "mp4"
    file_prefix: str = "seg"
    rtmp_url: Optional[str] = None  # Livestream-Ingest, None => kein Stream
    hw_encoder: Optional[str] = None  # z. B. "nvv4l2h264enc"; None => Passthrough
    device: str = "generic"  # "generic" | "orin_nano" | "orin_nx" | "orin_agx"
    overlay: bool = False  # Overlay/Ausschnitt => Re-Encode noetig
    enhanced_rtmp: bool = False  # Ingest versteht H.265 ueber Enhanced RTMP
    stream_bitrate_kbps: int = 8000  # nur bei Re-Encode
    rtsp_latency_ms: int = 200
    rtsp_protocols: str = "tcp"
    analysis_width: int = 1920
    analysis_height: int = 1080
    analysis_fps: int = 10
    analysis_appsink: str = "analysis"
    extra: dict = field(default_factory=dict)

    @property
    def segment_pattern(self) -> str:
        return os.path.join(self.segment_dir, f"{self.file_prefix}_%05d.{self.container}")


def _validate(cfg: RecorderConfig) -> None:
    if cfg.codec not in ("h264", "h265"):
        raise ValueError(f"codec muss h264 oder h265 sein, nicht {cfg.codec!r}")
    if cfg.container not in ("mkv", "mp4"):
        raise ValueError(f"container muss mkv oder mp4 sein, nicht {cfg.container!r}")
    if cfg.segment_s < 10:
        raise ValueError("segment_s zu klein (< 10 s)")
    if cfg.hw_encoder is not None:
        if cfg.hw_encoder not in _HW_ENCODERS:
            raise EncoderUnavailableError(f"unbekannter Hardware-Encoder {cfg.hw_encoder!r}")
        if cfg.device not in HW_ENCODER_DEVICES:
            raise EncoderUnavailableError(
                f"{cfg.hw_encoder} auf Geraet {cfg.device!r} nicht verfuegbar (nur "
                f"{sorted(HW_ENCODER_DEVICES)}); der Orin Nano hat keinen Hardware-Encoder")
    if cfg.overlay and cfg.hw_encoder is None:
        raise EncoderUnavailableError(
            "Overlay/Ausschnitt erfordert Re-Encode und damit einen Hardware-Encoder "
            "(Orin NX/AGX); Orin Nano: nur Passthrough")
    if cfg.rtmp_url and cfg.codec == "h265" and cfg.hw_encoder is None and not cfg.enhanced_rtmp:
        raise ValueError("H.265 per klassischem RTMP nicht moeglich: hw_encoder (H.264-Transcode) "
                         "oder enhanced_rtmp=True setzen")


def _depay_parse(codec: str) -> str:
    n = codec[1:]  # "264" / "265"
    return f"rtph{n}depay ! h{n}parse config-interval=-1"


def _decoder(cfg: RecorderConfig) -> str:
    if cfg.device in JETSON_DEVICES:
        return ("nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert")
    return f"avdec_{cfg.codec} ! videoconvert"


def build_gstreamer_pipeline(cfg: RecorderConfig) -> str:
    """gst-launch-Pipeline (ohne ``gst-launch-1.0``-Praefix) fuer Aufzeichnung, Stream, Analyse."""
    _validate(cfg)
    src = (f"rtspsrc location={cfg.camera_url} protocols={cfg.rtsp_protocols} "
           f"latency={cfg.rtsp_latency_ms} drop-on-latency=true ! {_depay_parse(cfg.codec)} ! "
           f"tee name=t")
    # (a) Aufzeichnung: immer an, grosse Queue (Disk-Stalls duerfen den Rest nicht blockieren)
    if cfg.container == "mkv":
        mux = "muxer-factory=matroskamux"
    else:
        mux = 'muxer-factory=mp4mux muxer-properties="properties,fragment-duration=2000"'
    record = (f"t. ! queue name=q_rec max-size-buffers=0 max-size-bytes=0 max-size-time=10000000000 ! "
              f"splitmuxsink name=rec location={cfg.segment_pattern} "
              f"max-size-time={int(cfg.segment_s) * 1_000_000_000} {mux} send-keyframe-requests=true "
              f"async-finalize=true")
    branches = [record]
    # (b) Livestream
    if cfg.rtmp_url:
        if cfg.hw_encoder is None:
            if cfg.codec == "h264":
                stream = (f"t. ! queue name=q_rtmp leaky=downstream max-size-time=2000000000 ! "
                          f"flvmux streamable=true name=flv ! rtmpsink location=\"{cfg.rtmp_url} live=1\"")
            else:  # Enhanced RTMP (validiert)
                stream = (f"t. ! queue name=q_rtmp leaky=downstream max-size-time=2000000000 ! "
                          f"eflvmux streamable=true name=flv ! rtmpsink location=\"{cfg.rtmp_url} live=1\"")
        else:
            enc_codec = _HW_ENCODERS[cfg.hw_encoder][1:]
            # Platzhalter: reale Overlay-Stufe (nvcompositor/nvdsosd + RGBA-Konvertierung) haengt vom Overlay-Renderer ab
            overlay = " ! nvdsosd" if cfg.overlay else ""
            stream = (f"t. ! queue name=q_rtmp leaky=downstream max-size-time=2000000000 ! "
                      f"nvv4l2decoder ! nvvidconv{overlay} ! {cfg.hw_encoder} "
                      f"bitrate={cfg.stream_bitrate_kbps * 1000} insert-sps-pps=true ! "
                      f"h{enc_codec}parse ! flvmux streamable=true name=flv ! "
                      f"rtmpsink location=\"{cfg.rtmp_url} live=1\"")
        branches.append(stream)
    # (c) Analyse
    analysis = (f"t. ! queue name=q_ana leaky=downstream max-size-buffers=2 ! {_decoder(cfg)} ! "
                f"videoscale ! videorate ! video/x-raw,format=BGR,width={cfg.analysis_width},"
                f"height={cfg.analysis_height},framerate={cfg.analysis_fps}/1 ! "
                f"appsink name={cfg.analysis_appsink} emit-signals=true max-buffers=2 drop=true sync=false")
    branches.append(analysis)
    return f"{src} " + " ".join(branches)


def gst_launch_command(cfg: RecorderConfig) -> str:
    return "gst-launch-1.0 -e " + build_gstreamer_pipeline(cfg)


# ---------------------------------------------------------------------------
# Lokale Segmentverwaltung
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    path: str
    index: int
    size: int
    mtime: float

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


class LocalRecorder:
    """Verwaltet die Segmentdateien auf der Platte (Auflisten, Platz, Loeschen).

    Loesch-Regeln: ein Segment wird nur geloescht, wenn der Upload bestaetigt ist
    (:meth:`mark_uploaded`) UND die Aufbewahrung abgelaufen ist; bei Platzmangel
    (:meth:`prune_for_space`) duerfen bestaetigte Segmente auch frueher gehen. Nicht
    hochgeladene Segmente werden nie geloescht. Das neueste Segment gilt, solange
    aufgezeichnet wird, als offen (nicht abgeschlossen).
    """

    _RE = re.compile(r"_(\d+)\.(mkv|mp4)$")

    def __init__(self, segment_dir: str, *, retention_s: float = 7 * 86400.0,
                 min_free_bytes: int = 20 * 1024**3, clock: Callable[[], float] = time.time) -> None:
        self.dir = segment_dir
        self.retention_s = retention_s
        self.min_free_bytes = min_free_bytes
        self._clock = clock
        os.makedirs(segment_dir, exist_ok=True)
        self._state_path = os.path.join(segment_dir, ".uploaded.json")

    # -- Zustand: bestaetigte Uploads
    def _uploaded(self) -> dict[str, float]:
        try:
            with open(self._state_path) as f:
                return {k: float(v) for k, v in json.load(f).items()}
        except (OSError, ValueError):
            return {}

    def _save_uploaded(self, d: dict[str, float]) -> None:
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self._state_path)

    def mark_uploaded(self, path_or_name: str) -> None:
        d = self._uploaded()
        d[os.path.basename(path_or_name)] = self._clock()
        self._save_uploaded(d)

    def is_uploaded(self, path_or_name: str) -> bool:
        return os.path.basename(path_or_name) in self._uploaded()

    # -- Auflisten
    def list_segments(self) -> list[Segment]:
        out = []
        for name in os.listdir(self.dir):
            m = self._RE.search(name)
            if not m:
                continue
            p = os.path.join(self.dir, name)
            st = os.stat(p)
            out.append(Segment(p, int(m.group(1)), st.st_size, st.st_mtime))
        return sorted(out, key=lambda s: s.index)

    def closed_segments(self, recording: bool = True) -> list[Segment]:
        """Abgeschlossene Segmente (aufzeichnend: ohne das neueste, das noch beschrieben wird)."""
        segs = self.list_segments()
        return segs[:-1] if recording and segs else segs

    def total_bytes(self) -> int:
        return sum(s.size for s in self.list_segments())

    # -- Platz
    def free_bytes(self) -> int:
        return shutil.disk_usage(self.dir).free

    def check_free_space(self, expected_game_s: float = 2 * 3600, bitrate_mbps: float = 20.0) -> bool:
        """Reicht der freie Platz fuer ein Spiel (Plan: ca. 20 Mbit/s x 2 h = 18 GB) plus Reserve?"""
        need = required_bytes(expected_game_s, bitrate_mbps)
        return self.free_bytes() >= need + self.min_free_bytes

    # -- Loeschen
    def deletable(self, now: Optional[float] = None) -> list[Segment]:
        now = self._clock() if now is None else now
        up = self._uploaded()
        return [s for s in self.closed_segments(recording=False)
                if s.name in up and now - up[s.name] >= self.retention_s]

    def delete_uploaded(self, now: Optional[float] = None) -> list[str]:
        """Loescht bestaetigt hochgeladene Segmente nach Ablauf der Aufbewahrung."""
        return self._delete(self.deletable(now))

    def prune_for_space(self, recording: bool = True) -> list[str]:
        """Bei Platzmangel: aelteste bestaetigt hochgeladene Segmente loeschen, bis ``min_free_bytes``
        frei sind. Gibt die geloeschten Pfade zurueck; reicht es nicht, bleibt der Rest stehen
        (Aufrufer prueft :meth:`free_bytes` und alarmiert)."""
        up = self._uploaded()
        gone: list[str] = []
        for s in self.closed_segments(recording=recording):
            if self.free_bytes() >= self.min_free_bytes:
                break
            if s.name in up:
                gone += self._delete([s])
        return gone

    def _delete(self, segs: list[Segment]) -> list[str]:
        gone = []
        for s in segs:
            try:
                os.remove(s.path)
                gone.append(s.path)
            except FileNotFoundError:
                pass
        up = self._uploaded()
        for s in segs:
            up.pop(s.name, None)
        self._save_uploaded(up)
        return gone


def required_bytes(duration_s: float, bitrate_mbps: float) -> int:
    return int(duration_s * bitrate_mbps * 1e6 / 8)
