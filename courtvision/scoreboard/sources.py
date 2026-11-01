"""Erfassungswege A (Datenschnittstelle), B (Zweitkamera), C (Hauptkamera)."""
from __future__ import annotations

import re
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .sevenseg import ScoreboardOCR
from .state import ScoreboardReading

# --------------------------------------------------------------------------
# Protokoll-Parser (Weg A)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    offset: int  # relativ zum Nutzdatenanfang (nach dem Startbyte)
    length: int
    kind: str = "int"  # int | clock | flag
    #: Akzeptierte Zeichen für kind="flag": Uhr läuft.
    flag_true: str = "R1"


_CLOCK_MS = re.compile(r"^(\d{0,2}):(\d{2})$")
_CLOCK_ST = re.compile(r"^(\d{1,2})\.(\d)$")


def parse_clock(text: str) -> Optional[float]:
    """"MM:SS", " M:SS" oder "SS.t" -> Sekunden."""
    t = text.strip()
    m = _CLOCK_MS.match(t)
    if m:
        return int(m.group(1) or 0) * 60 + int(m.group(2)) + 0.0
    m = _CLOCK_ST.match(t)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 10.0
    return None


@dataclass(frozen=True)
class FixedFieldProtocol:
    """Konfigurierbares ASCII-Paket mit festen Feldpositionen.

    Ein Paket ist ``start + nutzdaten + end``; Offsets zählen ab Nutzdatenanfang.
    Ohne ``start`` wird nach ``end`` bzw. ``packet_length`` getrennt.
    """

    name: str
    fields: dict[str, FieldSpec]
    start: bytes = b""
    end: bytes = b""
    packet_length: Optional[int] = None
    encoding: str = "ascii"
    verified: bool = False

    def split(self, buf: bytearray) -> list[bytes]:
        """Entnimmt vollständige Pakete aus dem Puffer (Reststück bleibt)."""
        out: list[bytes] = []
        while True:
            if self.start:
                s = buf.find(self.start)
                if s < 0:
                    buf.clear()
                    break
                del buf[:s]
            if self.end:
                e = buf.find(self.end, len(self.start))
                if e < 0:
                    break
                n = e + len(self.end)
            elif self.packet_length:
                n = self.packet_length
                if len(buf) < n:
                    break
            else:
                raise ValueError("Protokoll braucht end oder packet_length")
            out.append(bytes(buf[:n]))
            del buf[:n]
        return out

    def parse(self, packet: bytes, video_ts: float = 0.0, source: str = "data") -> Optional[ScoreboardReading]:
        if self.start and packet.startswith(self.start):
            packet = packet[len(self.start):]
        if self.end and packet.endswith(self.end):
            packet = packet[: -len(self.end)]
        need = max((f.offset + f.length for f in self.fields.values()), default=0)
        if len(packet) < need:
            return None
        txt = packet.decode(self.encoding, errors="replace")
        vals: dict = {}
        for name, spec in self.fields.items():
            raw = txt[spec.offset: spec.offset + spec.length]
            if spec.kind == "clock":
                v = parse_clock(raw)
            elif spec.kind == "flag":
                vals[name] = raw.strip()[:1] in tuple(spec.flag_true) if raw.strip() else None
                continue
            else:
                s = raw.strip()
                v = int(s) if s.isdigit() else None
            if v is not None:
                vals[name] = v
        if not vals:
            return None
        return ScoreboardReading(video_ts=video_ts, source=source, **vals)


#: ACHTUNG: Offsets/Steuerzeichen sind ein ILLUSTRATIVES Beispiel, kein
#: geprüftes Daktronics-RTD-Layout (Details nicht vollständig öffentlich).
#: TO BE VERIFIED ON SITE gegen das Handbuch der Tafel bzw. einen Mitschnitt.
DAKTRONICS_RTD_GENERIC = FixedFieldProtocol(
    name="daktronics_rtd_generic",
    start=b"\x01", end=b"\x04",
    fields={
        "home": FieldSpec(0, 3), "away": FieldSpec(3, 3),
        "clock": FieldSpec(6, 5, "clock"),
        "period": FieldSpec(11, 1),
        "fouls_home": FieldSpec(12, 2), "fouls_away": FieldSpec(14, 2),
        "running": FieldSpec(16, 1, "flag"),
    },
    verified=False,
)
PROTOCOLS = {DAKTRONICS_RTD_GENERIC.name: DAKTRONICS_RTD_GENERIC}


# --------------------------------------------------------------------------
# Quellen
# --------------------------------------------------------------------------
class ScoreboardSource(ABC):
    quality = 0  # A=3, B=2, C=1
    name = "source"

    def __init__(self, clock: Callable[[], float] = time.monotonic, max_age: float = 3.0):
        self.clock = clock
        self.max_age = max_age
        self.last_update: Optional[float] = None

    @abstractmethod
    def _poll(self) -> list[ScoreboardReading]: ...

    def poll(self) -> list[ScoreboardReading]:
        out = self._poll()
        if out:
            self.last_update = self.clock()
        return out

    def healthy(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return self.last_update is not None and now - self.last_update <= self.max_age

    def close(self) -> None:  # pragma: no cover
        pass


class _ProtocolSource(ScoreboardSource):
    quality = 3

    def __init__(self, protocol: FixedFieldProtocol, **kw):
        super().__init__(**kw)
        self.protocol = protocol
        self._buf = bytearray()

    def _feed(self, data: bytes) -> list[ScoreboardReading]:
        self._buf += data
        res = []
        for pkt in self.protocol.split(self._buf):
            r = self.protocol.parse(pkt, self.clock(), source=self.name)
            if r is not None:
                res.append(r)
        return res


class SerialScoreboardSource(_ProtocolSource):
    """RS-232 am Steuerpult; pyserial wird erst in ``open`` importiert."""

    name = "serial"

    def __init__(self, port: str, protocol: FixedFieldProtocol, baudrate: int = 9600,
                 serial_factory: Optional[Callable[..., object]] = None, **kw):
        super().__init__(protocol, **kw)
        self.port, self.baudrate = port, baudrate
        self._factory = serial_factory
        self._ser = None

    def open(self) -> None:
        factory = self._factory
        if factory is None:
            import serial  # lazy: optionale Abhängigkeit

            factory = serial.Serial
        self._ser = factory(self.port, self.baudrate, timeout=0)

    def _poll(self) -> list[ScoreboardReading]:
        if self._ser is None:
            return []
        n = getattr(self._ser, "in_waiting", 0)
        return self._feed(self._ser.read(n)) if n else []

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None


class UdpScoreboardSource(_ProtocolSource):
    """UDP-Broadcast/Unicast (nicht blockierend)."""

    name = "udp"

    def __init__(self, protocol: FixedFieldProtocol, host: str = "0.0.0.0", port: int = 0, **kw):
        super().__init__(protocol, **kw)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    @property
    def bound_port(self) -> int:
        return self.sock.getsockname()[1]

    def _poll(self) -> list[ScoreboardReading]:
        res: list[ScoreboardReading] = []
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError):
                break
            got = self._feed(data)
            if not got:  # Datagramm ohne Rahmenbytes: als ganzes Paket versuchen
                r = self.protocol.parse(data, self.clock(), source=self.name)
                got = [r] if r else []
            res += got
        return res

    def close(self) -> None:
        self.sock.close()


class CameraScoreboardSource(ScoreboardSource):
    """Weg B (Zweitkamera, quality 2) oder C (Hauptkamera-ROI, quality 1)."""

    def __init__(self, ocr: ScoreboardOCR, kind: str = "B", min_confidence: float = 0.6, **kw):
        super().__init__(**kw)
        assert kind in ("B", "C")
        self.ocr = ocr
        self.kind = kind
        self.quality = 2 if kind == "B" else 1
        self.name = f"camera_{kind}"
        self.min_confidence = min_confidence
        self._queue: list[ScoreboardReading] = []

    def push_frame(self, image: np.ndarray, video_ts: Optional[float] = None) -> Optional[ScoreboardReading]:
        r = self.ocr.read(image, self.clock() if video_ts is None else video_ts)
        r.source = self.name
        if (r.home is not None or r.away is not None or r.clock is not None) \
                and r.overall_confidence >= self.min_confidence:
            self._queue.append(r)
            return r
        return None

    def _poll(self) -> list[ScoreboardReading]:
        q, self._queue = self._queue, []
        return q


def select_source(sources: Sequence[ScoreboardSource], now: Optional[float] = None) -> Optional[ScoreboardSource]:
    """Beste gesunde Quelle (A > B > C). Veraltete Quellen werden übergangen;
    keine gesunde Quelle -> None."""
    ok = [s for s in sources if s.healthy(now)]
    return max(ok, key=lambda s: s.quality) if ok else None
