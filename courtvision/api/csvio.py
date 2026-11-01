"""CSV-Import (Kader) und -Export (Verband)."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Optional

from ..domain.models import Consent

_ALIASES = {
    "number": {"number", "nr", "nummer", "trikot", "trikotnummer", "no", "#", "rueckennummer", "rückennummer"},
    "name": {"name", "spieler", "spielername"},
    "birth_year": {"birthyear", "geburtsjahr", "jahrgang", "jg", "birth", "year", "geburtsjahrgang"},
    "consent": {"consent", "einwilligung", "zustimmung"},
}
_YES = {"ja", "j", "yes", "y", "1", "true", "alle", "all"}
_NO = {"nein", "n", "no", "0", "false", "widerspruch", "objected", "object"}
_LEVELS = {
    "intern": Consent.STATS_INTERNAL, "statsintern": Consent.STATS_INTERNAL,
    "stats_internal": Consent.STATS_INTERNAL,
    "public": Consent.STATS_PUBLIC, "oeffentlich": Consent.STATS_PUBLIC, "öffentlich": Consent.STATS_PUBLIC,
    "stats_public": Consent.STATS_PUBLIC,
    "highlights": Consent.HIGHLIGHTS, "livestream": Consent.LIVESTREAM,
    "training": Consent.TRAINING_DATA, "trainingsdaten": Consent.TRAINING_DATA,
}


@dataclass
class RosterRow:
    line: int
    number: int
    name: str
    birth_year: Optional[int]
    consent: str  # Rohwert


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def detect_delimiter(text: str) -> str:
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    counts = {d: first.count(d) for d in (";", ",", "\t")}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] else ";"


def _norm(h: str) -> str:
    return re.sub(r"[\s_\-.]", "", h.strip().lower())


def parse_roster(text: str) -> tuple[list[RosterRow], list[dict[str, Any]]]:
    """Spalten ``number;name;birth_year;consent`` (Trennzeichen ``;`` oder ``,`` oder Tab, Kopfzeile optional)."""
    delim = detect_delimiter(text)
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
    errors: list[dict[str, Any]] = []
    out: list[RosterRow] = []
    if not rows:
        return out, [{"line": 0, "message": "Leere Datei"}]
    cols = {"number": 0, "name": 1, "birth_year": 2, "consent": 3}
    start = 0
    head = [_norm(c) for c in rows[0]]
    found: dict[str, int] = {}
    for k, al in _ALIASES.items():
        al_n = {_norm(x) for x in al}
        for i, h in enumerate(head):
            if h in al_n:
                found[k] = i
    if "number" in found or "name" in found:
        cols = {k: found.get(k) for k in cols}
        start = 1
    year_now = date.today().year
    seen: set[int] = set()
    for idx, r in enumerate(rows[start:], start=start + 1):
        def cell(k: str) -> str:
            i = cols.get(k)
            return r[i].strip() if i is not None and i < len(r) else ""

        try:
            number = int(cell("number").lstrip("#"))
            if not 0 <= number <= 99:
                raise ValueError
        except ValueError:
            errors.append({"line": idx, "message": f"Ungültige Rückennummer '{cell('number')}' (0-99)"})
            continue
        name = cell("name")
        if not name:
            errors.append({"line": idx, "message": "Name fehlt"})
            continue
        if number in seen:
            errors.append({"line": idx, "message": f"Rückennummer {number} doppelt in der Datei"})
            continue
        by: Optional[int] = None
        if cell("birth_year"):
            try:
                by = int(cell("birth_year"))
                if not 1900 <= by <= year_now:
                    raise ValueError
            except ValueError:
                errors.append({"line": idx, "message": f"Ungültiges Geburtsjahr '{cell('birth_year')}'"})
                continue
        consent = cell("consent")
        try:
            parse_consent(consent)
        except ValueError as e:
            errors.append({"line": idx, "message": str(e)})
            continue
        seen.add(number)
        out.append(RosterRow(idx, number, name, by, consent))
    return out, errors


def age_from_birth_year(by: int, today: Optional[date] = None) -> int:
    return (today or date.today()).year - by


def is_minor_by_year(by: int, today: Optional[date] = None) -> bool:
    """Nur das Jahr ist bekannt: vorsichtig als minderjaehrig behandeln, wenn der Spieler
    *moeglicherweise* unter 18 ist (Jahresdifferenz <= 18)."""
    return age_from_birth_year(by, today) <= 18


def parse_consent(raw: str) -> tuple[str, Consent]:
    """-> ("default"|"object"|"grant", Stufen). Leer = Standard (Erwachsene: Opt-out, Minderjaehrige: keine Freigabe)."""
    s = raw.strip().lower()
    if not s:
        return "default", Consent.NONE
    if s in _NO:
        return "object", Consent.NONE
    if s in _YES:
        return "grant", Consent.adult_default()
    levels = Consent.NONE
    for tok in re.split(r"[|+/,;\s]+", s):
        if not tok:
            continue
        if tok not in _LEVELS:
            raise ValueError(f"Unbekannte Einwilligungsstufe '{tok}' "
                             f"(erlaubt: ja, nein, intern, public, highlights, livestream, training)")
        levels |= _LEVELS[tok]
    return "grant", levels


def to_csv(header: list[str], rows: Iterable[Iterable[Any]], delimiter: str = ";") -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delimiter, lineterminator="\r\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    return buf.getvalue()
