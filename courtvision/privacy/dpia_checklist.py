"""Pflichtbausteine im Produkt (Recht & Datenschutz) als Daten mit Statusverfolgung."""
from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional

from pydantic import BaseModel


class ItemStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class ChecklistItem(BaseModel):
    id: str
    title: str
    reference: str = ""
    blocks_pilot: bool = False
    status: ItemStatus = ItemStatus.OPEN
    owner: str = ""
    note: str = ""


_ITEMS = [
    ChecklistItem(id="dpia", blocks_pilot=True, reference="Art. 35 DSGVO",
                  title="Datenschutz-Folgenabschaetzung vor dem Pilot: systematische Videoerfassung "
                        "Minderjaehriger plus Profilbildung"),
    ChecklistItem(id="club_contract", blocks_pilot=True, reference="Art. 28 / Art. 26 DSGVO",
                  title="Vertrag mit dem Verein: Auftragsverarbeitung (Art. 28) oder gemeinsame "
                        "Verantwortung (Art. 26) klaeren"),
    ChecklistItem(id="consent_management", blocks_pilot=True,
                  title="Einwilligungsverwaltung je Spieler mit Stufen: Stats intern / Stats oeffentlich / "
                        "Highlights / Livestream"),
    ChecklistItem(id="objection_handling", blocks_pilot=True,
                  title="Spieler mit Widerspruch: Overlay ohne Namen, keine Highlights, Stats nur als Team-Wert"),
    ChecklistItem(id="eu_hosting_retention", blocks_pilot=True,
                  title="Hosting und Verarbeitung ausschliesslich in der EU; Loeschfristen fuer Rohvideo "
                        "(z. B. 90 Tage), laenger nur mit ausdruecklicher Einwilligung zur Nutzung als "
                        "Trainingsdaten"),
    ChecklistItem(id="signage", blocks_pilot=True,
                  title="Hinweisschilder in der Halle und Info im Spielplan"),
]


class Checklist:
    def __init__(self, items: Optional[Iterable[ChecklistItem]] = None) -> None:
        self.items: dict[str, ChecklistItem] = {i.id: i for i in (items or [i.model_copy() for i in _ITEMS])}

    def set_status(self, item_id: str, status: ItemStatus, owner: str = "", note: str = "") -> ChecklistItem:
        it = self.items[item_id]
        it.status = ItemStatus(status)
        if owner:
            it.owner = owner
        if note:
            it.note = note
        return it

    def open_items(self) -> list[ChecklistItem]:
        return [i for i in self.items.values() if i.status != ItemStatus.DONE]

    def pilot_blockers(self) -> list[ChecklistItem]:
        return [i for i in self.open_items() if i.blocks_pilot]

    def is_complete(self) -> bool:
        return not self.open_items()

    def progress(self) -> float:
        return sum(i.status == ItemStatus.DONE for i in self.items.values()) / len(self.items)


def default_checklist() -> Checklist:
    return Checklist()
