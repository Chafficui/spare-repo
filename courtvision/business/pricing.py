"""Preise und Unit Economics laut Produktplan (alle Preise sind Annahmen)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _r(x: float) -> float:
    return round(x + 1e-12, 2)


@dataclass(frozen=True)
class PricingPlan:
    hall_monthly: float = 199.0
    term_months: int = 24
    games_included: int = 20
    extra_game: float = 10.0
    player_premium_monthly: float = 3.99
    #: Anteil des Vereins an den Livestream-Pass-Einnahmen
    livestream_club_share: float = 0.7
    vat_rate: float = 0.0


@dataclass(frozen=True)
class CostPerGame:
    """Kosten pro Spiel (2 h, 50 Zuschauer a 60 min), Parameter aus der Kostentabelle."""
    gpu_hours: float = 2.0
    gpu_usd_per_hour: float = 0.64  # Mitte von 0,40-0,88 USD/h (L4)
    viewers: int = 50
    minutes_per_viewer: float = 60.0
    delivery_usd_per_1000_min: float = 1.0
    recording_minutes: float = 120.0
    stream_storage_usd_per_1000_min_month: float = 5.0
    raw_gb: float = 18.0
    raw_storage_eur_per_tb_month: float = 8.0  # 7-9 EUR/TB/Monat
    retention_months: float = 3.0  # 90 Tage
    usd_eur: float = 0.92

    @property
    def gpu_usd(self) -> float:
        return self.gpu_hours * self.gpu_usd_per_hour

    @property
    def stream_usd(self) -> float:
        return (self.viewers * self.minutes_per_viewer / 1000.0 * self.delivery_usd_per_1000_min
                + self.recording_minutes / 1000.0 * self.stream_storage_usd_per_1000_min_month)

    @property
    def raw_storage_eur(self) -> float:
        return self.raw_gb / 1000.0 * self.raw_storage_eur_per_tb_month * self.retention_months

    @property
    def total_eur(self) -> float:
        return _r((self.gpu_usd + self.stream_usd) * self.usd_eur + self.raw_storage_eur)


@dataclass
class Invoice:
    lines: list[tuple[str, float]] = field(default_factory=list)
    vat_rate: float = 0.0

    @property
    def net(self) -> float:
        return _r(sum(a for _, a in self.lines))

    @property
    def vat(self) -> float:
        return _r(self.net * self.vat_rate)

    @property
    def total(self) -> float:
        return _r(self.net + self.vat)


def invoice(games_played: int, plan: PricingPlan = PricingPlan()) -> Invoice:
    """Monatsrechnung der Halle: Abo plus Zusatzspiele ab Spiel 21."""
    inv = Invoice(vat_rate=plan.vat_rate)
    inv.lines.append(("Hallen-Abo", _r(plan.hall_monthly)))
    extra = max(0, games_played - plan.games_included)
    if extra:
        inv.lines.append((f"{extra} Zusatzspiele", _r(extra * plan.extra_game)))
    return inv


def livestream_split(gross: float, plan: PricingPlan = PricingPlan(),
                     club_share: Optional[float] = None) -> tuple[float, float]:
    """(Anteil Verein, Anteil Plattform) der Livestream-Pass-Einnahmen."""
    share = plan.livestream_club_share if club_share is None else club_share
    club = _r(gross * share)
    return club, _r(gross - club)


@dataclass(frozen=True)
class UnitEconomics:
    revenue: float
    operating_cost: float
    hardware_per_month: float
    contribution_margin: float
    contract_total: float


def unit_economics(games_per_month: int = 15, cost_per_game: float = 5.0, hardware_total: float = 2000.0,
                   months: int = 24, plan: PricingPlan = PricingPlan(), premium_subscribers: int = 0,
                   livestream_gross: float = 0.0, hardware_per_month: Optional[float] = None) -> UnitEconomics:
    """Deckungsbeitrag je Halle und Monat vor Vertrieb, Support und Personal.

    Standard: 199 - 15*5 - 2000/24 = 40,67 EUR. Der Plan rundet die Hardware auf ~85 EUR/Monat
    und nennt "etwa 39 EUR"; mit ``hardware_per_month=85`` ergibt sich exakt 39.
    """
    inv = invoice(games_per_month, plan)
    revenue = inv.net + premium_subscribers * plan.player_premium_monthly \
        + livestream_split(livestream_gross, plan)[1]
    op = games_per_month * cost_per_game
    hw = hardware_total / months if hardware_per_month is None else hardware_per_month
    return UnitEconomics(revenue=_r(revenue), operating_cost=_r(op), hardware_per_month=_r(hw),
                         contribution_margin=_r(revenue - op - hw),
                         contract_total=_r(plan.hall_monthly * plan.term_months))
