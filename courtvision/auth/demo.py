"""Demo-Anmeldung: Fake-Codai + vorbereitete Rollen fuer ein paar Demo-E-Mail-Adressen (``serve --demo``)."""
from __future__ import annotations

from typing import Optional

from ..sim import DemoSetup
from ..store.eventstore import EventStore
from .config import AuthConfig
from .fake_codai import DEFAULT_SERVICE, FakeCodai
from .users import UserDirectory

DEMO_DOMAIN = "demo.courtvision.local"
DEMO_ADMIN = f"admin@{DEMO_DOMAIN}"


def demo_accounts(setup: DemoSetup) -> dict[str, list[dict]]:
    """E-Mail -> Mitgliedschaften (Admin kommt ueber ``COURTVISION_ADMINS``)."""
    home, away = setup.home.id, setup.away.id
    first_home = setup.players[home][0]
    return {
        f"hallenwart@{DEMO_DOMAIN}": [{"role": "hallenwart"}],
        f"kampfgericht@{DEMO_DOMAIN}": [{"role": "kampfgericht"}],
        f"trainer.heim@{DEMO_DOMAIN}": [{"role": "trainer", "team_id": home}],
        f"trainer.gast@{DEMO_DOMAIN}": [{"role": "trainer", "team_id": away}],
        f"spieler@{DEMO_DOMAIN}": [{"role": "spieler", "player_id": first_home.id}],
        f"verband@{DEMO_DOMAIN}": [{"role": "verband"}],
        # mehrere Mitgliedschaften -> Rollenwechsel in der Kopfzeile
        f"mehrfach@{DEMO_DOMAIN}": [{"role": "trainer", "team_id": home}, {"role": "kampfgericht"},
                                    {"role": "spieler", "player_id": first_home.id}],
    }


def make_demo_codai(fixed_code: Optional[str] = None, announce=None) -> FakeCodai:
    """Fake-Codai mit E-Mail-Claim (wie ``EMAIL_CLAIM_SERVICES=courtvision``) und Demo-Passwoertern."""
    emails = [DEMO_ADMIN]
    return FakeCodai(email_claim={DEFAULT_SERVICE}, fixed_code=fixed_code, auto_usernames=True,
                     passwords={e: "demo-passwort" for e in emails}, announce=announce)


DEMO_LABELS = {
    DEMO_ADMIN: "Admin", f"hallenwart@{DEMO_DOMAIN}": "Hallenwart", f"kampfgericht@{DEMO_DOMAIN}": "Kampfgericht",
    f"trainer.heim@{DEMO_DOMAIN}": "Trainer Heim", f"trainer.gast@{DEMO_DOMAIN}": "Trainer Gast",
    f"spieler@{DEMO_DOMAIN}": "Spieler", f"verband@{DEMO_DOMAIN}": "Verband",
    f"mehrfach@{DEMO_DOMAIN}": "Trainer + Kampfgericht + Spieler",
}


def demo_config(cfg: AuthConfig, fake_url: bool = True) -> AuthConfig:
    # Alle Demo-Konten melden sich von derselben IP an: eigene Limits grosszuegig (Codai-Limits bleiben)
    kw = dict(demo=True, admins=frozenset(set(cfg.admins) | {DEMO_ADMIN}), demo_accounts=tuple(DEMO_LABELS.items()),
              start_per_ip=200, start_per_email=20, verify_per_ip=500, password_per_ip=200, password_per_email=50)
    if fake_url:
        kw.update(codai_public_url="/_fake_codai", service_token="", mode="codai")
    return cfg.with_(**kw)


def seed_demo_invites(store: EventStore, setup: DemoSetup, cfg: AuthConfig) -> list[str]:
    """Rollen der Demo-Adressen als Einladungen vormerken (greifen beim ersten Login; idempotent)."""
    users = UserDirectory(store, cfg)
    emails = []
    for email, ms in demo_accounts(setup).items():
        if users.find_by_email(email) is None:
            users.invite(email, ms, by="demo")
        emails.append(email)
    return emails
