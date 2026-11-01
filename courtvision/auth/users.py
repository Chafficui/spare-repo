"""Benutzer und Rollen in CourtVision.

Codai sagt nur *wer* (``subject``, ``username``, ggf. ``email``); *was* jemand darf, steht hier:
Dokumente der Art ``user`` (``subject``, ``username``, ``email``, ``memberships``, ``created_at``, ``last_login``)
und ``invite`` (Rollen fuer eine E-Mail-Adresse, die beim ersten Login vergeben werden).

Eine *Mitgliedschaft* ist ``{role, team_id?, player_id?}``: ``trainer`` braucht ``team_id``, ``spieler`` ``player_id``.
Wer keine hat, ist ``fan``. Admins aus ``COURTVISION_ADMINS`` (Subjects und/oder E-Mails) bekommen ``admin`` zusaetzlich.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Iterable, Optional

from ..store.eventstore import EventStore
from .config import AuthConfig

USER_KIND = "user"
INVITE_KIND = "invite"
#: Rollen, die vergeben werden duerfen (``fan`` ist der Standard, ``system`` nur per Bearer-Token).
ASSIGNABLE_ROLES = ("admin", "hallenwart", "kampfgericht", "trainer", "spieler", "verband")
FAN = {"role": "fan", "team_id": None, "player_id": None}
ROLE_LABEL = {"fan": "Fan", "spieler": "Spieler / Eltern", "trainer": "Trainer", "kampfgericht": "Kampfgericht",
              "hallenwart": "Hallenwart", "verband": "Verband", "admin": "Admin", "system": "System"}


class MembershipError(ValueError):
    pass


def norm_email(e: str) -> str:
    return e.strip().lower()


def m_key(m: dict[str, Any]) -> tuple:
    return (m["role"], m.get("team_id") or None, m.get("player_id") or None)


class UserDirectory:
    def __init__(self, store: EventStore, cfg: AuthConfig) -> None:
        self.store, self.cfg = store, cfg
        self._lock = threading.RLock()

    # ------------------------------------------------------------ Mitgliedschaften
    def normalize(self, raw: Iterable[Any]) -> list[dict[str, Any]]:
        """Prueft und bereinigt eine Mitgliedschaftsliste; wirft :class:`MembershipError` (deutsche Meldung)."""
        out: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        for item in raw or []:
            if not isinstance(item, dict):
                raise MembershipError("Mitgliedschaft muss ein Objekt {role, team_id?, player_id?} sein")
            role = str(item.get("role") or "").strip().lower()
            team = (item.get("team_id") or None)
            player = (item.get("player_id") or None)
            if role not in ASSIGNABLE_ROLES:
                raise MembershipError(f"Unbekannte Rolle '{role}'. Erlaubt: {', '.join(ASSIGNABLE_ROLES)}")
            if role == "trainer":
                if not team:
                    raise MembershipError("Trainer brauchen ein Team (team_id)")
                if self.store.get_team(team) is None:
                    raise MembershipError(f"Team '{team}' nicht gefunden")
                player = None
            elif role == "spieler":
                if not player:
                    raise MembershipError("Spieler brauchen eine Spieler-ID (player_id)")
                pl = self.store.get_player(player)
                if pl is None:
                    raise MembershipError(f"Spieler '{player}' nicht gefunden")
                team = pl.team_id
            else:
                team = player = None
            m = {"role": role, "team_id": team, "player_id": player}
            if m_key(m) not in seen:
                seen.add(m_key(m))
                out.append(m)
        return out

    def effective_memberships(self, user: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        ms = [dict(m) for m in (user or {}).get("memberships", [])]
        if user and self.is_bootstrap_admin(user) and not any(m["role"] == "admin" for m in ms):
            ms.insert(0, {"role": "admin", "team_id": None, "player_id": None})
        return ms or [dict(FAN)]

    def is_bootstrap_admin(self, user: dict[str, Any]) -> bool:
        if not self.cfg.admins:
            return False
        if user["subject"].lower() in self.cfg.admins:
            return True
        email = user.get("email")
        return bool(email) and email.lower() in self.cfg.admins and user.get("email_source") != "untrusted"

    def label(self, m: dict[str, Any]) -> str:
        base = ROLE_LABEL.get(m["role"], m["role"])
        if m.get("team_id"):
            t = self.store.get_team(m["team_id"])
            return f"{base} · {t.name if t else m['team_id']}"
        if m.get("player_id"):
            p = self.store.get_player(m["player_id"])
            return f"{base} · {('#%d %s' % (p.number, p.name)) if p else m['player_id']}"
        return base

    # ------------------------------------------------------------ Benutzer
    def get(self, subject: str) -> Optional[dict[str, Any]]:
        return self.store.get_doc(USER_KIND, subject.lower())

    def list(self) -> list[dict[str, Any]]:
        users = list(self.store.list_docs(USER_KIND).values())
        users.sort(key=lambda u: (-(u.get("last_login") or 0), u["subject"]))
        return users

    def find_by_email(self, email: str) -> Optional[dict[str, Any]]:
        email = norm_email(email)
        for u in self.store.list_docs(USER_KIND).values():
            if (u.get("email") or "").lower() == email and u.get("email_source") != "untrusted":
                return u
        return None

    def record_login(self, subject: str, username: Optional[str], codai_email: Optional[str],
                     flow_email: Optional[str]) -> dict[str, Any]:
        """Benutzer anlegen/aktualisieren; Einladungen zur (bekannten) E-Mail anwenden.

        ``codai_email`` stammt von Codai (EMAIL_CLAIM_SERVICES). Fehlt sie, gilt die im Login getippte Adresse
        (``flow_email``) als bestaetigt, wenn ``trust_flow_email`` gesetzt ist -- der Code ging an sie bzw. das Passwort
        gehoert zu ihr. Sonst bleibt die Adresse unbekannt und nur Subject-Zuweisungen greifen."""
        subject = subject.lower()
        now = time.time()
        with self._lock:
            u = self.get(subject)
            if u is None:
                u = {"subject": subject, "username": None, "email": None, "email_source": None,
                     "memberships": [], "created_at": now, "last_login": now}
            u["username"] = username
            if codai_email:
                u["email"], u["email_source"] = norm_email(codai_email), "codai"
            elif flow_email and self.cfg.trust_flow_email and u.get("email_source") != "codai":
                u["email"], u["email_source"] = norm_email(flow_email), "login"
            u["last_login"] = now
            if u.get("email"):
                inv = self.store.get_doc(INVITE_KIND, u["email"])
                if inv:
                    u["memberships"] = self.normalize(list(u["memberships"]) + list(inv.get("memberships", [])))
                    self.store.delete_doc(INVITE_KIND, u["email"])
            self.store.put_doc(USER_KIND, subject, u)
            return u

    def set_memberships(self, subject: str, memberships: list[Any]) -> dict[str, Any]:
        with self._lock:
            u = self.get(subject)
            if u is None:
                raise KeyError(subject)
            u["memberships"] = self.normalize(memberships)
            self.store.put_doc(USER_KIND, u["subject"], u)
            return u

    # ------------------------------------------------------------ Einladungen
    def invite(self, email: str, memberships: list[Any], by: str = "") -> dict[str, Any]:
        """Rollen fuer eine E-Mail vormerken (greift beim ersten Login; existiert der Benutzer schon, sofort)."""
        email = norm_email(email)
        if "@" not in email or " " in email:
            raise MembershipError("Ungueltige E-Mail-Adresse")
        ms = self.normalize(memberships)
        if not ms:
            raise MembershipError("Mindestens eine Mitgliedschaft angeben")
        with self._lock:
            existing = self.find_by_email(email)
            if existing is not None:
                existing["memberships"] = self.normalize(list(existing["memberships"]) + ms)
                self.store.put_doc(USER_KIND, existing["subject"], existing)
                return {"email": email, "memberships": ms, "applied_to": existing["subject"]}
            inv = self.store.get_doc(INVITE_KIND, email) or {"email": email, "memberships": [], "created_at": time.time()}
            inv["memberships"] = self.normalize(list(inv["memberships"]) + ms)
            inv["created_by"] = by
            self.store.put_doc(INVITE_KIND, email, inv)
            return inv

    def invites(self) -> list[dict[str, Any]]:
        return sorted(self.store.list_docs(INVITE_KIND).values(), key=lambda i: i["email"])

    def delete_invite(self, email: str) -> bool:
        return self.store.delete_doc(INVITE_KIND, norm_email(email))
