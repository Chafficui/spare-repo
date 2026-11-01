"""Anmelde-Endpunkte (BFF vor Codai ID) und Benutzerverwaltung.

Browser -> CourtVision (diese Routen) -> Codai (``/service/login/*`` mit Service-Token). Der Browser sieht weder das
Service-Token noch das Codai-``flow``: das liegt serverseitig (docs-Art ``loginflow``) hinter dem signierten Cookie ``cv_login``.

====================================  ================================================================
``GET  /auth/config``                 Modus, Links; setzt das CSRF-Cookie
``GET  /auth/me``                     aktuelle Identität, Mitgliedschaften
``POST /auth/code/start``             ``{email, locale?}`` -> Code per Mail
``POST /auth/code/resend``            Code erneut senden (30 s Wartezeit)
``POST /auth/code/verify``            ``{code}`` -> Sitzung
``POST /auth/code/cancel``            Flow verwerfen ("Andere E-Mail verwenden")
``POST /auth/password``               ``{email, password}`` -> Sitzung oder ``{step_up: true}`` (neues Gerät: Code)
``POST /auth/role``                   ``{role, team_id?, player_id?}`` aktive Rolle wechseln
``POST /auth/logout``
``GET  /admin/users``                 Benutzer + Einladungen (admin)
``PUT  /admin/users/{subject}/memberships``  ``{memberships: [...]}`` (admin)
``POST /admin/invites``               ``{email, memberships}`` (admin); ``DELETE /admin/invites/{email}``
====================================  ================================================================

Fehler: ``{"detail": {"code": ..., "message": <deutsch>, ...}}`` (``bad_code`` mit ``attempts_left``,
``rate_limited``/``resend_wait`` mit ``retry_after``).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..auth import codai as cx
from ..auth.service import EMAIL_RE, AuthService
from ..auth.session import CSRF_COOKIE, DEVICE_COOKIE, FLOW_KIND, LOGIN_COOKIE, SESSION_COOKIE
from ..auth.users import MembershipError, m_key
from .auth import Principal, current_principal, require

log = logging.getLogger("courtvision.auth")
DEVICE_MAX_AGE = 365 * 86400

MESSAGES = {
    "bad_email": "Das sieht nicht wie eine E-Mail-Adresse aus.",
    "rate_limited": "Zu viele Versuche. Bitte warte einen Moment und versuche es dann erneut.",
    "resend_wait": "Bitte warte kurz, bevor du einen neuen Code anforderst.",
    "mail_failed": "Die E-Mail mit dem Code konnte nicht gesendet werden. Bitte versuche es später noch einmal.",
    "bad_code": "Der Code stimmt nicht.",
    "expired": "Der Code ist abgelaufen oder wurde schon benutzt. Bitte fordere einen neuen Code an.",
    "bad_credentials": "E-Mail-Adresse oder Passwort stimmen nicht.",
    "busy": "Die Anmeldung ist gerade ausgelastet. Bitte versuche es gleich noch einmal.",
    "unavailable": "Die Anmeldung ist gerade nicht erreichbar. Bitte versuche es gleich noch einmal.",
    "misconfigured": "Die Anmeldung ist nicht richtig eingerichtet. Bitte informiere den Betreiber.",
    "not_configured": "Die Anmeldung ist auf diesem Server nicht eingerichtet.",
    "no_flow": "Es läuft keine Anmeldung (mehr). Bitte gib deine E-Mail-Adresse erneut ein.",
    "bad_device": "Das Gerät konnte nicht erkannt werden. Bitte lade die Seite neu.",
}


def fail(status: int, code: str, message: Optional[str] = None, **extra: Any) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message or MESSAGES.get(code, code), **extra})


def _codai_http(e: cx.CodaiError) -> HTTPException:
    """Codai-Fehler -> Antwort an den Browser."""
    if isinstance(e, cx.Misconfigured):
        log.error("Codai-Konfiguration: %s", e)
        return fail(503, "misconfigured")
    if isinstance(e, cx.BadCode):
        extra = {"attempts_left": e.attempts_left} if e.attempts_left is not None else {}
        return fail(400, "bad_code", **extra)
    if isinstance(e, (cx.BadEmail, cx.Expired, cx.BadCredentials, cx.RateLimited, cx.MailFailed, cx.Busy,
                      cx.Unavailable, cx.BadDevice)):
        extra = {"retry_after": 60} if isinstance(e, cx.RateLimited) else {}
        if isinstance(e, cx.Unavailable):
            log.warning("Codai nicht verfügbar: %s", e)
        return fail(e.status if e.status < 600 else 503, e.code, **extra)
    log.error("Unerwarteter Codai-Fehler: %s", e)
    return fail(503, "unavailable")


class StartIn(BaseModel):
    email: str = Field(max_length=320)
    locale: str = "de"


class VerifyIn(BaseModel):
    code: str = Field(max_length=64)


class PasswordIn(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1024)
    locale: str = "de"


class RoleIn(BaseModel):
    role: str
    team_id: Optional[str] = None
    player_id: Optional[str] = None


class MembershipsIn(BaseModel):
    memberships: list[dict[str, Any]]


class InviteIn(BaseModel):
    email: str = Field(max_length=320)
    memberships: list[dict[str, Any]]


def build(state, auth: AuthService) -> APIRouter:
    r = APIRouter()
    cfg = auth.cfg
    store = auth.store

    # ------------------------------------------------------------ Cookies
    def put_cookie(resp: Response, name: str, value: str, max_age: Optional[int], *, httponly: bool = True) -> None:
        resp.set_cookie(name, value, max_age=max_age, path="/", secure=not cfg.insecure_cookies,
                        httponly=httponly, samesite="lax")

    def drop_cookie(resp: Response, name: str, *, httponly: bool = True) -> None:
        resp.delete_cookie(name, path="/", secure=not cfg.insecure_cookies, httponly=httponly, samesite="lax")

    def ensure_csrf(request: Request, resp: Response, session_id: str = "", rotate: bool = False) -> str:
        cur = request.cookies.get(CSRF_COOKIE)
        if not rotate and auth.signer.csrf_valid(cur, session_id):
            return cur  # type: ignore[return-value]
        tok = auth.signer.csrf_token(session_id)
        put_cookie(resp, CSRF_COOKIE, tok, cfg.abs_ttl_s, httponly=False)
        return tok

    def device_id(request: Request, resp: Response) -> str:
        d = request.cookies.get(DEVICE_COOKIE)
        if not d or not (16 <= len(d) <= 128):
            d = auth.new_device_id()
            put_cookie(resp, DEVICE_COOKIE, d, DEVICE_MAX_AGE)
        return d

    # ------------------------------------------------------------ Login-Flow (serverseitig)
    def load_flow(request: Request) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        fid = auth.signer.unsign("loginflow", request.cookies.get(LOGIN_COOKIE))
        if not fid:
            return None, None
        doc = store.get_doc(FLOW_KIND, fid)
        if doc is None or doc["created_at"] + cfg.flow_ttl_s < time.time():
            if doc is not None:
                store.delete_doc(FLOW_KIND, fid)
            return None, None
        return fid, doc

    def save_flow(resp: Response, fid: Optional[str], doc: dict[str, Any]) -> None:
        import secrets

        fid = fid or secrets.token_urlsafe(24)
        store.put_doc(FLOW_KIND, fid, doc)
        put_cookie(resp, LOGIN_COOKIE, auth.signer.sign("loginflow", fid), cfg.flow_ttl_s)

    def end_flow(resp: Response, fid: Optional[str]) -> None:
        if fid:
            store.delete_doc(FLOW_KIND, fid)
        drop_cookie(resp, LOGIN_COOKIE)

    def need_codai() -> cx.CodaiClient:
        if auth.codai is None:
            raise fail(503, "not_configured")
        return auth.codai

    def limit(limiter, key: str) -> None:
        if not limiter.hit(key):
            raise fail(429, "rate_limited", retry_after=limiter.retry_after(key))

    def clean_email(raw: str) -> str:
        email = raw.strip().lower()
        if not EMAIL_RE.match(email):
            raise fail(400, "bad_email")
        return email

    def complete_login(request: Request, resp: Response, ident: cx.Identity, flow_email: Optional[str]) -> dict[str, Any]:
        user = auth.users.record_login(ident.subject, ident.username, ident.email, flow_email)
        auth.sessions.delete(request.cookies.get(SESSION_COOKIE))  # keine Session-Fixation: immer neue ID
        sid = auth.sessions.create(user["subject"], auth.client_ip(request), auth.user_agent(request))
        put_cookie(resp, SESSION_COOKIE, sid, cfg.abs_ttl_s)
        ensure_csrf(request, resp, sid, rotate=True)
        return auth.me(user, None)

    # ------------------------------------------------------------ Abfragen
    @r.get("/auth/config")
    def get_config(request: Request, response: Response):
        sid, _ = auth.session_for(request.cookies)
        ensure_csrf(request, response, sid or "")
        response.headers["Cache-Control"] = "no-store"
        return {"mode": cfg.mode, "codai_url": cfg.public_url, "account_url": cfg.account_url,
                "password_reset_url": cfg.password_reset_url, "demo": cfg.demo, "configured": auth.codai is not None,
                "code_length": 6, "resend_wait_s": cfg.resend_wait_s,
                "demo_accounts": [{"email": e, "label": l} for e, l in cfg.demo_accounts]}

    @r.get("/auth/me")
    def me(request: Request, response: Response, p: Principal = Depends(current_principal)):
        sid, session = auth.session_for(request.cookies)
        response.headers["Cache-Control"] = "no-store"
        if p.source == "token":
            return {"mode": cfg.mode, "authenticated": True, "subject": "system", "username": "system",
                    "role": "system", "team_id": None, "player_id": None, "memberships": [], "is_admin": False,
                    "label": "System"}
        out: dict[str, Any]
        if session is None or p.source != "session":
            out = auth.me(None, None)
        else:
            user = auth.users.get(session["subject"])
            out = auth.me(user, {"role": p.role, "team_id": p.team_id, "player_id": p.player_id})
            out["csrf"] = ensure_csrf(request, response, sid or "")
        return out

    # ------------------------------------------------------------ Code-Login
    @r.post("/auth/code/start")
    def code_start(body: StartIn, request: Request, response: Response):
        client = need_codai()
        email = clean_email(body.email)
        ip = auth.client_ip(request) or "-"
        limit(auth.lim_start_ip, ip)
        limit(auth.lim_start_email, email)
        fid, old = load_flow(request)
        if old and old["email"] == email and time.time() - old["sent_at"] < cfg.resend_wait_s:
            raise fail(429, "resend_wait", retry_after=int(cfg.resend_wait_s - (time.time() - old["sent_at"])) + 1)
        dev = device_id(request, response)
        try:
            res = client.start_code(email, body.locale, ip, auth.user_agent(request), dev)
        except cx.CodaiError as e:
            raise _codai_http(e)
        now = time.time()
        save_flow(response, fid, {"flow": res.flow, "email": email, "kind": "code", "created_at": now, "sent_at": now,
                                  "code_length": res.code_length, "locale": body.locale})
        return {"email": email, "expires_at": res.expires_at, "code_length": res.code_length,
                "resend_after": cfg.resend_wait_s}

    @r.post("/auth/code/resend")
    def code_resend(request: Request, response: Response):
        client = need_codai()
        fid, doc = load_flow(request)
        if doc is None:
            raise fail(410, "no_flow")
        if doc["kind"] == "stepup":
            raise fail(409, "no_flow", "Bitte gib dein Passwort noch einmal ein, um einen neuen Code zu erhalten.")
        wait = cfg.resend_wait_s - (time.time() - doc["sent_at"])
        if wait > 0:
            raise fail(429, "resend_wait", retry_after=int(wait) + 1)
        ip = auth.client_ip(request) or "-"
        limit(auth.lim_start_ip, ip)
        limit(auth.lim_start_email, doc["email"])
        dev = device_id(request, response)
        try:
            res = client.start_code(doc["email"], doc.get("locale", "de"), ip, auth.user_agent(request), dev)
        except cx.CodaiError as e:
            raise _codai_http(e)
        now = time.time()
        save_flow(response, fid, {**doc, "flow": res.flow, "created_at": now, "sent_at": now,
                                  "code_length": res.code_length})
        return {"email": doc["email"], "expires_at": res.expires_at, "code_length": res.code_length,
                "resend_after": cfg.resend_wait_s}

    @r.post("/auth/code/verify")
    def code_verify(body: VerifyIn, request: Request, response: Response):
        client = need_codai()
        fid, doc = load_flow(request)
        if doc is None:
            raise fail(410, "no_flow")
        ip = auth.client_ip(request) or "-"
        limit(auth.lim_verify_ip, ip)
        code = "".join(ch for ch in body.code if ch.isdigit())  # "123 456" -> "123456"
        if len(code) != doc.get("code_length", 6):
            raise fail(400, "bad_code", f"Bitte gib den {doc.get('code_length', 6)}-stelligen Code ein.")
        dev = device_id(request, response)
        try:
            ident = client.verify_code(doc["flow"], code, ip, auth.user_agent(request), dev)
        except cx.BadCode as e:
            if e.attempts_left == 0:  # fuenfter Fehlversuch beendet den Flow bei Codai
                end_flow(response, fid)
                raise fail(410, "expired", "Zu viele falsche Codes. Bitte fordere einen neuen Code an.")
            raise _codai_http(e)
        except cx.Expired as e:
            end_flow(response, fid)
            raise _codai_http(e)
        except cx.CodaiError as e:
            raise _codai_http(e)
        end_flow(response, fid)
        return complete_login(request, response, ident, doc["email"])

    @r.post("/auth/code/cancel")
    def code_cancel(request: Request, response: Response):
        fid, _ = load_flow(request)
        end_flow(response, fid)
        return {"ok": True}

    # ------------------------------------------------------------ Passwort-Login
    @r.post("/auth/password")
    def password(body: PasswordIn, request: Request, response: Response):
        client = need_codai()
        email = clean_email(body.email) if EMAIL_RE.match(body.email.strip().lower()) else ""
        ip = auth.client_ip(request) or "-"
        limit(auth.lim_pw_ip, ip)
        if not email:
            raise fail(401, "bad_credentials")
        limit(auth.lim_pw_email, email)
        dev = device_id(request, response)
        try:
            res = client.password_login(email, body.password, body.locale, ip, auth.user_agent(request), dev)
        except cx.CodaiError as e:
            raise _codai_http(e)
        if isinstance(res, cx.StepUp):  # richtiges Passwort, neues Geraet -> Code bestaetigen
            fid, _ = load_flow(request)
            now = time.time()
            save_flow(response, fid, {"flow": res.flow, "email": email, "kind": "stepup", "created_at": now,
                                      "sent_at": now, "code_length": res.code_length, "locale": body.locale})
            return {"step_up": True, "email": email, "expires_at": res.expires_at, "code_length": res.code_length,
                    "resend_after": cfg.resend_wait_s}
        return complete_login(request, response, res, email)

    # ------------------------------------------------------------ Sitzung
    @r.post("/auth/role")
    def switch_role(body: RoleIn, request: Request, response: Response):
        sid, session = auth.session_for(request.cookies)
        if session is None:
            raise fail(401, "unauthenticated", "Bitte anmelden")
        user = auth.users.get(session["subject"])
        m = auth.resolve_membership(user, session, body.role, body.team_id, body.player_id)
        if m is None:
            raise fail(403, "not_member", "Diese Rolle gehört nicht zu deinen Mitgliedschaften")
        auth.sessions.set_active(sid, {"role": m["role"], "team_id": m.get("team_id"), "player_id": m.get("player_id")})
        return auth.me(user, m)

    @r.post("/auth/logout")
    def logout(request: Request, response: Response):
        auth.sessions.delete(request.cookies.get(SESSION_COOKIE))
        drop_cookie(response, SESSION_COOKIE)
        ensure_csrf(request, response, "", rotate=True)
        return {"ok": True}

    # ------------------------------------------------------------ Admin
    def user_view(u: dict[str, Any]) -> dict[str, Any]:
        eff = auth.users.effective_memberships(u)
        return {**u, "effective": [{**m, "label": auth.users.label(m)} for m in eff],
                "bootstrap_admin": auth.users.is_bootstrap_admin(u)}

    @r.get("/admin/users")
    def admin_users(_: Principal = Depends(require())):
        return {"users": [user_view(u) for u in auth.users.list()], "invites": auth.users.invites(),
                "roles": ["admin", "hallenwart", "kampfgericht", "trainer", "spieler", "verband"],
                "email_trust": cfg.trust_flow_email}

    @r.put("/admin/users/{subject}/memberships")
    def admin_set_memberships(subject: str, body: MembershipsIn, p: Principal = Depends(require())):
        try:
            u = auth.users.set_memberships(subject, body.memberships)
        except KeyError:
            raise HTTPException(404, "Benutzer nicht gefunden")
        except MembershipError as e:
            raise HTTPException(422, str(e))
        log.info("Mitgliedschaften von %s gesetzt durch %s: %s", subject, p.subject, [m_key(m) for m in u["memberships"]])
        return user_view(u)

    @r.post("/admin/invites", status_code=201)
    def admin_invite(body: InviteIn, p: Principal = Depends(require())):
        try:
            return auth.users.invite(body.email, body.memberships, by=p.subject or p.role)
        except MembershipError as e:
            raise HTTPException(422, str(e))

    @r.delete("/admin/invites/{email}")
    def admin_delete_invite(email: str, _: Principal = Depends(require())):
        if not auth.users.delete_invite(email):
            raise HTTPException(404, "Einladung nicht gefunden")
        return {"ok": True}

    return r
