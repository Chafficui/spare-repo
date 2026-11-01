"""Anmeldung per Codai ID (BFF): CodaiClient gegen Fake-Codai, Login-Ablauf, Sitzungen, CSRF, Rollen, Admin, Token, WebSocket."""
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from courtvision.api.app import create_app
from courtvision.auth import codai as cx
from courtvision.auth.codai import CodaiClient
from courtvision.auth.config import AuthConfig
from courtvision.auth.fake_codai import DEFAULT_SERVICE, DEFAULT_TOKEN, FakeCodai
from courtvision.sim import demo_setup, load_demo
from courtvision.store.eventstore import EventStore

SYS_TOKEN = "system-token-for-the-edge-pipeline-0123456789"
ADMIN_EMAIL = "chef@club.example"
CODE = "123456"


def fake_codai(**kw):
    kw.setdefault("fixed_code", CODE)
    kw.setdefault("email_claim", {DEFAULT_SERVICE})
    kw.setdefault("auto_usernames", True)
    return FakeCodai(**kw)


def codai_client(fake, **kw):
    http = TestClient(fake.app, base_url="http://fake-codai")
    return CodaiClient("http://fake-codai", kw.pop("token", DEFAULT_TOKEN), http=http, **kw)


def make_app(fake=None, base_url="http://testserver", store=None, **cfg_kw):
    fake = fake or fake_codai()
    store = store or EventStore(":memory:")
    if store.get_game("demo-game-1") is None:
        load_demo(store, demo_setup())
    cfg_kw.setdefault("insecure_cookies", True)
    cfg_kw.setdefault("admins", frozenset({ADMIN_EMAIL}))
    cfg_kw.setdefault("system_tokens", (SYS_TOKEN,))
    cfg = AuthConfig(mode="codai", **cfg_kw)
    app = create_app(store, auth_config=cfg, fake_codai=fake)
    return app, fake, TestClient(app, base_url=base_url)


def csrf(c):
    return {"X-CSRF-Token": c.cookies.get("cv_csrf")}


def start(c, email, **kw):
    c.get("/auth/config")  # CSRF-Cookie holen
    return c.post("/auth/code/start", json={"email": email, **kw}, headers=csrf(c))


def login(c, fake, email):
    r = start(c, email)
    assert r.status_code == 200, r.text
    r = c.post("/auth/code/verify", json={"code": fake.last_code(email)}, headers=csrf(c))
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ CodaiClient <-> Fake-Codai (Vertrag)
def test_client_code_flow_contract():
    fake = fake_codai()
    cl = codai_client(fake)
    st = cl.start_code("Anna@Example.com ", "de", ip="203.0.113.7", user_agent="UA/1")
    assert st.code_length == 6 and len(st.flow) == 43 and st.expires_at > time.time() * 1000
    assert fake.outbox[-1]["to"] == "anna@example.com"
    ident = cl.verify_code(st.flow, CODE, ip="203.0.113.7", user_agent="UA/1")
    assert ident.created and ident.subject == fake.subject_for("anna@example.com")
    assert ident.username == "anna" and ident.email == "anna@example.com"  # EMAIL_CLAIM
    # IP und User-Agent des Nutzers kamen bei Codai an
    assert fake.requests[0]["ip"] == "203.0.113.7" and fake.requests[0]["userAgent"] == "UA/1"
    assert fake.sign_ins[-1]["ip"] == "203.0.113.7" and fake.sign_ins[-1]["method"] == "email_code"
    # zweiter Login: created=False; Flow ist verbraucht
    st2 = cl.start_code("anna@example.com")
    assert cl.verify_code(st2.flow, CODE).created is False
    with pytest.raises(cx.Expired):
        cl.verify_code(st2.flow, CODE)
    assert cl.get_user(ident.subject).username == "anna"
    with pytest.raises(cx.UnknownUser):
        cl.get_user("00000000-0000-4000-8000-000000000000")


def test_client_without_email_claim_and_invalid_ip_dropped():
    fake = fake_codai(email_claim=set(), auto_usernames=False)
    cl = codai_client(fake)
    st = cl.start_code("b@example.com", ip="not-an-ip")
    ident = cl.verify_code(st.flow, CODE)
    assert ident.email is None and ident.username is None
    assert "ip" not in fake.requests[0]


def test_client_errors():
    fake = fake_codai(mail_fail={"down@example.com"})
    cl = codai_client(fake)
    with pytest.raises(cx.BadEmail):
        cl.start_code("kein-email")
    with pytest.raises(cx.MailFailed):
        cl.start_code("down@example.com")
    st = cl.start_code("c@example.com")
    with pytest.raises(cx.BadCode) as e:
        cl.verify_code(st.flow, "000000")
    assert e.value.attempts_left == 4
    for left in (3, 2, 1, 0):
        with pytest.raises(cx.BadCode) as e:
            cl.verify_code(st.flow, "000000")
        assert e.value.attempts_left == left
    with pytest.raises(cx.Expired):  # fuenfter Fehlversuch beendet den Flow, auch der richtige Code geht nicht mehr
        cl.verify_code(st.flow, CODE)
    with pytest.raises(cx.Expired):
        cl.verify_code("x" * 43, CODE)
    # Limit: 5 Codes je Adresse
    for _ in range(5):
        cl.start_code("lim@example.com")
    with pytest.raises(cx.RateLimited):
        cl.start_code("lim@example.com")
    # Service-Token falsch / Dienst nicht freigeschaltet
    with pytest.raises(cx.Misconfigured) as e:
        codai_client(fake, token="x" * 40).start_code("d@example.com")
    assert e.value.code == "unauthorized"
    other = FakeCodai(tokens={"other": "o" * 40}, headless=set())
    with pytest.raises(cx.Misconfigured) as e:
        codai_client(other, token="o" * 40).start_code("d@example.com")
    assert e.value.code == "forbidden"
    # Netzwerkfehler -> Unavailable
    dead = CodaiClient("http://127.0.0.1:1", DEFAULT_TOKEN, timeout=0.5)
    with pytest.raises(cx.Unavailable):
        dead.start_code("e@example.com")


def test_client_code_expires_after_ten_minutes():
    now = [1_000_000.0]
    fake = fake_codai(clock=lambda: now[0])
    cl = codai_client(fake)
    st = cl.start_code("t@example.com")
    now[0] += 601
    with pytest.raises(cx.Expired):
        cl.verify_code(st.flow, CODE)


def test_client_password_and_step_up():
    fake = fake_codai(passwords={"p@example.com": "geheim"})
    cl = codai_client(fake)
    with pytest.raises(cx.BadCredentials):
        cl.password_login("p@example.com", "falsch")
    with pytest.raises(cx.BadCredentials):
        cl.password_login("unbekannt@example.com", "geheim")
    # neues Geraet -> Step-up per Code
    step = cl.password_login("p@example.com", "geheim", device="dev-" + "a" * 20)
    assert isinstance(step, cx.StepUp)
    ident = cl.verify_code(step.flow, CODE, device="dev-" + "a" * 20)
    assert ident.subject == fake.subject_for("p@example.com") and fake.sign_ins[-1]["method"] == "password"
    # bekanntes Geraet -> direkt
    again = cl.password_login("p@example.com", "geheim", device="dev-" + "a" * 20)
    assert isinstance(again, cx.Identity) and again.subject == ident.subject
    # Step-up-Code gilt nur auf dem Geraet, das ihn angefordert hat
    step2 = cl.password_login("p@example.com", "geheim", device="dev-" + "b" * 20)
    with pytest.raises(cx.Expired):
        cl.verify_code(step2.flow, CODE, device="dev-" + "c" * 20)
    with pytest.raises(cx.BadDevice):
        cl.password_login("p@example.com", "geheim", device="zu kurz!")


def test_client_password_ip_limit():
    fake = fake_codai(passwords={"p@example.com": "geheim"})
    cl = codai_client(fake)
    for _ in range(20):
        with pytest.raises(cx.BadCredentials):
            cl.password_login("x@example.com", "nein", ip="198.51.100.9")
    with pytest.raises(cx.RateLimited):
        cl.password_login("x@example.com", "nein", ip="198.51.100.9")


# ------------------------------------------------------------------ Login-Ablauf ueber die API
def test_login_flow_end_to_end():
    app, fake, c = make_app()
    # anonym: oeffentliche Seiten ja, geschuetzte 401
    assert c.get("/auth/me").json()["authenticated"] is False
    assert c.get("/games").status_code == 200
    assert c.post("/games/demo-game-1/start", json={"starters": {"home": [], "away": []}}).status_code in (401, 403)
    assert c.get("/admin/users").status_code == 401

    me = login(c, fake, "neu@example.com")
    assert me["authenticated"] and me["role"] == "fan" and me["username"] == "neu"
    assert me["memberships"][0]["role"] == "fan"
    assert c.get("/auth/me").json()["subject"] == fake.subject_for("neu@example.com")
    # kein Admin: 403 (angemeldet, aber nicht berechtigt)
    assert c.get("/admin/users").status_code == 403
    # Logout loescht die Sitzung serverseitig
    sid = c.cookies.get("cv_session")
    assert c.post("/auth/logout", headers=csrf(c)).status_code == 200
    assert c.get("/auth/me").json()["authenticated"] is False
    c.cookies.set("cv_session", sid)
    assert c.get("/auth/me").json()["authenticated"] is False  # altes Cookie ist wertlos
    assert app.state.store.list_docs("session") == {}


def test_role_switch_limited_to_memberships():
    app, fake, c = make_app()
    auth = app.state.auth
    team = "team-dessau"
    login(c, fake, "multi@example.com")
    sub = fake.subject_for("multi@example.com")
    # Fan darf sich nichts geben
    assert c.post("/auth/role", json={"role": "admin"}, headers=csrf(c)).status_code == 403
    assert c.get("/admin/users", headers={"X-Active-Role": "admin"}).status_code == 403
    auth.users.set_memberships(sub, [{"role": "trainer", "team_id": team}, {"role": "kampfgericht"}])
    me = c.get("/auth/me").json()
    assert me["role"] == "trainer" and me["team_id"] == team and len(me["memberships"]) == 2
    assert [m["active"] for m in me["memberships"]] == [True, False]
    # nicht-Mitglieds-Rolle / fremdes Team verweigert
    assert c.post("/auth/role", json={"role": "admin"}, headers=csrf(c)).status_code == 403
    assert c.post("/auth/role", json={"role": "trainer", "team_id": "team-halle"}, headers=csrf(c)).status_code == 403
    assert c.get("/auth/me", headers={"X-Active-Role": "trainer", "X-Active-Team": "team-halle"}).status_code == 403
    # Wechsel auf eine eigene Mitgliedschaft wirkt auf die Sitzung
    r = c.post("/auth/role", json={"role": "kampfgericht"}, headers=csrf(c))
    assert r.status_code == 200 and r.json()["role"] == "kampfgericht"
    assert c.get("/auth/me").json()["role"] == "kampfgericht"
    # Header gilt je Anfrage und ueberstimmt die Sitzung
    assert c.get("/auth/me", headers={"X-Active-Role": "trainer"}).json()["role"] == "trainer"
    # ohne CSRF-Header kein Wechsel
    assert c.post("/auth/role", json={"role": "trainer", "team_id": team}).status_code == 403


def test_protected_endpoint_with_session_role():
    app, fake, c = make_app()
    login(c, fake, "kg@example.com")
    sub = fake.subject_for("kg@example.com")
    gid = "demo-game-1"
    body = {"type": "TIMEOUT", "team": "home", "period": 1, "game_clock": 500, "video_ts": 90}
    assert c.post(f"/games/{gid}/events", json=body, headers=csrf(c)).status_code == 403  # Fan
    app.state.auth.users.set_memberships(sub, [{"role": "kampfgericht"}])
    r = c.post(f"/games/{gid}/events", json=body, headers=csrf(c))
    assert r.status_code == 201, r.text


def test_password_flow_with_step_up_and_device_cookie():
    fake = fake_codai(passwords={"pw@example.com": "geheim-123"})
    app, fake, c = make_app(fake)
    c.get("/auth/config")
    r = c.post("/auth/password", json={"email": "pw@example.com", "password": "falsch"}, headers=csrf(c))
    assert r.status_code == 401 and r.json()["detail"]["code"] == "bad_credentials"
    # neues Geraet -> Code zur Bestaetigung
    r = c.post("/auth/password", json={"email": "pw@example.com", "password": "geheim-123"}, headers=csrf(c))
    assert r.status_code == 200 and r.json()["step_up"] is True
    assert c.get("/auth/me").json()["authenticated"] is False
    r = c.post("/auth/code/verify", json={"code": fake.last_code("pw@example.com")}, headers=csrf(c))
    assert r.status_code == 200 and r.json()["authenticated"]
    c.post("/auth/logout", headers=csrf(c))
    # dasselbe Browser-Geraet (cv_device) ist jetzt bekannt: direkter Login
    r = c.post("/auth/password", json={"email": "pw@example.com", "password": "geheim-123"}, headers=csrf(c))
    assert r.status_code == 200 and r.json()["authenticated"] and r.json()["email"] == "pw@example.com"
    # anderer Browser -> wieder Step-up
    c2 = TestClient(app)
    c2.get("/auth/config")
    r = c2.post("/auth/password", json={"email": "pw@example.com", "password": "geheim-123"}, headers=csrf(c2))
    assert r.json().get("step_up") is True
    # Resend ist fuer Step-up nicht moeglich (Passwort noetig)
    assert c2.post("/auth/code/resend", headers=csrf(c2)).status_code == 409


def test_code_errors_through_api():
    app, fake, c = make_app(fake_codai(mail_fail={"down@example.com"}))
    r = start(c, "kaputt")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "bad_email"
    r = start(c, "down@example.com")
    assert r.status_code == 502 and r.json()["detail"]["code"] == "mail_failed"
    assert start(c, "x@example.com").status_code == 200
    # falscher Code mit verbleibenden Versuchen
    r = c.post("/auth/code/verify", json={"code": "000 000"}, headers=csrf(c))
    assert r.status_code == 400 and r.json()["detail"] == {"code": "bad_code", "message": "Der Code stimmt nicht.",
                                                          "attempts_left": 4}
    # zu kurz: lokal abgelehnt, kostet keinen Versuch bei Codai
    assert c.post("/auth/code/verify", json={"code": "12"}, headers=csrf(c)).status_code == 400
    r = c.post("/auth/code/verify", json={"code": "000000"}, headers=csrf(c))
    assert r.json()["detail"]["attempts_left"] == 3
    # Leerzeichen im Code ("123 456") werden bereinigt
    r = c.post("/auth/code/verify", json={"code": " 123 456 "}, headers=csrf(c))
    assert r.status_code == 200
    # fuenf Fehlversuche -> Flow tot (410), neuer Code noetig
    assert start(c, "y@example.com").status_code == 200
    for _ in range(4):
        assert c.post("/auth/code/verify", json={"code": "000000"}, headers=csrf(c)).status_code == 400
    r = c.post("/auth/code/verify", json={"code": "000000"}, headers=csrf(c))
    assert r.status_code == 410 and r.json()["detail"]["code"] == "expired"
    assert c.post("/auth/code/verify", json={"code": CODE}, headers=csrf(c)).json()["detail"]["code"] == "no_flow"
    # Codai-Limit (5 Codes je Adresse) kommt als 429 an -- eigener Browser, damit unsere Limits nicht greifen
    app, fake, c = make_app(start_per_email=50, start_per_ip=50)
    for i in range(5):
        c2 = TestClient(app)
        assert start(c2, "viel@example.com").status_code == 200
    r = start(TestClient(app), "viel@example.com")
    assert r.status_code == 429 and r.json()["detail"]["code"] == "rate_limited"


def test_resend_wait_and_cancel():
    now = [time.time()]
    app, fake, c = make_app()
    assert start(c, "r@example.com").status_code == 200
    r = c.post("/auth/code/resend", headers=csrf(c))
    assert r.status_code == 429 and r.json()["detail"]["code"] == "resend_wait" and 1 <= r.json()["detail"]["retry_after"] <= 31
    assert start(c, "r@example.com").status_code == 429  # auch "neu starten" mit gleicher Adresse
    # nach 30 s geht es (Zeit im Flow-Dokument zurueckdrehen)
    fid, doc = next(iter(app.state.store.list_docs("loginflow").items()))
    doc["sent_at"] -= 31
    app.state.store.put_doc("loginflow", fid, doc)
    assert c.post("/auth/code/resend", headers=csrf(c)).status_code == 200
    assert len(fake.outbox) == 2
    # "Andere E-Mail verwenden"
    assert c.post("/auth/code/cancel", headers=csrf(c)).status_code == 200
    assert app.state.store.list_docs("loginflow") == {}
    assert c.post("/auth/code/verify", json={"code": CODE}, headers=csrf(c)).status_code == 410


def test_our_own_login_rate_limits():
    app, fake, c = make_app(start_per_ip=3, start_per_email=100, password_per_ip=2)
    for i in range(3):
        assert start(c, f"u{i}@example.com").status_code == 200
    r = start(c, "u9@example.com")
    assert r.status_code == 429 and r.json()["detail"]["retry_after"] > 0
    assert len(fake.outbox) == 3  # Codai wurde nicht mehr behelligt
    for _ in range(2):
        assert c.post("/auth/password", json={"email": "a@b.de", "password": "x"}, headers=csrf(c)).status_code == 401
    assert c.post("/auth/password", json={"email": "a@b.de", "password": "x"}, headers=csrf(c)).status_code == 429


def test_client_ip_forwarded_only_behind_trusted_proxy():
    app, fake, c = make_app()
    start(c, "ip1@example.com")  # Codai sieht die TestClient-Adresse, nicht den Header
    c.post("/auth/code/cancel", headers=csrf(c))
    r = c.post("/auth/code/start", json={"email": "ip2@example.com"},
               headers={**csrf(c), "X-Forwarded-For": "6.6.6.6", "User-Agent": "Mozilla/9"})
    assert r.status_code == 200
    assert fake.requests[-1].get("ip") != "6.6.6.6"
    assert fake.requests[-1]["userAgent"] == "Mozilla/9"
    app, fake, c = make_app(trust_proxy=True)
    c.get("/auth/config")
    c.post("/auth/code/start", json={"email": "ip3@example.com"},
           headers={**csrf(c), "X-Forwarded-For": "6.6.6.6, 203.0.113.5"})  # ein Proxy: rechter Eintrag zaehlt
    assert fake.requests[-1]["ip"] == "203.0.113.5"


# ------------------------------------------------------------------ Cookies, Flow-Bindung, CSRF
def test_cookie_flags():
    app, fake, c = make_app(base_url="https://testserver", insecure_cookies=False)
    r1 = c.get("/auth/config")
    csrf_hdr = r1.headers["set-cookie"]
    assert "cv_csrf=" in csrf_hdr and "httponly" not in csrf_hdr.lower() and "Secure" in csrf_hdr
    r = start(c, "flags@example.com")
    flow = r.headers["set-cookie"].lower()
    assert "cv_login=" in flow and "httponly" in flow and "secure" in flow and "samesite=lax" in flow and "max-age=600" in flow
    r = c.post("/auth/code/verify", json={"code": CODE}, headers=csrf(c))
    cookies = {x.split("=", 1)[0]: x for x in r.headers.get_list("set-cookie")}
    sess = cookies["cv_session"].lower()
    assert "httponly" in sess and "secure" in sess and "samesite=lax" in sess and "path=/" in sess
    assert "httponly" not in cookies["cv_csrf"].lower()
    assert len(c.cookies["cv_session"]) >= 43  # 256 Bit
    # lokal ueber http (COURTVISION_INSECURE_COOKIES=1): kein Secure
    app, fake, c = make_app()
    r = start(c, "local@example.com")
    assert "secure" not in r.headers["set-cookie"].lower()


def test_flow_cookie_binds_flow_to_browser():
    app, fake, c = make_app()
    r = start(c, "bound@example.com")
    flow_id = fake.outbox[-1]["flow"]
    # das Codai-Flow taucht nirgends in Antworten/Cookies des Browsers auf
    assert flow_id not in r.text and flow_id not in str(r.headers) and flow_id not in str(c.cookies.items())
    code = fake.last_code("bound@example.com")
    # anderer Browser mit dem richtigen Code, aber ohne cv_login -> kein Login
    other = TestClient(app)
    other.get("/auth/config")
    r = other.post("/auth/code/verify", json={"code": code}, headers=csrf(other))
    assert r.status_code == 410 and r.json()["detail"]["code"] == "no_flow"
    # gefaelschtes / manipuliertes Cookie
    forged = c.cookies["cv_login"]
    other.cookies.set("cv_login", forged[:-3] + "AAA")
    assert other.post("/auth/code/verify", json={"code": code}, headers=csrf(other)).status_code == 410
    other.cookies.set("cv_login", "abc." + forged.split(".")[1])
    assert other.post("/auth/code/verify", json={"code": code}, headers=csrf(other)).status_code == 410
    # ein Cookie mit fremdem Schluessel (andere Instanz) ist ungueltig
    app2, _, c2 = make_app(secret="zweite-instanz-geheimnis")
    start(c2, "bound@example.com")
    other.cookies.set("cv_login", c2.cookies["cv_login"])
    assert other.post("/auth/code/verify", json={"code": code}, headers=csrf(other)).status_code == 410
    assert c.post("/auth/code/verify", json={"code": code}, headers=csrf(c)).status_code == 200


def test_csrf_protection():
    app, fake, c = make_app()
    login(c, fake, "csrf@example.com")
    sub = fake.subject_for("csrf@example.com")
    app.state.auth.users.set_memberships(sub, [{"role": "admin"}])
    body = {"memberships": [{"role": "kampfgericht"}]}
    url = f"/admin/users/{sub}/memberships"
    # ohne Header, falscher Header, Header ohne Cookie, Token einer anderen Sitzung
    assert c.put(url, json=body).status_code == 403
    assert c.put(url, json=body, headers={"X-CSRF-Token": "x.y"}).status_code == 403
    good = c.cookies["cv_csrf"]
    c.cookies.delete("cv_csrf")
    assert c.put(url, json=body, headers={"X-CSRF-Token": good}).status_code == 403
    c.cookies.set("cv_csrf", good)
    other = TestClient(app)
    login(other, fake, "csrf2@example.com")
    assert c.put(url, json=body, headers={"X-CSRF-Token": other.cookies["cv_csrf"]}).status_code == 403
    r = c.put(url, json=body)
    assert r.json()["detail"]["code"] == "csrf"
    # fremde Herkunft
    assert c.put(url, json=body, headers={**csrf(c), "Origin": "https://evil.example"}).status_code == 403
    # richtig
    r = c.put(url, json=body, headers={**csrf(c), "Origin": "http://testserver"})
    assert r.status_code == 200, r.text
    # GET braucht kein Token
    assert c.get("/auth/me").status_code == 200
    # Login-Endpunkte sind ebenfalls geschuetzt (Login-CSRF)
    anon = TestClient(app)
    assert anon.post("/auth/code/start", json={"email": "a@b.de"}).status_code == 403
    # CSRF-Token der abgemeldeten Sitzung ist nach Logout nutzlos
    c2 = TestClient(app)
    login(c2, fake, "csrf3@example.com")
    old = c2.cookies["cv_csrf"]
    c2.post("/auth/logout", headers=csrf(c2))
    assert old != c2.cookies["cv_csrf"]


# ------------------------------------------------------------------ Benutzer, Rollen, Admin
def test_bootstrap_admin_by_email_and_subject():
    app, fake, c = make_app()
    me = login(c, fake, ADMIN_EMAIL)
    assert me["role"] == "admin" and me["is_admin"]
    r = c.get("/admin/users")
    assert r.status_code == 200 and r.json()["users"][0]["bootstrap_admin"] is True
    # Subject-basiert (ohne Adresse von Codai, Adresse nicht vertraut)
    sub = fake.subject_for("anderer@example.com")
    app2, fake2, c2 = make_app(fake_codai(email_claim=set()), admins=frozenset({sub}), trust_flow_email=False)
    me = login(c2, fake2, "anderer@example.com")
    assert me["role"] == "admin" and me["email"] is None
    # Wer weder Subject noch Adresse trifft, ist kein Admin
    assert login(TestClient(app2), fake2, "fremd@example.com")["role"] == "fan"
    # Adresse nur getippt (kein EMAIL_CLAIM): gilt, weil der Code an sie ging
    app3, fake3, c3 = make_app(fake_codai(email_claim=set()))
    assert login(c3, fake3, ADMIN_EMAIL)["role"] == "admin"
    # ... ausser trust_flow_email ist aus
    app4, fake4, c4 = make_app(fake_codai(email_claim=set()), trust_flow_email=False)
    assert login(c4, fake4, ADMIN_EMAIL)["role"] == "fan"


def test_admin_assigns_memberships_and_invites():
    app, fake, admin = make_app()
    login(admin, fake, ADMIN_EMAIL)
    user = TestClient(app)
    login(user, fake, "trainer@example.com")
    sub = fake.subject_for("trainer@example.com")
    assert user.get("/auth/me").json()["role"] == "fan"
    # Validierung
    url = f"/admin/users/{sub}/memberships"
    for bad in ([{"role": "trainer"}], [{"role": "trainer", "team_id": "gibts-nicht"}], [{"role": "zauberer"}],
                [{"role": "spieler", "player_id": "nope"}], [{"role": "system"}]):
        assert admin.put(url, json={"memberships": bad}, headers=csrf(admin)).status_code == 422, bad
    assert admin.put("/admin/users/00000000-0000-4000-8000-000000000000/memberships", json={"memberships": []},
                     headers=csrf(admin)).status_code == 404
    # Nicht-Admins duerfen nichts
    assert user.put(url, json={"memberships": [{"role": "admin"}]}, headers=csrf(user)).status_code == 403
    r = admin.put(url, json={"memberships": [{"role": "trainer", "team_id": "team-dessau"},
                                              {"role": "spieler", "player_id": "dsx7"}]}, headers=csrf(admin))
    assert r.status_code == 200, r.text
    # wirkt sofort in der bestehenden Sitzung des Benutzers
    me = user.get("/auth/me").json()
    assert me["role"] == "trainer" and me["team_id"] == "team-dessau"
    assert me["memberships"][0]["label"] == "Trainer · TV Dessau Herren 1"
    assert me["memberships"][1]["team_id"] == "team-dessau"  # Team des Spielers wird abgeleitet
    # Einladung per E-Mail: greift beim ersten Login
    r = admin.post("/admin/invites", json={"email": "Neu@Example.com", "memberships": [{"role": "hallenwart"}]},
                   headers=csrf(admin))
    assert r.status_code == 201
    assert [i["email"] for i in admin.get("/admin/users").json()["invites"]] == ["neu@example.com"]
    nu = TestClient(app)
    me = login(nu, fake, "neu@example.com")
    assert me["role"] == "hallenwart"
    assert admin.get("/admin/users").json()["invites"] == []  # verbraucht
    # Einladung fuer einen bekannten Benutzer wird sofort angewendet
    r = admin.post("/admin/invites", json={"email": "trainer@example.com", "memberships": [{"role": "verband"}]},
                   headers=csrf(admin))
    assert r.json()["applied_to"] == sub
    assert {m["role"] for m in user.get("/auth/me").json()["memberships"]} == {"trainer", "spieler", "verband"}
    # Einladung zuruecknehmen
    admin.post("/admin/invites", json={"email": "weg@example.com", "memberships": [{"role": "verband"}]}, headers=csrf(admin))
    assert admin.delete("/admin/invites/weg@example.com", headers=csrf(admin)).status_code == 200
    assert admin.delete("/admin/invites/weg@example.com", headers=csrf(admin)).status_code == 404


def test_invite_needs_trusted_email():
    # Ohne EMAIL_CLAIM und ohne Vertrauen in die getippte Adresse greifen nur Subject-Zuweisungen
    app, fake, _ = make_app(fake_codai(email_claim=set()), trust_flow_email=False)
    app.state.auth.users.invite("x@example.com", [{"role": "hallenwart"}])
    me = login(TestClient(app), fake, "x@example.com")
    assert me["role"] == "fan" and me["email"] is None
    assert len(app.state.auth.users.invites()) == 1


# ------------------------------------------------------------------ Sitzungsdauer
def test_session_sliding_and_absolute_expiry():
    app, fake, c = make_app()
    login(c, fake, "ttl@example.com")
    sm = app.state.auth.sessions
    t0 = time.time()
    sm.clock = lambda: t0 + 11 * 3600
    assert c.get("/auth/me").json()["authenticated"]          # <12 h: gueltig und gleitend verlaengert
    sm.clock = lambda: t0 + 22 * 3600
    assert c.get("/auth/me").json()["authenticated"]          # 11 h nach der letzten Nutzung
    sm.clock = lambda: t0 + 35 * 3600
    assert not c.get("/auth/me").json()["authenticated"]      # 13 h Pause -> abgelaufen
    assert app.state.store.list_docs("session") == {}
    sm.clock = time.time
    c2 = TestClient(app)
    login(c2, fake, "ttl2@example.com")
    t1 = time.time()
    for k in range(1, 72):                                    # alle 10 h aktiv, aber nach 30 Tagen ist Schluss
        sm.clock = lambda k=k: t1 + k * 10 * 3600
        assert c2.get("/auth/me").json()["authenticated"], k
    sm.clock = lambda: t1 + 30 * 86400 + 3600
    assert not c2.get("/auth/me").json()["authenticated"]


# ------------------------------------------------------------------ Token, Modi
def test_system_bearer_token_for_observations():
    app, fake, c = make_app()
    body = [{"kind": "ClockState", "video_ts": 10 + i, "period": 1, "clock": 600.0 - i, "running": True} for i in range(3)]
    url = "/games/demo-game-1/observations"
    assert c.post(url, json=body).status_code == 401                                  # anonym
    assert c.post(url, json=body, headers={"Authorization": "Bearer falsch"}).status_code == 401
    assert c.post(url, json=body, headers={"Authorization": "Bearer " + SYS_TOKEN[:-1]}).status_code == 401
    r = c.post(url, json=body, headers={"Authorization": "Bearer " + SYS_TOKEN})      # kein CSRF fuer Maschinen
    assert r.status_code == 200 and r.json()["fed"] == 3
    # Token ersetzt nicht die Sitzungs-Rolle: ein Fan-Cookie plus falsches Token bleibt 401
    login(c, fake, "fan@example.com")
    assert c.post(url, json=body, headers={**csrf(c), "Authorization": "Bearer nope"}).status_code == 401
    assert c.post(url, json=body, headers=csrf(c)).status_code == 403
    # /auth/me meldet die Rolle system
    assert TestClient(app).get("/auth/me", headers={"Authorization": "Bearer " + SYS_TOKEN}).json()["role"] == "system"


def test_short_system_tokens_are_ignored():
    cfg = AuthConfig.from_env({"COURTVISION_SYSTEM_TOKENS": "kurz, " + SYS_TOKEN, "COURTVISION_ADMINS": " A@b.de ,SUB"})
    assert cfg.system_tokens == (SYS_TOKEN,) and cfg.admins == {"a@b.de", "sub"} and cfg.mode == "codai"
    assert not cfg.insecure_cookies and cfg.codai_url == "https://auth.codai.app"
    with pytest.raises(ValueError):
        AuthConfig.from_env({"COURTVISION_AUTH_MODE": "offen"})
    assert AuthConfig.from_env({"COURTVISION_AUTH_MODE": "dev", "COURTVISION_INSECURE_COOKIES": "1"}).insecure_cookies


def test_dev_mode_accepts_x_role_and_codai_mode_rejects_it():
    store = EventStore(":memory:")
    load_demo(store, demo_setup())
    dev = TestClient(create_app(store, auth_mode="dev"))
    assert dev.get("/admin/users", headers={"X-Role": "admin"}).status_code == 200
    assert dev.get("/admin/users", headers={"X-Role": "fan"}).status_code == 403
    assert dev.post("/games/demo-game-1/simulate", json={"seed": 1}).status_code == 403  # Dev: 403 wie bisher
    assert dev.get("/auth/config").json()["mode"] == "dev"
    app, fake, c = make_app()
    assert c.get("/admin/users", headers={"X-Role": "admin"}).status_code == 401
    assert c.get("/admin/users", headers={"X-Role": "admin", "X-Team-Id": "team-dessau"}).status_code == 401
    r = c.get("/auth/me", headers={"X-Role": "admin"})
    assert r.json()["role"] == "fan" and not r.json()["authenticated"]
    # auch als eingeloggter Fan bringt X-Role nichts
    login(c, fake, "fan2@example.com")
    assert c.get("/admin/users", headers={"X-Role": "admin"}).status_code == 403


def test_not_configured_without_service_token():
    store = EventStore(":memory:")
    c = TestClient(create_app(store, auth_config=AuthConfig(mode="codai", insecure_cookies=True)))
    assert c.get("/auth/config").json()["configured"] is False
    r = start(c, "a@example.com")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "not_configured"


def test_security_headers_and_csp_pages():
    app, fake, c = make_app()
    r = c.get("/login.html")
    assert r.status_code == 200
    assert "script-src 'self'" in r.headers["content-security-policy"] and "'unsafe-inline'" not in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "<script>" not in r.text and "onclick=" not in r.text and "style=" not in r.text  # CSP-tauglich
    a = c.get("/admin.html")
    assert a.status_code == 200 and "content-security-policy" in a.headers and "<script>" not in a.text


# ------------------------------------------------------------------ WebSocket
def test_websocket_authenticates_with_session_cookie():
    app, fake, c = make_app()
    login(c, fake, ADMIN_EMAIL)
    r = c.post("/games/demo-game-1/simulate", json={"seed": 3, "include_frames": False, "speed": None}, headers=csrf(c))
    assert r.status_code == 200, r.text

    def types(ws):
        out = []
        while True:
            m = ws.receive_json()
            if m["kind"] == "synced":
                return out
            out.append(m["event"]["type"])

    with c.websocket_connect("/ws/games/demo-game-1") as ws:        # Admin per Cookie
        admin_types = types(ws)
    assert "STARTERS" in admin_types
    anon = TestClient(app)
    with anon.websocket_connect("/ws/games/demo-game-1") as ws:     # anonym = oeffentlicher Fan
        fan_types = types(ws)
    assert "STARTERS" not in fan_types and "SHOT" in fan_types
    with anon.websocket_connect("/ws/games/demo-game-1?role=admin") as ws:   # Rollen-Query wird im Codai-Modus ignoriert
        assert "STARTERS" not in types(ws)
    with anon.websocket_connect("/ws/games/demo-game-1", headers={"X-Role": "admin"}) as ws:
        assert "STARTERS" not in types(ws)
    # Bearer-Token (Maschinen) und Cross-Site-Origin
    with anon.websocket_connect("/ws/games/demo-game-1", headers={"Authorization": "Bearer " + SYS_TOKEN}) as ws:
        assert "STARTERS" in types(ws)
    with pytest.raises(Exception):
        with c.websocket_connect("/ws/games/demo-game-1", headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_json()
    with pytest.raises(Exception):
        with anon.websocket_connect("/ws/games/demo-game-1", headers={"Authorization": "Bearer falsch"}) as ws:
            ws.receive_json()


# ------------------------------------------------------------------ Demo-Verdrahtung (serve --demo)
def test_demo_accounts_get_their_roles():
    from courtvision.auth.demo import DEMO_ADMIN, demo_accounts, demo_config, make_demo_codai, seed_demo_invites

    store = EventStore(":memory:")
    setup = demo_setup()
    load_demo(store, setup)
    cfg = demo_config(AuthConfig(mode="codai", insecure_cookies=True))
    seed_demo_invites(store, setup, cfg)
    seed_demo_invites(store, setup, cfg)  # idempotent
    fake = make_demo_codai(fixed_code=CODE, announce=lambda _l: None)
    app = create_app(store, auth_config=cfg, fake_codai=fake)
    assert login(TestClient(app), fake, DEMO_ADMIN)["role"] == "admin"
    me = login(TestClient(app), fake, "trainer.gast@demo.courtvision.local")
    assert me["role"] == "trainer" and me["team_id"] == setup.away.id
    me = login(TestClient(app), fake, "mehrfach@demo.courtvision.local")
    assert [m["role"] for m in me["memberships"]] == ["trainer", "kampfgericht", "spieler"]
    assert login(TestClient(app), fake, "irgendwer@example.com")["role"] == "fan"
    assert len(demo_accounts(setup)) == 7
    # Passwort-Login des Demo-Admins (neues Geraet -> Code)
    c = TestClient(app)
    c.get("/auth/config")
    r = c.post("/auth/password", json={"email": DEMO_ADMIN, "password": "demo-passwort"}, headers=csrf(c))
    assert r.json()["step_up"] is True
    # Fake-Codai ist unter /_fake_codai erreichbar (Konto-Link der Oberflaeche)
    assert c.get("/_fake_codai/account").status_code == 200
    assert c.get("/auth/config").json()["account_url"] == "/_fake_codai/account"
