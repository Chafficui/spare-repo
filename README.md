# CourtVision

Automatisches Basketball-Scouting für den Amateur- und Jugendbereich: Eine fest installierte Weitwinkelkamera pro Halle
liefert Video, daraus entstehen **Live-Spielstand, Play-by-Play, Box-Score, Shot Chart, Spielernoten und Highlights** –
ohne Statistiker. Der Anzeigetafel-Stand dient als Wahrheit für die Punkte, ein Mensch (Kampfgericht) bestätigt unsichere
Ereignisse und tippt Fouls. Alle Oberflächen lesen aus **einem Event-Log** (Event Sourcing); Korrekturen sind neue
Events, nichts wird überschrieben.

Grundlage ist der Produktplan (*CourtVision – Produktplan automatisiertes Basketball-Scouting*, liegt nicht im Repo). Dieses
Repository enthält die **Software-Kette ab der Beobachtung** (Zustandsautomat, Tafel-Fusion, Statistik, Review,
Datenschutz, API, Web-UI) vollständig und getestet. Die **Wahrnehmung am Bild** (trainierte Modelle, echte Hallen)
ist als Schnittstellen und Algorithmen angelegt, aber nicht mit echten Daten belegt – siehe
[Was ist umgesetzt, was fehlt](#was-ist-umgesetzt-was-braucht-reale-assets).

## Architektur

```
 Kamera (PoE, RTSP)                      Anzeigetafel (Serie / 2. Kamera / Hauptkamera)
        |                                         |
   [edge]  Aufzeichnung (GStreamer), Upload       [scoreboard]  7-Segment-OCR, Plausibilität
        |                                         |
   [cv]  Detektion -> Tracking -> Teams -> Nummern -> Feldkoordinaten [geometry]
        |  Observation-Strom (domain.observations)         |  ScoreboardReading
        v                                                  v
   POST /games/{id}/observations                POST /games/{id}/scoreboard
        \                                                 /
         +------------- live.GameSession ----------------+
              scoring.EventEngine  +  scoreboard.ScoreboardFusion  +  live.PeriodTracker
                                   |
                                   v
                      store.EventStore  (append-only, SQLite)
                                   |
        stats (Box-Score, Noten)  qa (Korrekturliste, Freigabe)  highlights  privacy
                                   |
                         api (FastAPI + WebSocket)  ->  web (6 Oberflächen)
```

| Komponente laut Plan | Paket | Inhalt |
|---|---|---|
| Event-Log, Stammdaten, Beobachtungs-Schnittstelle | `courtvision.domain` | `Event`/`EventType`, `Game`/`Team`/`Player`/`Consent`, `Observation`-Dataclasses (CV → Event-Erkennung), `serialize` (JSON ↔ Observation) |
| Persistenz | `courtvision.store` | `EventStore`: append-only Event-Log, Stammdaten und generische JSON-Dokumente (`put_doc`/`get_doc`/`list_docs`) auf SQLite (Schema PostgreSQL-kompatibel gehalten); Listener treiben die WebSocket-Updates |
| Spielfeld und Kalibrierung | `courtvision.geometry` | FIBA-Feld, Dreierlinie, Homographie aus Landmarken-Klicks, Zonen |
| CV-Pipeline | `courtvision.cv` | Detektor-Protocol (RF-DETR-Adapter), Tracker (OC-SORT-Stil, numpy/scipy), Team-Clustering, Trikotnummern-Abstimmung, Ball (Parabel-Fit, Wurf-/Ring-Detektor), Ballbesitz, Formationen, Pose-Protocol, `CVPipeline` (EDGE-/CLOUD-Modus) |
| Event-Erkennung / Live-Scoring | `courtvision.scoring` | regelbasierter Zustandsautomat (Beobachtungen → Wurf, Rebound, Assist, Block, Steal, Turnover ...) und Wechselerkennung über Wechselzonen |
| Anzeigetafel-Abgleich | `courtvision.scoreboard` | 7-Segment-OCR (numpy), Erfassungswege A/B/C, Plausibilitätsfilter mit Entprellung, Fusion (Tafel bestätigt/korrigiert optische Würfe, `clock_at`) |
| Live-Pfad | `courtvision.live` | `GameSession` (Engine + Filter + Fusion + `PeriodTracker` → Store), `PeriodTracker` (GAME_START/PERIOD_START/PERIOD_END/GAME_END aus der Uhr), `reprocess` (Cloud-Lauf ersetzt vorläufige Live-Events, Menschen haben das letzte Wort) |
| Statistik und Noten | `courtvision.stats` | Box-Score als reine Projektion des Event-Logs (Minuten, +/-, Lineups, Teamfouls), PIR/EFF/Game Score, Bayes-Shrinkage, Note 1,0–10,0 |
| Human-in-the-Loop / QA | `courtvision.qa` | Korrekturliste nach Konfidenz und Wirkung, Fouleingabe per Klick, Spielberichtsbogen-Abgleich, Freigabe (`final`) |
| Highlight-Engine | `courtvision.highlights` | Bewertung je Ereignis, Szenen/Clip-Pläne (Live, Recap, Spieler-Reel), virtuelle Kamera (Crop-Fenster), Overlay mit Einwilligungslogik, ffmpeg-Kommandos (werden gebaut, nicht ausgeführt) |
| Recht und Datenschutz | `courtvision.privacy` | `ConsentManager` mit Stufen (Stats intern/öffentlich, Highlights, Livestream, Trainingsdaten), Minderjährigen-Regeln und Audit-Trail, Aufbewahrungsfristen, DSFA-Checkliste |
| Geschäftsmodell | `courtvision.business` | Preise, Rechnung, Kosten pro Spiel, Unit Economics (alle Preise sind **Annahmen** aus dem Plan) |
| Edge-Rechner | `courtvision.edge` | GStreamer-Pipeline als String (Passthrough, HW-Encoder-Regeln für Orin), Segmentverwaltung, resumable persistente Upload-Queue mit LTE-Policy (Transport injizierbar) |
| Evaluation | `courtvision.eval` | Benchmark gegen die Genauigkeitsziele (Anteil exakt, F1 je Ereignistyp, Latenz), Rollout-Gate |
| Simulator | `courtvision.sim` | deterministischer Spielgenerator: Ground-Truth-Events, Beobachtungsstrom, Tafelstrom, optional Rauschen; fiktive Demo-Stammdaten |
| API | `courtvision.api` | FastAPI: REST + WebSocket, Rollen, Sichtbarkeitsregeln je Einwilligung, Import/Export, Demo-Simulation |
| Oberflächen | `courtvision.web` | statische HTML/JS-Seiten ohne Build-Schritt, werden von der API unter `/` ausgeliefert |

## Installation und Betrieb

Python 3.13.

```bash
pip install -r requirements.txt            # numpy, scipy, pydantic, fastapi, uvicorn, websockets, httpx, pytest
python -m pytest -q tests                  # 249 Tests

python -m courtvision.cli demo [--db spiel.sqlite] [--seed 1] [--no-frames] [--noise]   # Demo-Spiel simulieren, Box-Score ausgeben
python -m courtvision.cli benchmark --seeds 1 2 3 --frames                              # Genauigkeit gegen die Plan-Ziele
python -m courtvision.cli serve --demo [--db demo.sqlite] [--port 8000]                 # API + Web-UI mit Demo-Daten und Fake-Codai (Anmeldung, s. u.)
python -m courtvision.cli gst-pipeline --camera rtsp://...                              # GStreamer-Kommando des Edge-Rechners ausgeben
```

`serve --demo` lädt Teams, Kader und das Spiel `demo-game-1` (bereits vorhandene Demo-Daten und Spielstände in einer
`--db`-Datei bleiben unberührt) und gibt die URLs aller Seiten aus. Ohne `--db` liegt alles im Speicher (oder in
`$COURTVISION_DB`). `uvicorn courtvision.api.app:app` startet die App ebenfalls direkt.

**Demo ansehen:** `serve --demo` läuft im Auth-Modus `codai` mit einem **Fake-Codai** (siehe [Anmeldung](#anmeldung-codai-id)):
unter `/login.html` mit `admin@demo.courtvision.local` anmelden (der 6-stellige Code steht in der Server-Konsole; die
Schnellauswahl auf der Anmeldeseite füllt die Demo-Adressen ein). Auf der Übersicht (`/index.html`) erscheint bei geplanten
Spielen der Button **„Demo-Spiel simulieren“** (Echtzeitfaktor 20, ca. 3 Minuten für ein volles Spiel) – Spielseite,
Konsole und Dashboard laufen per WebSocket live mit. Für Skripte ohne Browser: `--auth-mode dev` (oder
`COURTVISION_AUTH_MODE=dev`) schaltet die Platzhalter-Header wieder ein:

```bash
python -m courtvision.cli serve --demo --auth-mode dev --port 8000
curl -X POST -H "X-Role: admin" -H "Content-Type: application/json" \
     -d '{"seed": 1, "include_frames": false, "speed": null}' http://127.0.0.1:8000/games/demo-game-1/simulate
```

`speed: null` simuliert synchron (Antwort enthält den Endstand), eine Zahl > 0 spielt in einem Hintergrund-Thread mit diesem Echtzeitfaktor ab (`GET /games/{id}/simulate` zeigt den Fortschritt). Optional `periods`, `period_length_s`.
Die Wahrheits-Fouls des Simulators werden als Konsolen-Fouls zu ihren Zeitstempeln eingespeist. Simuliert werden können nur
geplante Spiele.

### Spiel-Lebenszyklus und Datenfluss

```
scheduled --POST /start--> live --POST /end--> processing --POST /processed--> review --POST /finalize--> final
```

* **start** (Hallenwart): legt die `GameSession` des Spiels an (5 Starter je Team, Wechselzonen aus der gespeicherten
  Kalibrierung) und schreibt GAME_START/STARTERS/PERIOD_START. Nach einem Neustart des Servers wird die Session lazy neu
  aufgebaut (Aufstellung aus den STARTERS-Events); flüchtiger Zustand wie offene Würfe geht dabei verloren.
* **Ingest**: `POST /games/{id}/scoreboard` (Kampfgericht/Hallenwart/System) und `POST /games/{id}/observations`
  (System/Admin; JSON-Liste von Beobachtungen mit `kind`, Format siehe `courtvision/domain/serialize.py`) speisen dieselbe
  Session. Aus der Tafel entstehen auch Periodengrenzen und Spielende.
* **end**: `session.finish()` leert Engine und Fusion, schließt Periode und Spiel, Status `processing`.
* **processing → review**: `POST /processed` (System) meldet, dass die Cloud-Nachverarbeitung durch ist. Die Korrekturliste
  (`/review`, `/fouls`, `/scoresheet`, `/finalize`) funktioniert in `processing` **und** `review`; in `final` ist alles gesperrt (nur Admin).
* Kalibrierung, Medien-URLs, Einwilligungs-Audit und DSFA-Status liegen als Dokumente im Store und überstehen einen Neustart.

## Rollen und Oberflächen

Die Rolle eines Benutzers vergibt CourtVision selbst (Mitgliedschaften, s. [Anmeldung](#anmeldung-codai-id)); wer nicht angemeldet
ist, sieht die öffentlichen Seiten als `fan`. Die Platzhalter-Header `X-Role`/`X-Team-Id`/`X-Player-Id` gibt es nur noch
im Entwicklungsmodus (`COURTVISION_AUTH_MODE=dev`, so laufen die Tests).

| Rolle | Oberfläche | Aufgaben |
|---|---|---|
| `hallenwart` | `setup.html` (Tablet) | Teams und Kader (CSV), Kamera kalibrieren, Aufstellung, Spielstart/-ende |
| `kampfgericht` | `console.html` (Tablet) | unsichere Ereignisse bestätigen, Fouls tippen, Spielbericht, Freigabe |
| `trainer` | `dashboard.html` | Box-Score, Shot Chart, Lineups, Play-by-Play, Noten, Highlights des eigenen Teams |
| `spieler` / Eltern | `player.html` | eigene Stats, Notenverlauf, Highlights, Einwilligungen |
| `fan` (Standard) | `game.html` | Live-Score, Play-by-Play, Livestream, datenschutzkonformer Box-Score |
| `verband` | `verband.html` | Bestenliste, CSV-Export, DSFA-Checkliste, Abrechnung |
| `system` | – | Edge-/Cloud-Pipeline: Beobachtungen, Tafelwerte, optische Events |
| `admin` | `index.html`, `admin.html` | darf alles, inkl. Demo-Simulation und Benutzerverwaltung |

Sichtbarkeit richtet sich nach der Einwilligung je Spieler (`courtvision/api/visibility.py`): Widerspruch → Overlay/Text
ohne Namen, keine Highlights, Statistik nur als Team-Wert; Minderjährige brauchen Eltern- (ab 14 auch eigene) Einwilligung.

## Anmeldung (Codai ID)

CourtVision hat eine **eigene Anmeldeseite** (`login.html`), die echte Anmeldung übernimmt aber **Codai ID** (`CodaiApp/auth`, „Headless
sign-in“). Codai beantwortet nur *wer* sich anmeldet (`subject`, `username`, optional `email`); *was* die Person darf, entscheidet CourtVision.

```
 Browser  --/auth/* (Cookie cv_session, X-CSRF-Token)-->  CourtVision (BFF)  --Bearer-Service-Token-->  Codai ID
 (login.html)                                             Sitzungen, Rollen,   POST /service/login/code/start|verify
                                                          Rate-Limits          POST /service/login/password
```

* **BFF, nur Server-zu-Server:** Der Browser spricht nie mit Codai und sieht weder das Service-Token noch das Codai-`flow`.
  `courtvision/auth/codai.py` (`CodaiClient`) reicht IP (`X-Forwarded-For` nur mit `COURTVISION_TRUST_PROXY=1`), User-Agent und eine
  Geräte-ID (Cookie `cv_device`) des Nutzers an Codai weiter und bildet alle Codai-Fehler (`bad_email`, `rate_limited`, `mail_failed`,
  `bad_code` mit `attemptsLeft`, `expired`, `bad_credentials`, `busy`/`unavailable`, `forbidden`/`unauthorized`) auf typisierte Ausnahmen und
  deutsche Meldungen ab.
* **Ablauf Code-Login:** E-Mail → `POST /auth/code/start` → Code per Mail → `POST /auth/code/verify` → Sitzung. Der Codai-Flow liegt
  serverseitig (`docs`-Art `loginflow`) hinter dem signierten, kurzlebigen Cookie `cv_login` (10 min, HttpOnly): ein anderer
  Browser kann den Code nicht einlösen. **Passwort-Login** (`POST /auth/password`): bei unbekanntem Gerät verlangt Codai zusätzlich
  einen Mail-Code („Step-up“); die Oberfläche zeigt dann den Code-Schritt. Passkeys werden nicht unterstützt (siehe unten).
* **Sitzungen:** serverseitig im Store (`docs`-Art `session`, abgelegt wird nur `sha256(id)`), 256-Bit-ID im Cookie `cv_session`
  (HttpOnly, `SameSite=Lax`, `Secure`, Pfad `/`), 12 h gleitend / 30 d absolut, Logout löscht sie.
* **CSRF:** Double-Submit-Token (Cookie `cv_csrf` + Header `X-CSRF-Token`), HMAC-signiert und an die Sitzung gebunden; geprüft für alle
  zustandsändernden Anfragen mit Cookies und für `/auth/*`. Zusätzlich wird `Origin` gegen `Host` geprüft. Bearer-Clients sind
  ausgenommen. Die Login-Endpunkte haben pro IP/E-Mail eigene Limits (im Speicher) zusätzlich zu denen von Codai.
* **Maschinen-Clients** (Edge/CV-Pipeline, Rolle `system`): `Authorization: Bearer <Token>` mit einem Token aus
  `COURTVISION_SYSTEM_TOKENS` (Vergleich in konstanter Zeit, mind. 16 Zeichen), z. B. für `POST /games/{id}/observations`. Kein CSRF nötig.
* **WebSocket:** authentifiziert sich im Codai-Modus über das Sitzungs-Cookie (bzw. Bearer-Header); Rollen-Query-Parameter werden ignoriert,
  fremde `Origin`s abgelehnt.
* **Nicht angemeldet:** öffentlicher `fan` für öffentliche Seiten/Endpunkte, `401` für geschützte (angemeldet ohne Berechtigung: `403`);
  `app.js` leitet auf `login.html?next=…` um.

### Benutzer und Rollen

Nutzer-Datensätze (`docs`-Art `user`): `subject`, `username`, `email?`, `memberships`, `created_at`, `last_login`. Eine **Mitgliedschaft** ist
`{role, team_id?, player_id?}` (`trainer` braucht ein Team, `spieler` einen Spieler; `hallenwart`, `kampfgericht`, `verband`, `admin` keins).
Wer keine hat, ist `fan`. Mehrere Mitgliedschaften sind möglich; die **aktive Rolle** wählt man im Benutzermenü (Sitzung) oder je Anfrage mit
`X-Active-Role` (+ `X-Active-Team`/`X-Active-Player`) – der Server prüft sie gegen die Mitgliedschaften, der Client kann keine fremde
Rolle, kein fremdes Team und keinen fremden Spieler behaupten.

* **Admins** werden über `COURTVISION_ADMINS` (Codai-Subjects und/oder E-Mails, kommagetrennt) bestimmt, weitere vergibt ein Admin.
* **Verwaltung:** `admin.html` bzw. `GET /admin/users`, `PUT /admin/users/{subject}/memberships`, `POST /admin/invites`
  (`{email, memberships}`), `DELETE /admin/invites/{email}`. Eine **Einladung per E-Mail** merkt Rollen für eine Adresse vor und wendet sie
  beim ersten Login an (bei bereits bekannten Benutzern sofort). Das setzt eine *bekannte* Adresse voraus: entweder liefert Codai sie
  (`EMAIL_CLAIM_SERVICES`) oder – Standard – die beim Login getippte Adresse gilt als bestätigt, weil der Code an sie ging bzw. das Passwort zu ihr
  gehört. Mit `COURTVISION_TRUST_TYPED_EMAIL=0` greifen ohne Codai-Adresse nur Zuweisungen über das `subject`.

### Umgebungsvariablen

| Variable | Bedeutung |
|---|---|
| `COURTVISION_AUTH_MODE` | `codai` (Standard) oder `dev` (Platzhalter-Header `X-Role` – **nur Entwicklung/Tests**) |
| `CODAI_AUTH_URL` | Basis-URL von Codai ID (Standard `https://auth.codai.app`); auch Ziel der Links „Passwort vergessen?“ (`/password?reset=1`) und „Konto verwalten“ (`/account`) |
| `CODAI_SERVICE_TOKEN` | Bearer-Token dieses Dienstes bei Codai (mind. 32 Zeichen; **geheim**, nur auf dem Server). Fehlt es, antworten die Login-Endpunkte mit 503 |
| `CODAI_CLIENT_ID` | optional: Hydra-Client-ID, die Codai als `client` für den Namen in der Code-Mail bekommt |
| `COURTVISION_ADMINS` | kommagetrennte Codai-Subjects und/oder E-Mails mit Admin-Rolle |
| `COURTVISION_SYSTEM_TOKENS` | kommagetrennte Bearer-Tokens für Maschinen-Clients (Rolle `system`) |
| `COURTVISION_SECRET` | Signaturschlüssel für `cv_login`/`cv_csrf`; ohne Angabe wird einer im Store erzeugt (bei mehreren Instanzen mit getrennten Stores setzen) |
| `COURTVISION_TRUST_PROXY` | `1`: `X-Forwarded-For` und `X-Forwarded-Host` auswerten; `COURTVISION_PROXY_HOPS` (Standard 1) = Anzahl vertrauter Proxys (von rechts gezählt) |
| `COURTVISION_INSECURE_COOKIES` | `1`: Cookies ohne `Secure` (nur für lokales `http://`) |
| `COURTVISION_TRUST_TYPED_EMAIL` | `0`: getippte E-Mail nicht als bestätigt behandeln (Standard `1`) |
| `COURTVISION_DEMO_CODE` | nur `serve --demo`: fester Code statt zufälliger |

### Was der Codai-Betreiber einrichten muss

Im Codai-Dienst (`CodaiApp/auth`):

1. Einen Dienstnamen wählen, z. B. `courtvision`, und ihn in `SERVICE_TOKENS` eintragen (`courtvision=<sha256-Hex des Tokens>`; das Klartext-Token
   – mind. 32 Zeichen – kommt als `CODAI_SERVICE_TOKEN` zu CourtVision).
2. Denselben Namen in `HEADLESS_LOGIN_SERVICES` aufnehmen (sonst `403 forbidden`); produktiv braucht Codai dafür `RESEND_API_KEY`.
3. Optional in `EMAIL_CLAIM_SERVICES`, damit Codai die bestätigte Adresse mitliefert (Einladungen und `COURTVISION_ADMINS` per E-Mail arbeiten dann auf Codais
   Adresse statt der getippten).
4. Optional einen Hydra-Client registrieren (`scripts/clients.ts`) und dessen ID als `CODAI_CLIENT_ID` setzen, damit die Code-Mail „CourtVision“ nennt
   (sonst steht der Dienstname dort).

### Entwicklung und Demo ohne Codai

* `courtvision/auth/fake_codai.py` bildet die Headless-Endpunkte samt Fehlersemantik nach (Limits, 5 Fehlversuche, 10 min, Step-up, E-Mail-Claim). Die
  Codes stehen im Log; **nie im offenen Netz betreiben**. Einbindung im Prozess (`create_app(store, fake_codai=FakeCodai(...))`, gemountet unter
  `/_fake_codai`) oder eigenständig: `python -m courtvision.auth.fake_codai --port 4460` und `CODAI_AUTH_URL=http://127.0.0.1:4460`
  mit `CODAI_SERVICE_TOKEN` aus der Ausgabe.
* `python -m courtvision.cli serve --demo` nutzt standardmäßig den Codai-Modus mit eingehängtem Fake-Codai und druckt eine Warnung sowie die Codes in die
  Konsole. Demo-Adressen (alle `@demo.courtvision.local`): `admin@` (Admin, auch Passwort `demo-passwort`), `hallenwart@`, `kampfgericht@`,
  `trainer.heim@`, `trainer.gast@`, `spieler@`, `verband@`, `mehrfach@` (Trainer + Kampfgericht + Spieler, zum Rollenwechsel); jede andere Adresse ist Fan.
  Für lokales `http://` `COURTVISION_INSECURE_COOKIES=1` setzen. `--codai-url URL` verwendet ein echtes Codai (Token aus `CODAI_SERVICE_TOKEN`),
  `--auth-mode dev` die Platzhalter-Header.
* Screenshots der Oberflächen (angemeldet über die echte Oberfläche) liegen in `docs/screenshots/`.

### Grenzen

* **Passkeys** (Codai: `/service/passkey/*`) werden nicht angebunden: WebAuthn ist an die Relying-Party-Domain `codai.app` gebunden; für CourtVision
  auf einer anderen Domain wären Related Origin Requests und `PASSKEY_SERVICE_ORIGINS` auf Codai-Seite nötig.
* Die älteren Seiten (`dashboard.html` usw.) enthalten noch Inline-Skripte; eine strenge CSP (kein `unsafe-inline`) gilt daher nur für `login.html` und `admin.html`.
* Die Rate-Limits der Anmeldung liegen pro Prozess im Speicher; mehrere Instanzen teilen Sitzungen und Login-Flows über den Store, Limits nicht.

## Benchmark

Ein simuliertes Spiel läuft durch die komplette Live-Kette (`GameSession`) und wird gegen die Wahrheit des Simulators
gemessen (`python -m courtvision.cli benchmark --seeds 1 2 3 --frames`, rauschfrei, Fouls als Konsoleneingabe):

```
Seeds [1, 2, 3]  frames=True  noise=None  console_fouls=True  periods=4x600s
Kategorie            Anteil     MAE     n    Ziel  Ergebnis
ast                  100.0%    0.00    72     75%  PASS
blk                  100.0%    0.00    72     70%  PASS
drb                  100.0%    0.00    72       -  info
fg3a                 100.0%    0.00    72     90%  PASS
fga                  100.0%    0.00    72     90%  PASS
fta                  100.0%    0.00    72       -  info
ftm                  100.0%    0.00    72       -  info
orb                  100.0%    0.00    72       -  info
pts                  100.0%    0.00    72     95%  PASS
reb                  100.0%    0.00    72     80%  PASS
stl                  100.0%    0.00    72     70%  PASS
tov                   79.2%    0.21    72     70%  PASS
tov_obs              100.0%    0.00    72       -  info
minutes (+-30s)      100.0%    0.00    72     95%  PASS
minutes_10s (+-10s)  100.0%    0.00    72       -  info
score                100.0%    0.00     3    100%  PASS
Event-F1 (nur Stat-Typen):
  ASSIST         P=1.00 R=1.00 F1=1.00 (tp 56, fp 0, fn 0)
  BLOCK          P=1.00 R=1.00 F1=1.00 (tp 21, fp 0, fn 0)
  FOUL           P=1.00 R=1.00 F1=1.00 (tp 102, fp 0, fn 0)
  FREE_THROW     P=1.00 R=1.00 F1=1.00 (tp 136, fp 0, fn 0)
  REBOUND        P=0.94 R=0.94 F1=0.94 (tp 250, fp 17, fn 17)
  SHOT           P=1.00 R=1.00 F1=1.00 (tp 441, fp 0, fn 0)
  STEAL          P=1.00 R=1.00 F1=1.00 (tp 32, fp 0, fn 0)
  SUBSTITUTION   P=1.00 R=1.00 F1=1.00 (tp 185, fp 0, fn 0)
  TURNOVER       P=1.00 R=0.59 F1=0.75 (tp 44, fp 0, fn 30)
Latenz board_to_log: n=294 mean=1.00s p95=1.00s max=1.00s  Ziel < 3s: PASS
Latenz shot_to_log: n=294 mean=4.24s p95=5.25s max=5.70s
ALLE ZIELE ERFUELLT
```

**Einordnung:** Das ist der Test der *Software-Kette* auf synthetischen, sauberen Beobachtungen. Er sagt nichts über die
Genauigkeit am echten Videobild; die 100 % sind auf diesem Eingang zu erwarten. Mit `--noise` (fehlende Trikotnummern,
verpasste Ring-Treffer) wird es ehrlicher, ersetzt aber keinen Test mit Hallenmaterial. Auffällig bleiben die
Ballverluste (`tov`, Recall 59 %): nicht optisch beobachtbare Arten (Schrittfehler, 24 s) müssen von Konsole/Spielbericht kommen.

## Was ist umgesetzt, was braucht reale Assets

**Umgesetzt und getestet (synthetische Daten, Fakes für Modelle/Hardware):** Event-Log und alle Projektionen,
Zustandsautomat, Wechselerkennung, Tafel-Plausibilität und -Fusion (inkl. 7-Segment-OCR auf gerenderten Ziffern),
Periodenerkennung, Cloud-Nachlauf-Politik (`reprocess`), Review/Freigabe, Noten, Highlight-Planung, Einwilligungen,
DSFA-Checkliste, API mit Rollen und WebSocket, Web-UI, Persistenz in SQLite, Benchmark, Simulator.

**Benötigt echte Assets oder externe Zusagen (nicht in diesem Repo):**

* **Trainierte Modelle:** Spieler-/Ball-/Ring-Detektion (der RF-DETR-Adapter ist nur ein Adapter; `torch`/`rfdetr` sind
  optional und ungetestet), Trikotnummern-Klassifikator (nur Protocol + Abstimmungslogik), Pose-Modell (nur Protocol),
  Ball-Heatmap-Modell. Die Algorithmen darum (Tracking, Parabel-Fit, Besitz, Formationen) laufen auf Punktfolgen, sind aber
  ohne reale Detektionen nicht auf Robustheit gegen Verdeckung, Licht und Kamerawinkel geprüft.
* **Hallenmaterial:** Kalibrierung, Wechselzonen, Schwellen und Genauigkeitsziele müssen an echten Aufnahmen verifiziert
  werden; eigener Trainingsdatensatz und Annotation laut Plan.
* **Echte Anzeigetafel-Protokolle:** Weg A (serielle Datenschnittstelle) ist gegen keine reale Tafel (Hersteller, Protokoll)
  geprüft; Weg B/C (OCR vom Bild) nur auf synthetischen 7-Segment-Ziffern.
* **Edge-Hardware:** Die GStreamer-Pipeline wird nur als String gebaut und ohne GStreamer getestet; Jetson/Orin-Betrieb,
  Latenzbudget und Uploader gegen echtes Netz/Server sind ungemessen. ffmpeg-Kommandos der Highlights werden gebaut, nicht
  ausgeführt; Livestream-Ingest und Cloud-Nachlauf mit echtem GPU-Job sind nicht angebunden (`/processed` ist ein manuelles Signal).
* **Recht:** DSFA, Verträge mit Vereinen (Art. 28/26 DSGVO), Hinweisschilder und Löschkonzept sind Checklistenpunkte im
  Produkt, aber **nicht erledigt**; die Software ersetzt keine Rechtsprüfung. Die Minderjährigen-Regeln sind eine Umsetzung
  der Planannahmen, kein Rechtsgutachten.
* **Datenzugang:** Anbindung an DBB/NBN23 (Spielpläne, Spielberichte, Ligadaten) ist nicht vorhanden; Kader und Spiele
  kommen per CSV/API.
* **Betrieb:** Echte Authentifizierung (siehe oben), PostgreSQL statt SQLite, EU-Hosting, Mandantentrennung und Preise
  (Annahmen des Plans) sind offen.
