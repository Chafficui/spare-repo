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
python -m pytest -q tests                  # 221 Tests

python -m courtvision.cli demo [--db spiel.sqlite] [--seed 1] [--no-frames] [--noise]   # Demo-Spiel simulieren, Box-Score ausgeben
python -m courtvision.cli benchmark --seeds 1 2 3 --frames                              # Genauigkeit gegen die Plan-Ziele
python -m courtvision.cli serve --demo [--db demo.sqlite] [--port 8000]                 # API + Web-UI mit Demo-Daten
python -m courtvision.cli gst-pipeline --camera rtsp://...                              # GStreamer-Kommando des Edge-Rechners ausgeben
```

`serve --demo` lädt Teams, Kader und das Spiel `demo-game-1` (bereits vorhandene Demo-Daten und Spielstände in einer
`--db`-Datei bleiben unberührt) und gibt die URLs aller Seiten aus. Ohne `--db` liegt alles im Speicher (oder in
`$COURTVISION_DB`). `uvicorn courtvision.api.app:app` startet die App ebenfalls direkt.

**Demo ansehen:** In der Kopfzeile die Rolle *admin* wählen; auf der Übersicht (`/index.html`) erscheint bei geplanten Spielen
der Button **„Demo-Spiel simulieren“** (Echtzeitfaktor 20, ca. 3 Minuten für ein volles Spiel) – Spielseite, Konsole
und Dashboard laufen per WebSocket live mit. Alternativ per curl:

```bash
curl -X POST -H "X-Role: admin" -H "Content-Type: application/json" \
     -d '{"seed": 1, "include_frames": false, "speed": null}' http://127.0.0.1:8000/games/demo-game-1/simulate
```

`speed: null` simuliert synchron (Antwort enthält den Endstand), eine Zahl > 0 spielt in einem Hintergrund-Thread mit diesem
Echtzeitfaktor ab (`GET /games/{id}/simulate` zeigt den Fortschritt). Optional `periods`, `period_length_s`.
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

Die Rolle kommt aus dem Header `X-Role` (optional `X-Team-Id`, `X-Player-Id`; bei WebSockets als Query-Parameter).
**Das ist ein Platzhalter ohne Authentifizierung** – jeder Client kann sich jede Rolle geben. Vor einem echten Betrieb muss
`courtvision/api/auth.py` durch Login (z. B. OIDC/JWT) ersetzt werden.

| Rolle | Oberfläche | Aufgaben |
|---|---|---|
| `hallenwart` | `setup.html` (Tablet) | Teams und Kader (CSV), Kamera kalibrieren, Aufstellung, Spielstart/-ende |
| `kampfgericht` | `console.html` (Tablet) | unsichere Ereignisse bestätigen, Fouls tippen, Spielbericht, Freigabe |
| `trainer` | `dashboard.html` | Box-Score, Shot Chart, Lineups, Play-by-Play, Noten, Highlights des eigenen Teams |
| `spieler` / Eltern | `player.html` | eigene Stats, Notenverlauf, Highlights, Einwilligungen |
| `fan` (Standard) | `game.html` | Live-Score, Play-by-Play, Livestream, datenschutzkonformer Box-Score |
| `verband` | `verband.html` | Bestenliste, CSV-Export, DSFA-Checkliste, Abrechnung |
| `system` | – | Edge-/Cloud-Pipeline: Beobachtungen, Tafelwerte, optische Events |
| `admin` | `index.html` | darf alles, inkl. Demo-Simulation |

Sichtbarkeit richtet sich nach der Einwilligung je Spieler (`courtvision/api/visibility.py`): Widerspruch → Overlay/Text
ohne Namen, keine Highlights, Statistik nur als Team-Wert; Minderjährige brauchen Eltern- (ab 14 auch eigene) Einwilligung.

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
