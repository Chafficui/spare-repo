"""Kommandozeile: ``python -m courtvision.cli {demo,benchmark,serve,gst-pipeline}``."""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def _cmd_demo(a: argparse.Namespace) -> int:
    from .domain.events import EventType
    from .live.session import GameSession
    from .sim import SimNoise, demo_setup, load_demo, simulate_game
    from .stats.boxscore import compute_boxscore
    from .stats.rating import grade_boxscore
    from .store.eventstore import EventStore

    store = EventStore(a.db)
    setup = load_demo(store, demo_setup())
    game = setup.game.model_copy(update={"periods": a.periods, "period_length_s": a.period_length})
    store.put_game(game)
    noise = SimNoise(drop_player_id_prob=0.2, miss_rim_make_prob=0.05) if a.noise else None
    sim = simulate_game(a.seed, game, setup.rosters, periods=a.periods, period_length_s=a.period_length,
                        noise=noise, include_frames=not a.no_frames)
    sess = GameSession(store, game, setup.rosters, calibration=sim.zones)
    for side in ("home", "away"):
        st = next(e for e in sim.truth if e.type == EventType.STARTERS and e.team == side)
        sess.set_starters(side, st.payload["players"])
    sess.feed_many(sim.observations, sim.scoreboard)
    sess.finish()
    box = compute_boxscore(store.effective(game.id), game, setup.rosters)
    print(f"{setup.home.name} {box.score['home']} : {box.score['away']} {setup.away.name}"
          f"  (Wahrheit {sim.final_score['home']}:{sim.final_score['away']}, db={a.db})")
    grades = grade_boxscore(box)
    print(f"{'Nr':>3} {'Spieler':<18}{'Min':>6}{'PTS':>4}{'FG':>7}{'3P':>6}{'FT':>6}{'REB':>4}{'AST':>4}"
          f"{'STL':>4}{'BLK':>4}{'TOV':>4}{'PF':>3}{'Note':>6}")
    for side in ("home", "away"):
        for pl in sorted((p for p in box.players.values() if p.team == side and p.seconds_played > 0),
                         key=lambda p: -p.pts):
            print(f"{pl.number:>3} {pl.name:<18}{pl.seconds_played / 60:>6.1f}{pl.pts:>4}"
                  f"{pl.fgm:>4}/{pl.fga:<2}{pl.fg3m:>3}/{pl.fg3a:<2}{pl.ftm:>3}/{pl.fta:<2}{pl.reb:>4}"
                  f"{pl.ast:>4}{pl.stl:>4}{pl.blk:>4}{pl.tov:>4}{pl.pf:>3}"
                  f"{grades[pl.player_id].value:>6.1f}{'*' if grades[pl.player_id].provisional else ''}")
        print()
    print("* = vorlaeufig")
    return 0


def _cmd_benchmark(a: argparse.Namespace) -> int:
    from .eval.benchmark import evaluate, format_report
    from .sim import SimNoise

    noise = SimNoise(drop_player_id_prob=0.2, miss_rim_make_prob=0.05) if a.noise else None
    res = evaluate(a.seeds, noise=noise, include_frames=a.frames, console_fouls=not a.no_console_fouls,
                   periods=a.periods, period_length_s=a.period_length)
    print(format_report(res))
    return 0 if res["all_passed"] else 1


def _cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn  # lazy: nur fuer den Server noetig

    uvicorn.run("courtvision.api.app:app", host=a.host, port=a.port, reload=a.reload)
    return 0


def _cmd_gst(a: argparse.Namespace) -> int:
    from .edge.recorder import RecorderConfig, gst_launch_command

    cfg = RecorderConfig(camera_url=a.camera, codec=a.codec, segment_dir=a.segment_dir,
                         segment_s=a.segment_s, container=a.container, rtmp_url=a.rtmp,
                         hw_encoder=a.hw_encoder, device=a.device, overlay=a.overlay,
                         enhanced_rtmp=a.enhanced_rtmp)
    try:
        print(gst_launch_command(cfg))
    except ValueError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="courtvision", description="CourtVision Werkzeuge")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Demo-Spiel simulieren und Box-Score anzeigen")
    d.add_argument("--db", default=":memory:", help="SQLite-Datei (Default: Speicher)")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--periods", type=int, default=4)
    d.add_argument("--period-length", type=float, default=600.0)
    d.add_argument("--no-frames", action="store_true", help="ohne FrameState (schneller, keine Wechsel)")
    d.add_argument("--noise", action="store_true")
    d.set_defaults(fn=_cmd_demo)

    b = sub.add_parser("benchmark", help="Genauigkeit gegen die Plan-Ziele messen")
    b.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    b.add_argument("--frames", action="store_true", help="FrameState simulieren (Wechselerkennung)")
    b.add_argument("--noise", action="store_true")
    b.add_argument("--no-console-fouls", action="store_true")
    b.add_argument("--periods", type=int, default=4)
    b.add_argument("--period-length", type=float, default=600.0)
    b.set_defaults(fn=_cmd_benchmark)

    s = sub.add_parser("serve", help="API-Server starten (uvicorn)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=_cmd_serve)

    g = sub.add_parser("gst-pipeline", help="GStreamer-Pipeline des Edge-Rechners ausgeben")
    g.add_argument("--camera", required=True, help="rtsp://...")
    g.add_argument("--codec", choices=["h264", "h265"], default="h264")
    g.add_argument("--segment-dir", default="/data/courtvision/raw")
    g.add_argument("--segment-s", type=int, default=300)
    g.add_argument("--container", choices=["mkv", "mp4"], default="mkv")
    g.add_argument("--rtmp", default=None, help="rtmp://ingest/...")
    g.add_argument("--hw-encoder", default=None, help="z. B. nvv4l2h264enc (Orin NX/AGX)")
    g.add_argument("--device", default="generic", choices=["generic", "orin_nano", "orin_nx", "orin_agx"])
    g.add_argument("--overlay", action="store_true")
    g.add_argument("--enhanced-rtmp", action="store_true")
    g.set_defaults(fn=_cmd_gst)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
