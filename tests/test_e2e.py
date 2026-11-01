"""Ende-zu-Ende: Simulator -> Live-Session -> Box-Score gegen Wahrheit, Rollout-Gate."""
import copy

import pytest

from courtvision.eval import evaluate, format_report, rollout_gate, rollout_regressions
from courtvision.sim import SimNoise


@pytest.fixture(scope="module")
def clean():
    return evaluate([1], include_frames=True, periods=2, period_length_s=150.0)


def test_clean_short_game_meets_core_targets(clean):
    c = clean["categories"]
    assert c["pts"]["share"] == 1.0 and c["fga"]["share"] == 1.0 and c["score"]["share"] == 1.0
    assert c["reb"]["share"] >= 0.8
    assert c["minutes"]["share"] >= 0.7  # TODO(substitutions): Wechselerkennung in Arbeit
    assert clean["latency"]["board_to_log"]["p95"] < 3.0
    assert "FOUL" in clean["f1"] and clean["f1"]["FOUL"]["f1"] == 1.0  # console_fouls=True
    txt = format_report(clean)
    assert "PASS" in txt and "tov_obs" in txt


def test_console_fouls_option():
    r = evaluate([1], include_frames=False, periods=1, period_length_s=240.0, console_fouls=False)
    assert r["f1"].get("FOUL", {"recall": 1.0})["recall"] == 0.0 or "FOUL" not in r["f1"]


def test_observable_turnovers_exceed_all_turnovers():
    r = evaluate([2], include_frames=False, periods=2, period_length_s=300.0)
    assert r["categories"]["tov_obs"]["share"] >= r["categories"]["tov"]["share"]


def test_rollout_gate(clean):
    assert rollout_gate(clean, clean)
    better = copy.deepcopy(clean)
    better["categories"]["ast"]["share"] = min(1.0, better["categories"]["ast"]["share"] + 0.1)
    assert rollout_gate(clean, better)
    worse = copy.deepcopy(clean)
    worse["categories"]["pts"]["share"] -= 0.01
    assert not rollout_gate(clean, worse)
    assert any(s.startswith("pts") for s in rollout_regressions(clean, worse))
    # gleicher Anteil, aber groesserer Fehler -> schlechter; fehlende Kategorie -> schlechter
    mae = copy.deepcopy(clean)
    mae["categories"]["fga"]["mae"] += 0.5
    assert not rollout_gate(clean, mae)
    missing = copy.deepcopy(clean)
    del missing["categories"]["blk"]
    assert not rollout_gate(clean, missing)
    # auch rohe Kategorie-Dicts sind zulaessig; Verbesserung in einer Kategorie darf Rueckgang nicht aufwiegen
    mixed = copy.deepcopy(clean)
    mixed["categories"]["ast"]["share"] = 1.0
    mixed["categories"]["stl"]["share"] -= 0.2
    assert not rollout_gate(clean["categories"], mixed["categories"])


def test_noise_does_not_improve_and_gate_catches_regression():
    noise = SimNoise(drop_player_id_prob=0.4, miss_rim_make_prob=0.15)
    kw = dict(include_frames=False, periods=2, period_length_s=300.0)
    base = evaluate([1, 2], **kw)
    noisy = evaluate([1, 2], noise=noise, **kw)
    assert rollout_gate(base, base)
    assert not rollout_gate(base, noisy) or noisy["categories"]["reb"]["share"] >= base["categories"]["reb"]["share"]
