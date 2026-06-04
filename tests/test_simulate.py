"""Integration + invariant tests for the Monte Carlo simulator.

These use the real ingested data (data/raw) built into a temp DB. The per-run
sum invariants are exact regardless of run count, so they are strong checks.
"""

from wc26 import db, ingest, forecast, simulate

# Exactly this many teams reach each stage in every single run.
EXPECTED_SUMS = {
    "win_group": 12, "top2": 24, "advance": 32,
    "r16": 16, "qf": 8, "sf": 4, "final": 2, "champion": 1,
}


def _build(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    return forecast.load_tournament(conn)


def test_stage_population_invariants(tmp_path):
    t = _build(tmp_path)
    probs = simulate.simulate(t, n_runs=300, seed=7)["probs"]
    for stage, expected in EXPECTED_SUMS.items():
        total = sum(p[stage] for p in probs.values())
        assert abs(total - expected) < 1e-9, (stage, total, expected)


def test_probabilities_bounded_and_monotone(tmp_path):
    t = _build(tmp_path)
    probs = simulate.simulate(t, n_runs=300, seed=7)["probs"]
    for p in probs.values():
        for v in p.values():
            assert 0.0 <= v <= 1.0
        # deeper rounds are subsets of earlier ones
        assert p["champion"] <= p["final"] + 1e-9
        assert p["final"] <= p["sf"] + 1e-9
        assert p["sf"] <= p["qf"] + 1e-9
        assert p["qf"] <= p["r16"] + 1e-9
        assert p["r16"] <= p["advance"] + 1e-9
        assert p["win_group"] <= p["top2"] + 1e-9
        assert p["top2"] <= p["advance"] + 1e-9


def test_determinism_same_seed(tmp_path):
    t = _build(tmp_path)
    a = simulate.simulate(t, n_runs=200, seed=42)["probs"]
    b = simulate.simulate(t, n_runs=200, seed=42)["probs"]
    assert a == b


def test_all_48_teams_present(tmp_path):
    t = _build(tmp_path)
    probs = simulate.simulate(t, n_runs=100, seed=1)["probs"]
    assert len(probs) == 48
