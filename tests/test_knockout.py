import itertools
import json

import numpy as np

from wc26 import simulate, model
from wc26.config import DATA_RAW


def test_knockout_never_none_and_dominant_team_wins():
    rng = np.random.default_rng(0)
    params = model.ModelParams()
    wins = sum(simulate.decide_knockout("A", 2200, 0, "B", 1300, 0, params, rng) == "A"
               for _ in range(500))
    assert wins > 480  # a huge favourite wins almost always
    for _ in range(200):
        w = simulate.decide_knockout("A", 1500, 0, "B", 1500, 0, params, rng)
        assert w in ("A", "B")  # never a draw / None


def test_knockout_equal_teams_near_coinflip():
    rng = np.random.default_rng(1)
    params = model.ModelParams()
    a = sum(simulate.decide_knockout("A", 1500, 0, "B", 1500, 0, params, rng) == "A"
            for _ in range(4000))
    assert 0.42 < a / 4000 < 0.58


def test_annexc_routing_is_complete_bijection():
    routing = json.loads((DATA_RAW / "third_place_routing.json").read_text())["combinations"]
    assert len(routing) == 495  # C(12,8)
    expected = {"".join(c) for c in itertools.combinations("ABCDEFGHIJKL", 8)}
    assert set(routing.keys()) == expected
    for combo, mapping in routing.items():
        assert len(mapping) == 8
        # the 8 host slots map to exactly the 8 groups in the combo (a bijection)
        assert sorted(mapping.values()) == sorted(combo)
