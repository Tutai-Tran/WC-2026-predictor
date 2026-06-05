from wc26 import accuracy


def _m(d):
    return {k: v for d2 in [d] for k, v in d2.items()}


def test_metrics_counts_correct_and_total():
    played = [
        # home favourite, predicted 2-0, actual 2-0 -> everything correct
        {"p_home": 0.6, "p_draw": 0.25, "p_away": 0.15, "lambda_home": 1.8,
         "lambda_away": 0.9, "top_scoreline": "2-0", "result": "2-0"},
        # home favourite, actual 0-1 -> outcome wrong, exact wrong, favourite wrong
        {"p_home": 0.6, "p_draw": 0.25, "p_away": 0.15, "lambda_home": 1.8,
         "lambda_away": 0.9, "top_scoreline": "2-0", "result": "0-1"},
    ]
    m = {d["key"]: d for d in accuracy.metrics(played)}
    assert m["outcome"]["total"] == 2 and m["outcome"]["correct"] == 1
    assert m["favourite"]["correct"] == 1
    assert m["exact"]["correct"] == 1
    assert m["gd"]["correct"] == 1            # 2-0 GD matches the first match only
    assert m["ou25"]["total"] == 2           # computable (lambdas present)
    assert m["outcome"]["pct"] == 50.0


def test_missing_lambdas_not_counted_for_goal_metrics():
    played = [{"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
               "top_scoreline": "1-0", "result": "1-0"}]
    m = {d["key"]: d for d in accuracy.metrics(played)}
    assert m["ou25"]["total"] == 0 and m["btts"]["total"] == 0   # not computable
    assert m["outcome"]["total"] == 1 and m["exact"]["correct"] == 1


def test_unplayed_or_bad_result_ignored():
    played = [{"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "result": None},
              {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "result": "bad"}]
    m = {d["key"]: d for d in accuracy.metrics(played)}
    assert m["outcome"]["total"] == 0 and m["outcome"]["pct"] is None


def test_played_from_payload_collects_finished():
    payload = {
        "friendlies": [{"played": True, "result": "1-0", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2},
                       {"played": False, "result": None}],
        "matches": [{"result": "2-2", "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3},
                    {"result": None}],
    }
    assert len(accuracy.played_from_payload(payload)) == 2
