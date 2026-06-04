from wc26 import backtest


def test_model_beats_baselines_and_is_well_calibrated():
    report = backtest.run(write=False)
    t = report["test"]
    b = report["baselines"]
    # beats uniform, climatology, and the always-favourite baseline
    assert report["beats_climatology"] is True
    assert t["log_loss"] < b["climatology_log_loss"] < b["uniform_log_loss"]
    assert t["log_loss"] < b["favourite_log_loss"]
    # reliability slope near 1 (well calibrated home-win probabilities)
    assert 0.85 <= report["reliability"]["home_win_slope"] <= 1.20
    # and it beats climatology specifically on tournament-like matches
    to = report["tournament_only"]
    assert to["log_loss"] < to["climatology_log_loss"]
