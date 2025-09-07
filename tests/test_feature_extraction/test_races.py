from neigh_ai.feature_extraction.power_curve import RaceModel


def test_points() -> None:
    race_model = RaceModel
    assert race_model._score_finish(1) == 10
    assert race_model._score_finish(2) == 9
    assert race_model._score_finish(10) == 1
    assert race_model._score_finish(11) == 0
    assert race_model._score_finish(15) == 0
