def test_colors(scorecard) -> None:
    assert scorecard._score_color(10) == "red"
    assert scorecard._score_color(60) == "orange"
    assert scorecard._score_color(90) == "green"
