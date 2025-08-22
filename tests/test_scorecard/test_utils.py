from pathlib import Path

from neigh_ai.dashboard.scorecard import Scorecard

IMAGES_FOLDER = Path(__file__).parent.parent / "data" / "images"


def test_colors() -> None:
    scorecard = Scorecard(IMAGES_FOLDER)
    assert scorecard._score_color(10) == "red"
    assert scorecard._score_color(60) == "orange"
    assert scorecard._score_color(90) == "green"
