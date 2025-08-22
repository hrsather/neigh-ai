from pathlib import Path

from dash import html

from neigh_ai.dashboard.scorecard import Scorecard

IMAGES_FOLDER = Path(__file__).parent.parent.parent / "data" / "images"


def test_performance_score_structure():
    app = Scorecard(IMAGES_FOLDER)
    div = app._performance_score()

    # top-level should be a Div
    assert isinstance(div, html.Div)
    assert "children" in div.__dict__  # has children

    # children should contain H4 and H2
    children = div.children
    assert len(children) == 2
    assert isinstance(children[0], html.H4)
    assert children[0].children == "Performance Score"

    assert isinstance(children[1], html.H2)
    score = children[1].children
    assert isinstance(score, int)
    assert 40 <= score <= 100

    # color style exists
    assert "color" in children[1].style
