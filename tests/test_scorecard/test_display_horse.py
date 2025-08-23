from pathlib import Path

from dash import Dash, html

from neigh_ai.dashboard.pages.scorecard import Scorecard

IMAGES_FOLDER = Path(__file__).parent.parent.parent / "data" / "images"


def test_display_horse_empty_search():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    cb = scorecard.app.callback_map["page-content.children"]["callback"]
    fn = cb.__wrapped__  # get the raw function

    result = fn("")  # empty search
    assert isinstance(result, html.Div)
    assert "Type a horse name" in result.children[0].children


def test_display_horse_invalid_name():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    scorecard.horse_names = ["Alpha", "Beta"]
    cb = scorecard.app.callback_map["page-content.children"]["callback"]
    fn = cb.__wrapped__  # raw function

    result = fn("?name=NotInDB")
    assert isinstance(result, html.Div)
    assert "Horse not in database" in result.children[0].children


def test_display_horse_valid_name():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    scorecard.horse_names = ["Alpha"]
    scorecard._build_horse_page = lambda name: f"PAGE-{name}"  # stub

    cb = scorecard.app.callback_map["page-content.children"]["callback"]
    fn = cb.__wrapped__

    result = fn("?name=Alpha")
    assert result == "PAGE-Alpha"
