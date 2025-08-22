from pathlib import Path

from neigh_ai.dashboard.scorecard import Scorecard

IMAGES_FOLDER = Path(__file__).parent.parent.parent / "data" / "images"


def test_update_options_no_search():
    app = Scorecard(IMAGES_FOLDER)
    cb = app.app.callback_map["search-dropdown.options"]["callback"]
    fn = cb.__wrapped__  # get the raw function Dash wrapped
    result = fn("")  # call it directly
    assert isinstance(result, list)
    assert len(result) == 6


def test_update_options_no_horse():
    app = Scorecard(IMAGES_FOLDER)
    cb = app.app.callback_map["search-dropdown.options"]["callback"]
    fn = cb.__wrapped__  # get the raw function Dash wrapped
    result = fn("Not exists")  # call it directly
    assert isinstance(result, list)
    assert len(result) == 0


def test_update_options_horse():
    app = Scorecard(IMAGES_FOLDER)
    cb = app.app.callback_map["search-dropdown.options"]["callback"]
    fn = cb.__wrapped__  # get the raw function Dash wrapped
    result = fn("number 2")  # call it directly
    assert isinstance(result, list)
    assert len(result) == 1
