from pathlib import Path
from unittest.mock import patch

from dash import Dash

from neigh_ai.dashboard.pages.scorecard import Scorecard

IMAGES_FOLDER = Path(__file__).parent.parent.parent / "data" / "images"


def test_sync_dropdown_no_trigger():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    cb = scorecard.app.callback_map["..search-dropdown.value...url.search.."]["callback"]
    fn = cb.__wrapped__

    # No trigger scenario
    with patch("dash.callback_context") as mock_ctx:
        mock_ctx.triggered = []
        result = fn("Alpha", "?name=Alpha")  # dropdown_value, url_search
        assert result == ("Alpha", "?name=Alpha")


def test_sync_dropdown_trigger_dropdown_value():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    cb = scorecard.app.callback_map["..search-dropdown.value...url.search.."]["callback"]
    fn = cb.__wrapped__

    # Dropdown triggered
    with patch("dash.callback_context") as mock_ctx:
        mock_ctx.triggered = [{"prop_id": "search-dropdown.value"}]
        result = fn("Alpha", "")  # dropdown_value, url_search
        assert result == ("Alpha", "?name=Alpha")


def test_sync_dropdown_trigger_url():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    cb = scorecard.app.callback_map["..search-dropdown.value...url.search.."]["callback"]
    fn = cb.__wrapped__

    # URL triggered
    with patch("dash.callback_context") as mock_ctx:
        mock_ctx.triggered = [{"prop_id": "url.search"}]
        result = fn(None, "?name=Alpha")  # dropdown_value, url_search
        assert result == (None, "?name=Alpha")


def test_sync_dropdown_trigger_url_match():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    cb = scorecard.app.callback_map["..search-dropdown.value...url.search.."]["callback"]
    fn = cb.__wrapped__

    # URL triggered
    with patch("dash.callback_context") as mock_ctx:
        mock_ctx.triggered = [{"prop_id": "url.search"}]
        result = fn(None, "?name=Dam1")  # dropdown_value, url_search
        assert result == ("Dam1", "?name=Dam1")


def test_sync_dropdown_dropdown_empty():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    cb = scorecard.app.callback_map["..search-dropdown.value...url.search.."]["callback"]
    fn = cb.__wrapped__

    # Patch callback_context to simulate dropdown trigger
    with patch("dash.callback_context") as mock_ctx:
        mock_ctx.triggered = [{"prop_id": "search-dropdown.value"}]
        dropdown_value = None  # empty value triggers the None branch
        url_search = ""  # irrelevant in this case
        result = fn(dropdown_value, url_search)
        assert result == (None, "")


def test_sync_dropdown_unknown_trigger():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    cb = scorecard.app.callback_map["..search-dropdown.value...url.search.."]["callback"]
    fn = cb.__wrapped__  # get raw function

    fake_trigger = [{"prop_id": "some.other.input"}]  # not search-dropdown or url

    with patch("dash.callback_context") as mock_ctx:
        mock_ctx.triggered = fake_trigger
        dropdown_value, url_search = fn("foo", "?name=foo")

    # The bottom catch-all branch returns (None, "")
    assert dropdown_value is None
    assert url_search == ""


def test_sync_dropdown_trigger_url_none():
    scorecard = Scorecard(IMAGES_FOLDER, Dash(__name__, suppress_callback_exceptions=True))
    cb = scorecard.app.callback_map["..search-dropdown.value...url.search.."]["callback"]
    fn = cb.__wrapped__

    # Patch callback_context to simulate URL trigger
    with patch("dash.callback_context") as mock_ctx:
        # Dash expects a CallbackContext object with a triggered property
        mock_ctx.triggered = [{"prop_id": "url.search"}]
        dropdown_value = None
        url_search = ""  # This will trigger the `if not url_search:` branch
        result = fn(dropdown_value, url_search)
        assert result == (None, "")
