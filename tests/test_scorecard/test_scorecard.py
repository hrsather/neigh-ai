from pathlib import Path
from unittest.mock import patch

from dash import html

from neigh_ai.dashboard.scorecard import Scorecard

IMAGES_FOLDER = Path(__file__).parent.parent.parent / "data" / "images"


def test_setup_layout():
    # Arrange
    app = Scorecard(IMAGES_FOLDER)  # images_folder can be dummy for layout test

    # Act
    app._setup_layout()
    layout = app.app.layout

    # Assert: layout is a Div
    assert layout._type == "Div"

    # Assert: it has children
    children = layout.children
    assert len(children) == 4

    # First child is Location
    assert children[0]._type == "Location"
    # Second child is H2
    assert children[1]._type == "H2"
    assert children[1].children == "Search for a horse"

    # Third child is Dropdown
    dropdown = children[2]
    assert dropdown._type == "Dropdown"
    assert dropdown.id == "search-dropdown"

    # Fourth child is page-content Div
    assert children[3]._type == "Div"
    assert children[3].id == "page-content"


def test_build_horse_page_structure():
    app = Scorecard(IMAGES_FOLDER)
    name = "number 2"  # pick one from your dummy horse names

    div = app._build_horse_page(name)
    assert isinstance(div, html.Div)
    assert len(div.children) == 2  # left and right columns

    left_col, right_col = div.children

    # Left column checks
    assert isinstance(left_col, html.Div)
    left_children = left_col.children
    assert any(isinstance(c, html.H1) and c.children == name for c in left_children)
    assert any(isinstance(c, html.Img) for c in left_children)
    assert any(isinstance(c, html.H3) and "Family Tree" in c.children for c in left_children)

    # Right column checks
    assert isinstance(right_col, html.Div)
    right_children = right_col.children
    assert any(isinstance(c, html.Div) for c in right_children)  # performance graph/divs


def test_run_calls_dash_run():
    app = Scorecard(IMAGES_FOLDER)
    with patch.object(app.app, "run") as mock_run:
        app.run()
        mock_run.assert_called_once_with(debug=True)
