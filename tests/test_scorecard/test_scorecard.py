from dash import html


def test_setup_layout(scorecard):
    assert scorecard.layout.children[0].id == "url"
    assert "Grandsire1" in [opt_dict["label"] for opt_dict in scorecard.layout.children[2].options]


def test_build_horse_page_structure(scorecard):
    name = "number 2"  # pick one from your dummy horse names

    div = scorecard._build_horse_page(name)
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
