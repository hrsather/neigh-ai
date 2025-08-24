from dash import html

from neigh_ai.dashboard.pages.scorecard import Scorecard


def test_family_tree():
    scorecard = Scorecard()
    horse_name = "number 2"

    # Act
    tree_div = scorecard._family_tree(horse_name)

    # Assert top-level Div
    assert isinstance(tree_div, html.Div)
    assert tree_div.style["display"] == "flex"
    assert len(tree_div.children) == 3  # horse, parents, grandparents

    horse_col, parents_col, grandparents_col = tree_div.children

    # Assert horse column
    assert isinstance(horse_col, html.Div)
    assert horse_col.children[0].children == horse_name

    # Assert parents column
    assert isinstance(parents_col, html.Div)
    assert len(parents_col.children) == 2
    assert all(isinstance(c, html.Div) for c in parents_col.children)

    # Assert grandparents column
    assert isinstance(grandparents_col, html.Div)
    assert len(grandparents_col.children) == 4
    assert all(isinstance(c, html.Div) for c in grandparents_col.children)


def test_family_tree_dict():
    scorecard = Scorecard()
    tree = scorecard._family_tree_dict

    # Known horses
    assert tree["Thunder Gulch"]["sire"] == "Hennessy"
    assert tree["Thunder Gulch"]["dam"] == "Personal Lady"

    assert tree["Hennessy"]["sire"] == "Storm Cat"
    assert tree["Hennessy"]["dam"] == "N/A"

    assert tree["Personal Lady"]["sire"] == "Smart Strike"
    assert tree["Personal Lady"]["dam"] == "Randie's Legend"

    assert tree["Randie's Legend"]["sire"] == "Benchmark"
    assert tree["Randie's Legend"]["dam"] == "N/A"

    # Grandparents should also be resolved
    assert tree["Thunder Gulch"]["paternal_grandsire"] == "Storm Cat"
    assert tree["Thunder Gulch"]["maternal_grandsire"] == "Smart Strike"

    # Missing / unknown horses fallback to N/A
    assert tree["UnknownHorse"]["sire"] == "N/A"
    assert tree["UnknownHorse"]["dam"] == "N/A"
