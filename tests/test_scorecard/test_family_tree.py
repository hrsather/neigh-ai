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


# def test_family_tree_dict():
#     scorecard = Scorecard()
#
#     # Test known horse
#     assert scorecard._family_tree_dict["number 2"] == ["Sire1", "Dam1"]
#     assert scorecard._family_tree_dict["Sire1"] == ["Grandsire1", "Granddam1"]
#     assert scorecard._family_tree_dict["Dam1"] == ["Grandsire2", "Granddam2"]
#
#     # Test unknown horse returns default
#     assert scorecard._family_tree_dict["UnknownHorse"] == ["N/A", "N/A"]
#
#     # Optional: test that defaultdict is actually a defaultdict
#     from collections import defaultdict
#
#     assert isinstance(scorecard._family_tree_dict, defaultdict)
