from unittest.mock import patch

from dash import html


def test_generate_scores(scorecard):
    # Patch random.randint to return predictable scores
    with patch("random.randint", side_effect=[70, 80]):
        result = scorecard._generate_scores()

    # Assert top-level Div
    assert isinstance(result, html.Div)
    assert result.style["display"] == "flex"
    assert len(result.children) == 3

    # Extract individual score divs
    pedigree_div, vision_div, combined_div = result.children

    # Pedigree Score
    assert pedigree_div.children[0].children == "Pedigree Score"
    assert pedigree_div.children[1].children == "70"
    assert pedigree_div.children[1].style["color"] == scorecard._score_color(70)

    # Vision Score
    assert vision_div.children[0].children == "Vision Score"
    assert vision_div.children[1].children == "80"
    assert vision_div.children[1].style["color"] == scorecard._score_color(80)

    # Combined Score
    combined_value = 10 + (70 + 80) / 2
    assert combined_div.children[0].children == "Combined Score"
    assert combined_div.children[1].children == str(combined_value)
    assert combined_div.children[1].style["color"] == scorecard._score_color(combined_value)
