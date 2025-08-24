from dash import dcc, html

from neigh_ai.dashboard.pages.race import RaceResults  # adjust import as needed


def test_race_results_layout():
    race_page = RaceResults()
    layout = race_page.layout

    # top-level Div
    assert isinstance(layout, html.Div)
    assert "display" in layout.style and layout.style["display"] == "flex"

    # should have two children: info and table
    children = layout.children
    assert len(children) == 2

    info_div = children[0]
    table_div = children[1]

    # left info section
    assert isinstance(info_div, html.Div)
    info_children = info_div.children
    assert any(isinstance(c, html.H2) and race_page.race_info["name"] in c.children for c in info_children)
    assert any(isinstance(c, html.P) and race_page.race_info["date"] in c.children for c in info_children)
    assert any(isinstance(c, html.P) and race_page.race_info["location"] in c.children for c in info_children)
    assert any(isinstance(c, html.P) and race_page.race_info["conditions"] in c.children for c in info_children)

    # right table section
    assert isinstance(table_div, html.Div)
    table = table_div.children[0]
    assert isinstance(table, html.Table)

    # check table header
    thead = table.children[0]
    assert isinstance(thead, html.Thead)
    header_cells = thead.children.children  # html.Tr -> children
    expected_headers = ["Place", "Horse", "Time (s)"]
    assert [c.children for c in header_cells] == expected_headers

    # check table body
    tbody = table.children[1]
    assert isinstance(tbody, html.Tbody)
    rows = tbody.children
    assert len(rows) == len(race_page.horses)

    for idx, row in enumerate(rows):
        cells = row.children
        # place
        assert cells[0].children == idx + 1
        # horse link
        horse_link = cells[1].children
        assert isinstance(horse_link, dcc.Link)
        assert horse_link.children in race_page.horses
        assert horse_link.href == f"/scorecard?name={horse_link.children}"
        # time
        assert isinstance(cells[2].children, float)
