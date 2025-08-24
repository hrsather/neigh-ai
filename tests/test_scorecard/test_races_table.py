from dash import html


def test_races_table_structure(scorecard):
    div = scorecard._races()

    # top-level is a Div
    assert isinstance(div, html.Div)

    # should have one child (the HTML table)
    children = div.children
    assert len(children) == 1
    table = children[0]
    assert isinstance(table, html.Table)

    # table should have a header row + 3 data rows
    rows = table.children
    assert len(rows) == 4  # 1 header + 3 rows

    header = rows[0]
    assert isinstance(header, html.Tr)
    header_cells = header.children
    expected_columns = ["Date", "Place", "Speed"]
    assert [cell.children for cell in header_cells] == expected_columns
