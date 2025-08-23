from dash import dash_table, html


def test_races_table_structure(scorecard):
    div = scorecard._races()

    # top-level is a Div
    assert isinstance(div, html.Div)

    # should have one child (the DataTable)
    children = div.children
    assert len(children) == 1
    table = children[0]
    assert isinstance(table, dash_table.DataTable)

    # check columns
    expected_columns = ["Date", "Place", "Speed"]
    table_columns = [col["name"] for col in table.columns]
    assert table_columns == expected_columns

    # check data
    data = table.data
    assert isinstance(data, list)
    assert all("date" in row and "place" in row and "speed" in row for row in data)
    assert len(data) == 3

    # check markdown presentation on 'place'
    place_col = next(col for col in table.columns if col["id"] == "place")
    assert place_col.get("presentation") == "markdown"
