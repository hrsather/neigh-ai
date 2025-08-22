from pathlib import Path

from dash import dcc, html

from neigh_ai.dashboard.scorecard import Scorecard

IMAGES_FOLDER = Path(__file__).parent.parent.parent / "data" / "images"


def test_performance_graph_structure():
    app = Scorecard(IMAGES_FOLDER)
    div = app._performance_graph()

    # Top-level div
    assert isinstance(div, html.Div)

    children = div.children
    assert len(children) == 1

    graph = children[0]
    assert isinstance(graph, dcc.Graph)
    fig = graph.figure

    # Check that figure is a plotly figure with expected data
    assert fig.layout.title.text == "Speed Chart"
    assert len(fig.data) == 1  # only one line trace
    trace = fig.data[0]
    assert list(trace.x) == ["400m", "800m", "1200m", "1600m", "2000m"]
    assert list(trace.text) == ["Kentucky", "Belmont", "Melbourne", "Dubai", "Grand National"]
    assert trace.line.color == "royalblue"
