import random
from collections import defaultdict
from typing import Optional, cast
from urllib.parse import parse_qs

import plotly.express as px  # type: ignore[reportMissingTypeStubs]
from dash import (
    Input,
    Output,
    callback,  # type: ignore[reportUnknownVariableType]
    callback_context,
    dash_table,
    dcc,
    html,
    register_page,  # type: ignore[reportUnknownVariableType]
)
from rapidfuzz import process

from neigh_ai.constants import IMAGES_FOLDER


class Scorecard:
    def __init__(self) -> None:
        self.horse_names: list[str] = [p.stem for p in IMAGES_FOLDER.glob("*.jpg")]

        # If not present, return List of N/A
        # Will populate with real data
        self._family_tree_dict: dict[str, list[str]] = defaultdict(
            lambda: ["N/A", "N/A"],
            {
                "number 2": ["Sire1", "Dam1"],
                "Sire1": ["Grandsire1", "Granddam1"],
                "Dam1": ["Grandsire2", "Granddam2"],
            },
        )

    @property
    def layout(self) -> html.Div:
        return html.Div([
            dcc.Location(id="url", refresh=True),
            html.H2("Search for a horse"),
            dcc.Dropdown(
                id="search-dropdown",
                placeholder="Type to search...",
                options=[{"label": n, "value": n} for n in self.horse_names],
                clearable=False,
                searchable=True,
                style={"width": "300px"},
            ),
            html.Div(id="page-content"),
        ])

    def _score_color(self, val: int | float) -> str:
        if val < 60:
            return "red"
        elif val > 80:
            return "green"
        return "orange"

    def _generate_scores(self) -> html.Div:
        pedigree_score = random.randint(0, 100)
        vision_score = random.randint(0, 100)
        combined = 10 + (pedigree_score + vision_score) / 2

        def score_div(title: str, value: int | float) -> html.Div:
            return html.Div(
                style={"text-align": "center"},
                children=[
                    html.H4(title),
                    html.H2(f"{value}", style={"color": self._score_color(value)}),
                ],
            )

        return html.Div(
            style={"display": "flex", "gap": "40px", "margin-top": "20px"},
            children=[
                score_div("Pedigree Score", pedigree_score),
                score_div("Vision Score", vision_score),
                score_div("Combined Score", combined),
            ],
        )

    def _family_tree(self, horse_name: str) -> html.Div:
        def link(name: str, padding: str = "5px"):
            if not name or name not in self.horse_names:
                return html.Div("N/A", style={"text-align": "center", "padding": padding})
            return html.Div(html.A(name, href=f"?name={name}"), style={"text-align": "center", "padding": padding})

        # Parents and grandparents
        sire, dam = self._family_tree_dict[horse_name]
        sire_parents = self._family_tree_dict[sire]
        dam_parents = self._family_tree_dict[dam]

        horse_col = html.Div(style={"text-align": "center", "padding": "15px"}, children=[html.A(horse_name)])
        parents_col = html.Div(
            style={"display": "flex", "flex-direction": "column", "justify-content": "space-around"},
            children=[link(sire, "10px"), link(dam, "10px")],
        )
        grandparents_col = html.Div(
            style={"display": "flex", "flex-direction": "column", "justify-content": "space-around"},
            children=[link(sire_parents[0]), link(sire_parents[1]), link(dam_parents[0]), link(dam_parents[1])],
        )
        return html.Div(
            style={"display": "flex", "flex-direction": "row", "align-items": "center", "gap": "40px"},
            children=[horse_col, parents_col, grandparents_col],
        )

    def _performance_graph(self) -> html.Div:
        distances = ["400m", "800m", "1200m", "1600m", "2000m"]
        speeds = [random.randint(70, 90)]
        for _ in range(4):
            speeds.append(speeds[-1] - random.randint(5, 15))
        races = ["Kentucky", "Belmont", "Melbourne", "Dubai", "Grand National"]

        fig = px.line(  # type: ignore[assignment]
            x=distances,
            y=speeds,
            markers=True,
            labels={"x": "Race Distance", "y": "Speed (km/h)"},
            title="Speed Chart",
            text=races,
        )
        fig.update_traces(line={"color": "royalblue", "width": 3}, textposition="top center")  # type: ignore[reportUnknownMemberType]
        fig.update_layout(yaxis_range=[0, 90])  # type: ignore[reportUnknownMemberType]

        return html.Div(
            style={"display": "flex", "flex-direction": "row", "align-items": "center", "gap": "40px"},
            children=[dcc.Graph(figure=fig)],
        )

    def _races(self) -> html.Div:
        last_races: list[dict[str, str]] = [
            {"date": "2025-08-01", "place": "[Kentucky](https://example.com/kentucky)", "speed": "85"},
            {"date": "2025-08-15", "place": "[Belmont](https://example.com/belmont)", "speed": "82"},
            {"date": "2025-09-01", "place": "[Melbourne](https://example.com/melbourne)", "speed": "87"},
        ]

        table = dash_table.DataTable(  # type: ignore[assignment]
            columns=[
                {"name": "Date", "id": "date"},
                {"name": "Place", "id": "place", "presentation": "markdown"},
                {"name": "Speed", "id": "speed"},
            ],
            data=last_races,  # type: ignore[assignment]
            style_table={"width": "400px"},
        )
        return html.Div(
            style={"display": "flex", "flex-direction": "row", "align-items": "center", "gap": "40px"},
            children=[table],
        )

    def _performance_score(self) -> html.Div:
        perf_score = random.randint(40, 100)
        return html.Div(
            style={"display": "flex", "flex-direction": "row", "align-items": "center", "gap": "40px"},
            children=[
                html.H4("Performance Score"),
                html.H2(perf_score, style={"color": self._score_color(perf_score)}),
            ],
        )

    def build_horse_page(self, name: str) -> html.Div:
        return html.Div(
            style={
                "display": "flex",
                "flex-direction": "row",
                "align-items": "flex-start",
                "gap": "40px",
                "width": "100%",
            },
            children=[
                html.Div(
                    style={"flex": "1", "display": "flex", "flex-direction": "column", "align-items": "center"},
                    children=[
                        html.H1(name),
                        html.Img(
                            src=f"/data/images/{name}.jpg",
                            style={"width": "auto", "height": "250px", "margin-top": "20px"},
                        ),
                        html.H3("Family Tree"),
                        self._family_tree(name),
                        self._generate_scores(),
                    ],
                ),
                html.Div(
                    style={
                        "flex": "1",
                        "display": "flex",
                        "flex-direction": "column",
                        "justify-content": "center",
                        "align-items": "center",
                    },
                    children=[
                        self._performance_graph(),
                        self._performance_score(),
                        self._races(),
                    ],
                ),
            ],
        )


page = Scorecard()
layout = page.layout


@callback(  # type: ignore[reportUnknownMemberType]
    Output("search-dropdown", "options"),
    Input("search-dropdown", "search_value"),
)
def update_options(search_value: Optional[str]) -> list[dict[str, str]]:  # type: ignore[reportUnusedFunction]
    if not search_value:
        return [{"label": n, "value": n} for n in page.horse_names]
    return [
        {"label": n, "value": n}
        for n, _, _ in process.extract(search_value, page.horse_names, limit=10, score_cutoff=50)
    ]


@callback(  # type: ignore[reportUnknownMemberType]
    Output("search-dropdown", "value"),
    Output("url", "search"),
    Input("search-dropdown", "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def sync_dropdown_and_url(dropdown_value: Optional[str], url_search: str) -> tuple[Optional[str], str]:  # type: ignore[reportUnusedFunction]
    if not callback_context.triggered:
        return dropdown_value, url_search

    prop_id = cast(str, callback_context.triggered[0]["prop_id"])

    if prop_id.startswith("search-dropdown"):
        if dropdown_value:
            return dropdown_value, f"?name={dropdown_value}"
        return None, ""

    if prop_id.startswith("url"):
        if not url_search:
            return None, ""
        name = parse_qs(url_search.lstrip("?")).get("name", [None])[0]
        if name in page.horse_names:
            return name, url_search
        return None, url_search

    return None, ""


@callback(Output("page-content", "children"), Input("url", "search"))  # type: ignore[reportUnknownMemberType]
def display_horse(search: str) -> html.Div:  # type: ignore[reportUnusedFunction]
    if not search:
        return html.Div([html.P("Type a horse name and select from dropdown.")])

    name: Optional[str] = parse_qs(search.lstrip("?")).get("name", [None])[0]
    if not name or name not in page.horse_names:
        return html.Div([html.P("Horse not in database")])

    return page.build_horse_page(name)


register_page(__name__)
