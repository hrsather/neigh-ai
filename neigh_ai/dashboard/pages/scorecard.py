import random
from typing import Optional, cast
from urllib.parse import parse_qs

import plotly.express as px
from dash import (
    Input,
    Output,
    callback,
    callback_context,
    dcc,
    html,
    register_page,
)
from rapidfuzz import process

from neigh_ai.feature_extraction.hip_sheet import get_family_tree, get_names


class Scorecard:
    def __init__(self) -> None:
        self.horse_names: list[str] = get_names()

        # If not present, return List of N/A
        # Will populate with real data
        self._family_tree_dict: dict[str, dict[str, str]] = get_family_tree()

    def get_layout(self) -> html.Div:
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
            if not name:
                return html.Div("N/A", style={"text-align": "center", "padding": padding})
            if name not in self.horse_names:
                return html.Div(name, style={"text-align": "center", "padding": padding})
            return html.Div(html.A(name, href=f"?name={name}"), style={"text-align": "center", "padding": padding})

        # Parents and grandparents
        sire = self._family_tree_dict[horse_name]["sire"]
        dam = self._family_tree_dict[horse_name]["dam"]
        sire_sire = self._family_tree_dict[horse_name]["paternal_grandsire"]
        sire_dam = self._family_tree_dict[horse_name]["paternal_granddam"]
        dam_sire = self._family_tree_dict[horse_name]["maternal_grandsire"]
        dam_dam = self._family_tree_dict[horse_name]["maternal_granddam"]

        horse_col = html.Div(style={"text-align": "center", "padding": "15px"}, children=[html.A(horse_name)])
        parents_col = html.Div(
            style={"display": "flex", "flex-direction": "column", "justify-content": "space-around"},
            children=[link(sire, "10px"), link(dam, "10px")],
        )
        grandparents_col = html.Div(
            style={"display": "flex", "flex-direction": "column", "justify-content": "space-around"},
            children=[link(sire_sire), link(sire_dam), link(dam_sire), link(dam_dam)],
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

        fig = px.line(
            x=distances,
            y=speeds,
            markers=True,
            labels={"x": "Race Distance", "y": "Speed (km/h)"},
            title="Speed Chart",
            text=races,
        )
        fig.update_traces(line={"color": "royalblue", "width": 3}, textposition="top center")
        fig.update_layout(yaxis_range=[0, 90])

        return html.Div(
            style={"display": "flex", "flex-direction": "row", "align-items": "center", "gap": "40px"},
            children=[dcc.Graph(figure=fig)],
        )

    def _races(self) -> html.Div:
        last_races = [
            {"date": "2025-08-01", "place": "Kentucky", "speed": "85"},
            {"date": "2025-08-15", "place": "Belmont", "speed": "82"},
            {"date": "2025-09-01", "place": "Melbourne", "speed": "87"},
        ]

        table = html.Table(
            # Header
            [
                html.Tr([
                    html.Th("Date", style={"padding": "12px", "fontSize": "22px", "textAlign": "center"}),
                    html.Th("Place", style={"padding": "12px", "fontSize": "22px", "textAlign": "center"}),
                    html.Th("Speed", style={"padding": "12px", "fontSize": "22px", "textAlign": "center"}),
                ])
            ]
            +
            # Body
            [
                html.Tr([
                    html.Td(r["date"], style={"padding": "10px", "textAlign": "center"}),
                    html.Td(
                        dcc.Link(r["place"], href="/race"),  # all links point to /race
                        style={"padding": "10px", "textAlign": "center"},
                    ),
                    html.Td(r["speed"], style={"padding": "10px", "textAlign": "center"}),
                ])
                for r in last_races
            ],
            style={"border": "2px solid black", "borderCollapse": "collapse", "width": "100%", "fontSize": "20px"},
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
layout = page.get_layout()


@callback(
    Output("search-dropdown", "options"),
    Input("search-dropdown", "search_value"),
)
def update_options(search_value: Optional[str]) -> list[dict[str, str]]:
    if not search_value:
        return [{"label": n, "value": n} for n in page.horse_names]
    return [
        {"label": n, "value": n}
        for n, _, _ in process.extract(search_value, page.horse_names, limit=10, score_cutoff=50)
    ]


@callback(
    Output("search-dropdown", "value"),
    Output("url", "search"),
    Input("search-dropdown", "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def sync_dropdown_and_url(dropdown_value: Optional[str], url_search: str) -> tuple[Optional[str], str]:
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


@callback(Output("page-content", "children"), Input("url", "search"))
def display_horse(search: str) -> html.Div:
    if not search:
        return html.Div([html.P("Type a horse name and select from dropdown.")])

    name: Optional[str] = parse_qs(search.lstrip("?")).get("name", [None])[0]
    if not name or name not in page.horse_names:
        return html.Div([html.P("Horse not in database")])

    return page.build_horse_page(name)


register_page(__name__)
