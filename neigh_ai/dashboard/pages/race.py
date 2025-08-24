import random

from dash import dcc, html, register_page


class RaceResults:
    def __init__(self):
        self.race_info = {
            "name": "Grand Prix Stakes",
            "date": "2025-08-23",
            "location": "Belmont Park",
            "conditions": "Fast track, sunny",
        }
        self.horses = ["Sire1", "Dam1", "Grandsire1", "Granddam1", "number 2"]

    def generate_results(self):
        times = sorted([round(random.uniform(55.0, 65.0), 2) for _ in self.horses])
        results = [{"Horse": h, "Time": t} for h, t in zip(self.horses, times)]
        results.sort(key=lambda r: r["Time"])
        return results

    @property
    def layout(self):
        results = self.generate_results()

        table_rows = [
            html.Tr([
                html.Td(idx + 1, style={"padding": "10px", "textAlign": "center"}),
                html.Td(
                    dcc.Link(r["Horse"], href=f"/scorecard?name={r['Horse']}"),
                    style={"padding": "10px", "textAlign": "center"},
                ),
                html.Td(r["Time"], style={"padding": "10px", "textAlign": "center"}),
            ])
            for idx, r in enumerate(results)
        ]

        # Flex container: left = info, right = table
        return html.Div(
            style={"display": "flex", "gap": "50px", "alignItems": "flex-start"},
            children=[
                html.Div(
                    style={"flex": "1", "fontSize": "20px"},
                    children=[
                        html.H2(f"Race: {self.race_info['name']}", style={"fontSize": "32px"}),
                        html.P(f"Date: {self.race_info['date']}", style={"fontSize": "24px"}),
                        html.P(f"Location: {self.race_info['location']}", style={"fontSize": "24px"}),
                        html.P(f"Conditions: {self.race_info['conditions']}", style={"fontSize": "24px"}),
                    ],
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Table(
                            [
                                html.Thead(
                                    html.Tr([
                                        html.Th("Place", style={"padding": "12px", "fontSize": "22px"}),
                                        html.Th("Horse", style={"padding": "12px", "fontSize": "22px"}),
                                        html.Th("Time (s)", style={"padding": "12px", "fontSize": "22px"}),
                                    ])
                                ),
                                html.Tbody(table_rows),
                            ],
                            style={
                                "border": "2px solid black",
                                "borderCollapse": "collapse",
                                "width": "100%",
                                "fontSize": "20px",
                            },
                        )
                    ],
                ),
            ],
        )


# Register page
page = RaceResults()
layout = page.layout
register_page(__name__, path="/race")
