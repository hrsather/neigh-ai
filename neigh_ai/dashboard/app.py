from pathlib import Path

from dash import Dash, dash, dcc, html

from neigh_ai.dashboard.pages.scorecard import Scorecard


def main() -> None:
    app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

    app.layout = html.Div([dcc.Location(id="url"), dash.page_container])

    IMAGES_FOLDER = Path(__file__).parent.parent.parent / "data" / "images"
    Scorecard(IMAGES_FOLDER, app=app)

    # Callback to redirect "/" to "/scorecard"
    @app.callback(dash.Output("url", "pathname"), dash.Input("url", "pathname"), prevent_initial_call=False)  # type: ignore[reportUnknownMemberType]
    def redirect_root(pathname: str) -> str | dash.no_update:  # type: ignore[reportUnusedFunction]
        if pathname in [None, "/"]:
            return "/scorecard"
        return dash.no_update

    app.run(debug=True)  # type: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    main()
