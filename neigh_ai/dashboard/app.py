from pathlib import Path

from dash import Dash, dcc, html, page_container
from flask import send_from_directory


def create_app() -> Dash:
    app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

    app.layout = html.Div([dcc.Location(id="url"), page_container])

    images_folder = Path(__file__).resolve().parent.parent.parent / "data" / "images"

    @app.server.route("/data/images/<path:filename>")
    def serve_image(filename: str):  # type: ignore[reportUnusedFunction]
        return send_from_directory(images_folder, filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)  # type: ignore[reportUnknownMemberType]
