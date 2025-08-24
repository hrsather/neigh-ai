from dash import Dash, dcc, html, page_container
from flask import send_from_directory

from neigh_ai.constants import get_images_folder


def create_app() -> Dash:
    app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

    app.layout = html.Div([dcc.Location(id="url"), page_container])

    @app.server.route("/data/images/<path:filename>")
    def serve_image(filename: str):
        return send_from_directory(get_images_folder(), filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8050, debug=True)
