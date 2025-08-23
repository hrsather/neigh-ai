from pathlib import Path

from dash import Dash

from neigh_ai.dashboard.pages.scorecard import Scorecard

IMAGES_FOLDER = Path(__file__).parent.parent.parent / "data" / "images"


def test_serve_image(tmp_path: Path):
    # Arrange: create a fake image file
    images_folder = tmp_path
    image_file = images_folder / "test_horse.jpg"
    image_file.write_bytes(b"fake image data")

    scorecard = Scorecard(tmp_path, Dash(__name__, suppress_callback_exceptions=True))
    client = scorecard.server.test_client()  # Flask test client

    # Act: request the image URL
    response = client.get("/data/images/test_horse.jpg")

    # Assert
    assert response.status_code == 200
    assert response.data == b"fake image data"
    assert response.mimetype == "image/jpeg"
