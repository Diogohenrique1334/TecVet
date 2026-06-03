import io

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_predict_valid_image(sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_predict_returns_neutrophil_count_header(sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert "x-neutrophil-count" in response.headers
    assert response.headers["x-neutrophil-count"].isdigit()


def test_predict_png_image(sample_image):
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    response = client.post(
        "/predict",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 200


def test_predict_unsupported_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_response_is_valid_image(sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    from PIL import Image
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0 and img.size[1] > 0
