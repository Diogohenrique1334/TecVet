import pytest
from PIL import Image

from predictor import DEFAULT_CONFIDENCE, MODEL_PATH, load_model, run_prediction


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Modelo não encontrado em: {MODEL_PATH}"


def test_load_model():
    model = load_model()
    assert model is not None


def test_run_prediction_returns_expected_keys(sample_image):
    model = load_model()
    result = run_prediction(model, sample_image)
    assert "annotated_image" in result
    assert "detections" in result
    assert "count" in result


def test_annotated_image_is_pil(sample_image):
    model = load_model()
    result = run_prediction(model, sample_image)
    assert isinstance(result["annotated_image"], Image.Image)


def test_annotated_image_same_size_as_input(sample_image):
    model = load_model()
    result = run_prediction(model, sample_image)
    assert result["annotated_image"].size == sample_image.size


def test_count_matches_detections_length(sample_image):
    model = load_model()
    result = run_prediction(model, sample_image)
    assert result["count"] == len(result["detections"])


def test_detection_structure(sample_image):
    model = load_model()
    result = run_prediction(model, sample_image)
    for det in result["detections"]:
        assert "box" in det
        assert "confidence" in det
        assert "class_name" in det
        assert len(det["box"]) == 4
        assert 0.0 <= det["confidence"] <= 1.0


def test_confidence_filter(sample_image):
    """High confidence threshold should return fewer or equal detections."""
    model = load_model()
    result_low = run_prediction(model, sample_image, confidence=0.10)
    result_high = run_prediction(model, sample_image, confidence=0.90)
    assert result_high["count"] <= result_low["count"]


def test_default_confidence_value():
    assert DEFAULT_CONFIDENCE == 0.25
