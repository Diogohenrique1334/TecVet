from pathlib import Path
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path("runs/detect/train/weights/best.pt")
DEFAULT_CONFIDENCE = 0.25


def load_model(model_path: Path = MODEL_PATH) -> YOLO:
    return YOLO(str(model_path))


def run_prediction(model: YOLO, image: Image.Image, confidence: float = DEFAULT_CONFIDENCE) -> dict:
    """Run neutrophil detection on a PIL image.

    Returns:
        dict with keys:
            - annotated_image: PIL.Image with bounding boxes
            - detections: list of {box, confidence, class_name}
            - count: total neutrophils detected
    """
    results = model.predict(image, conf=confidence, verbose=False)
    result = results[0]
    boxes = result.boxes

    annotated_bgr = result.plot()
    annotated_image = Image.fromarray(annotated_bgr[..., ::-1])  # BGR -> RGB

    detections = [
        {
            "box": boxes.xyxy[i].tolist(),
            "confidence": round(float(boxes.conf[i]), 4),
            "class_name": result.names[int(boxes.cls[i])],
        }
        for i in range(len(boxes))
    ]

    return {
        "annotated_image": annotated_image,
        "detections": detections,
        "count": len(detections),
    }
