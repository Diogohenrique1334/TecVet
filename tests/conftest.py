import io
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    """Blank white 640x640 image for fast inference tests."""
    return Image.fromarray(np.ones((640, 640, 3), dtype=np.uint8) * 255)


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image) -> bytes:
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def real_image() -> Optional[Image.Image]:
    """Returns the first real image from the test set, or None if unavailable."""
    test_dir = PROJECT_ROOT / "test" / "images"
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not images:
        return None
    return Image.open(images[0]).convert("RGB")
