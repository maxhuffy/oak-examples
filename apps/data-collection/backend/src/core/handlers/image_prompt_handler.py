import base64
import cv2
import numpy as np
from typing import Tuple, List, Dict
from core.handlers.base_embedding_handler import BasePromptHandler


def decode_base64_image(base64_data_uri: str) -> np.ndarray:
    """Convert base64-encoded image to OpenCV array."""
    if "," in base64_data_uri:
        _, base64_data = base64_data_uri.split(",", 1)
    else:
        base64_data = base64_data_uri
    np_arr = np.frombuffer(base64.b64decode(base64_data), np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


class ImagePromptHandler(BasePromptHandler):
    """Handles decoding and embedding extraction for uploaded images."""

    def process(self, payload: Dict) -> Tuple[np.ndarray, List[str], np.ndarray]:
        image = decode_base64_image(payload["data"])
        image_features = self.encoder.extract_embeddings(image)
        dummy = self._make_dummy()
        class_names = [payload["filename"].split(".")[0]]

        self._update_labels(class_names, offset=self.encoder.max_classes)
        return image_features, class_names, dummy
