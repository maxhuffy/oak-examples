import numpy as np
from typing import Tuple, Dict, List
from core.handlers.base_prompt_handler import BasePromptHandler


class BBoxPromptHandler(BasePromptHandler):
    """Handles extraction of embeddings for a specific bounding box region."""

    def process(
        self, image: np.ndarray, bbox: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        H, W = image.shape[:2]
        bx, by, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        x0, y0 = int(bx * W), int(by * H)
        x1, y1 = int((bx + bw) * W), int((by + bh) * H)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid bbox coordinates: {(x0, y0, x1, y1)}")

        mask = np.zeros((H, W), dtype=np.float32)
        mask[y0:y1, x0:x1] = 1.0

        embeddings = self.encoder.extract_embeddings(image, mask)
        dummy = self._make_dummy()
        class_names = ["Bbox Object"]

        self._update_labels(class_names, offset=self.encoder.max_classes)
        return embeddings, class_names, dummy
