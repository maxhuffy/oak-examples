from __future__ import annotations
from typing import Tuple, List
import numpy as np
from core.handlers.base_prompt_handler import BasePromptHandler


class TextPromptHandler(BasePromptHandler):
    """Handles embedding extraction and label synchronization for class name updates."""

    def process(self, new_classes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        embeddings = self.encoder.extract_embeddings(new_classes)
        dummy = self._make_dummy()
        self._update_labels(new_classes)
        return embeddings, dummy
