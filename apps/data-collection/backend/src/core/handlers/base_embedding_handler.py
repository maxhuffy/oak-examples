# app/modules/handlers/base_embedding_handler.py
from __future__ import annotations
from typing import Any, List
import numpy as np
from core.label_manager import LabelManager


class BasePromptHandler:
    """
    Abstract base handler for extracting embeddings from various input modalities.
    Provides shared utilities for feature extraction and label management.
    """

    def __init__(self, encoder: Any, label_manager: LabelManager):
        self.encoder = encoder
        self.label_manager = label_manager

    def _make_dummy(self) -> np.ndarray:
        """Create dummy tensor for balancing model inputs."""
        return self.encoder.make_dummy_features()

    def _update_labels(self, class_names: List[str], offset: int = 0):
        """Synchronize the label manager with new or updated class names."""
        self.label_manager.update_labels(class_names, offset=offset)
