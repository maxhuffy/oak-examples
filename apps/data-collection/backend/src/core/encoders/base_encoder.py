import os
from abc import ABC, abstractmethod
import onnxruntime
import numpy as np
import requests
from pathlib import Path

from config.yaml_loader import YamlLoader


class BaseEncoder(ABC):
    """
    Abstract base class for all embedding encoders (visual, text, etc.).

    Provides:
      - Model download & caching via `download_file`
      - ONNX session initialization
      - Common quantization pipeline
    """

    def __init__(
        self,
        precision: str,
        max_classes: int,
        config: YamlLoader,
        encoder_model_url: str,
        encoder_model_path: str,
    ):
        self.model_name = config.name
        self.precision = precision
        self.max_classes = max_classes
        self.quant_values = config.quant_values
        self.encoder_model_url = encoder_model_url
        self.encoder_model_path = encoder_model_path
        self.providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.session = None

    def _load_model(self) -> None:
        """Download and initialize the ONNX model."""
        path = self._download_file()
        self.session = onnxruntime.InferenceSession(path, providers=self.providers)

    @abstractmethod
    def extract_embeddings(self, *args, **kwargs) -> np.ndarray:
        """Subclasses must implement modality-specific preprocessing and inference."""
        pass

    def _pad_and_quantize_features(self, features):
        """
        Pad features to (1, 512, max_num_classes) and quantize if precision is int8.
        For FP16, return padded float16 features (no quantization).
        """
        num_padding = self.max_classes - features.shape[0]
        padded = np.pad(features, ((0, num_padding), (0, 0)), "constant").T.reshape(
            1, 512, self.max_classes
        )

        if self.precision == "fp16":
            return padded.astype(np.float16)

        quant = self.quant_values[self.model_name]
        out = (padded / quant["quant_scale"]) + quant["quant_zero_point"]
        return out.astype(np.uint8)

    def make_dummy_features(self) -> np.ndarray:
        """
        Create a dummy tensor of shape (1, 512, max_num_classes) for model input.
        For FP16, return zeros; for INT8, fill with the model's quantization zero point.
        """
        if self.precision == "fp16":
            return np.zeros((1, 512, self.max_classes), dtype=np.float16)
        qzp = int(
            round(self.quant_values.get(self.model_name, {}).get("quant_zero_point", 0))
        )
        return np.full((1, 512, self.max_classes), qzp, dtype=np.uint8)

    def _download_file(self, url: str = "", path: str = "") -> Path:
        if url == "":
            url = self.encoder_model_url
        if path == "":
            path = self.encoder_model_path
        if not os.path.exists(path):
            print(f"Downloading tokenizer config from {url}...")
            with open(path, "wb") as f:
                f.write(requests.get(url).content)
        return path
