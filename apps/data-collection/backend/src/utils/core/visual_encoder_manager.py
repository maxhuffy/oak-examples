import cv2
import numpy as np
import onnxruntime
from .io import download_file
from .quantization import pad_and_quantize_features
from ..constants import MODEL, MAX_NUM_CLASSES


class VisualEncoderManager:
    """
    Handles visual embedding extraction using a YOLOE visual encoder.

    Loads an ONNX visual encoder, preprocesses an image input, performs
    forward inference, and returns quantized visual feature tensors
    compatible with downstream models.
    """

    def __init__(self, model_name="yoloe", precision="fp16", max_classes=80):
        self.model_name = model_name
        self.precision = precision
        self.max_classes = max_classes

    def _load(self):
        path = download_file(
            "https://huggingface.co/sokovninn/yoloe-v8l-seg-visual-encoder/resolve/main/"
            "yoloe-v8l-seg_visual_encoder.onnx",
            "yoloe-v8l-seg_visual_encoder.onnx",
        )

        self.session = onnxruntime.InferenceSession(
            path,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def extract_embeddings(self, image: np.ndarray) -> np.ndarray:
        self._load()
        image_resized = cv2.resize(image, (640, 640))
        image_array = image_resized.astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        input_tensor = np.expand_dims(image_array, axis=0).astype(np.float32)

        prompts = np.zeros((1, 1, 80, 80), dtype=np.float32)
        prompts[0, 0, 5:75, 5:75] = 1.0
        outputs = self.session.run(None, {"images": input_tensor, "prompts": prompts})

        image_embeddings = outputs[0].squeeze(0).reshape(1, -1)
        image_features = pad_and_quantize_features(
            image_embeddings, MAX_NUM_CLASSES, MODEL, self.precision
        )

        del self.session

        return image_features
