from tokenizers import Tokenizer
import numpy as np
import onnxruntime
from .io import download_file
from .quantization import pad_and_quantize_features
from ..constants import MODEL, MAX_NUM_CLASSES


class TokenizerManager:
    """
    Handles text tokenization and embedding extraction for class names.

    Loads a CLIP-compatible tokenizer and ONNX text encoder, processes
    input class labels, and produces normalized and quantized text features
    suitable for model input.
    """

    def __init__(self, model_name="yoloe", precision="fp16", max_classes=80):
        self.model_name = model_name
        self.precision = precision
        self.max_classes = max_classes

    def _load(self):
        path = download_file(
            "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json",
            "tokenizer.json",
        )

        self.tokenizer = Tokenizer.from_file(str(path))

        model_path = download_file(
            "https://huggingface.co/Xenova/mobileclip_blt/resolve/main/onnx/text_model.onnx",
            "mobileclip_textual_hf.onnx",
        )

        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def extract_text_embeddings(self, class_names: list[str]) -> np.ndarray:
        self._load()
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("<|endoftext|>"),
            pad_token="<|endoftext|>",
        )

        encodings = self.tokenizer.encode_batch(class_names)
        text_onnx = np.array([e.ids for e in encodings], dtype=np.int64)

        if text_onnx.shape[1] < 77:
            text_onnx = np.pad(
                text_onnx, ((0, 0), (0, 77 - text_onnx.shape[1])), mode="constant"
            )

        textual_output = self.session.run(
            None,
            {
                self.session.get_inputs()[0].name: text_onnx,
            },
        )[0]

        textual_output /= np.linalg.norm(textual_output, ord=2, axis=-1, keepdims=True)

        text_features = pad_and_quantize_features(
            textual_output, MAX_NUM_CLASSES, MODEL, self.precision
        )

        del self.session

        return text_features
