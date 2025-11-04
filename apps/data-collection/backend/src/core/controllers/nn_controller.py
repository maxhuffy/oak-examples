import depthai as dai
import numpy as np
from depthai_nodes.node import YOLOExtendedParser

from core.model_state import ModelState


class YOLONNController:
    """Handles sending  conditioning inputs to the DepthAI YOlO NN."""

    def __init__(
        self,
        image_prompt_queue: dai.InputQueue,
        text_prompt_queue: dai.InputQueue,
        precision: str,
        parser: YOLOExtendedParser,
        model_state: ModelState,
    ):
        self._image_prompt_queue = image_prompt_queue
        self._text_prompt_queue = text_prompt_queue
        self._precision = precision
        self._parser: YOLOExtendedParser = parser
        self._model_state = model_state

    def _tensor_type(self):
        return (
            dai.TensorInfo.DataType.FP16
            if self._precision == "fp16"
            else dai.TensorInfo.DataType.U8F
        )

    def _send(self, queue: dai.InputQueue, name: str, data: np.ndarray):
        nn_data = dai.NNData()
        nn_data.addTensor(name, data, dataType=self._tensor_type())
        queue.send(nn_data)

    def _send_text_inputs(self, embeddings: np.ndarray):
        """Send class text embeddings (semantic prompts) to the NN."""
        self._send(self._text_prompt_queue, "texts", embeddings)

    def _send_visual_inputs(self, embeddings: np.ndarray):
        """Send visual prompts (mask- or bbox-based) to the NN."""
        self._send(self._image_prompt_queue, "image_prompts", embeddings)

    def send_embeddings_pair(
        self,
        visual_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        class_names: list[str],
    ):
        """Send both text and visual conditioning inputs if available."""
        self._send_visual_inputs(visual_embeddings)
        self._send_text_inputs(text_embeddings)
        self._model_state.update_classes(class_names)

    def set_confidence_threshold(self, threshold: float):
        """Apply threshold update directly to the NN parser."""
        self._parser.setConfidenceThreshold(threshold)
        self._model_state.update_threshold(threshold)
