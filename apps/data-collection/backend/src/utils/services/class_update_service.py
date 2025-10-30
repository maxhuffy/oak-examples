from .base_service import BaseService
import depthai as dai
from ..constants import MAX_NUM_CLASSES, MODEL
from ..core.tokenizer_manager import TokenizerManager
from ..core.quantization import make_dummy_features
from ..core.label_manager import LabelManager


class ClassUpdateService(BaseService):
    """
    Handles class updates from the frontend.

    Regenerates text embeddings for new class labels, updates the
    corresponding neural network inputs, and synchronizes label mappings
    in the detection and annotation pipeline.
    """

    def __init__(
        self,
        visualizer,
        text_queue,
        img_queue,
        precision,
        current_classes,
        det_filter,
        annotation,
    ):
        super().__init__(visualizer, "Class Update Service")
        self.text_queue = text_queue
        self.img_queue = img_queue
        self.precision = precision
        self.current_classes = current_classes
        self.det_filter = det_filter
        self.annotation = annotation
        self.embedder = TokenizerManager(
            model_name=MODEL, precision=precision, max_classes=MAX_NUM_CLASSES
        )
        self.label_manager = LabelManager(det_filter, annotation)

    def handle(self, payload: list | None = None):
        new_classes = payload
        if not new_classes:
            print("[ClassUpdateService] Empty class list.")
            return

        if len(new_classes) > MAX_NUM_CLASSES:
            print(
                f"Too many classes ({len(new_classes)}) > {MAX_NUM_CLASSES}, skipping."
            )
            return

        feats = self.embedder.extract_text_embeddings(new_classes)

        nn_txt = dai.NNData()
        nn_txt.addTensor(
            "texts",
            feats,
            dataType=(
                dai.TensorInfo.DataType.FP16
                if self.precision == "fp16"
                else dai.TensorInfo.DataType.U8F
            ),
        )
        self.text_queue.send(nn_txt)

        dummy = make_dummy_features(MAX_NUM_CLASSES, MODEL, self.precision)
        nn_img = dai.NNData()
        nn_img.addTensor(
            "image_prompts",
            dummy,
            dataType=(
                dai.TensorInfo.DataType.FP16
                if self.precision == "fp16"
                else dai.TensorInfo.DataType.U8F
            ),
        )
        self.img_queue.send(nn_img)

        self.label_manager.update_labels(new_classes)
        self.current_classes.clear()
        self.current_classes.extend(new_classes)

        print(f"[ClassUpdateService] Classes updated: {new_classes}")
        return {"ok": True, "classes": new_classes}
