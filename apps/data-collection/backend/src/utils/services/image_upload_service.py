import depthai as dai
from .base_service import BaseService
from ..constants import MAX_NUM_CLASSES, MODEL
from ..core.visual_encoder_manager import VisualEncoderManager
from ..core.io import base64_to_cv2_image
from ..core.quantization import make_dummy_features
from ..core.label_manager import LabelManager


class ImageUploadService(BaseService):
    """
    Handles image uploads from the frontend.

    Decodes base64-encoded images, extracts visual embeddings via
    the visual encoder, updates neural network tensors, and synchronizes
    class labels for the uploaded image.
    """

    def __init__(
        self, visualizer, precision, img_queue, text_queue, det_filter, annotation
    ):
        super().__init__(visualizer, "Image Upload Service")
        self.precision = precision
        self.img_queue = img_queue
        self.text_queue = text_queue
        self.det_filter = det_filter
        self.annotation = annotation
        self.image = None
        self.visual_encoder = VisualEncoderManager(MODEL, precision, MAX_NUM_CLASSES)
        self.label_manager = LabelManager(det_filter, annotation)

    def handle(self, payload: dict | None = None):
        self.image = base64_to_cv2_image(payload["data"])
        image_features = self.visual_encoder.extract_embeddings(self.image)

        input_NN_data_img = dai.NNData()
        input_NN_data_img.addTensor(
            "image_prompts",
            image_features,
            dataType=(
                dai.TensorInfo.DataType.FP16
                if self.precision == "fp16"
                else dai.TensorInfo.DataType.U8F
            ),
        )
        self.img_queue.send(input_NN_data_img)

        dummy = make_dummy_features(
            MAX_NUM_CLASSES, model_name=MODEL, precision=self.precision
        )
        inputNNDataTxt = dai.NNData()
        inputNNDataTxt.addTensor(
            "texts",
            dummy,
            dataType=(
                dai.TensorInfo.DataType.FP16
                if self.precision == "fp16"
                else dai.TensorInfo.DataType.U8F
            ),
        )
        self.text_queue.send(inputNNDataTxt)

        filename = payload["filename"]
        class_names = [filename.split(".")[0]]
        self.label_manager.update_labels(class_names, offset=MAX_NUM_CLASSES)
        return {"ok": True, "class": class_names}
