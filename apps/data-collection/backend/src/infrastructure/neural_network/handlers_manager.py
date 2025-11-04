from core.label_manager import LabelManager
from core.handlers.text_prompt_handler import TextPromptHandler
from core.handlers.image_prompt_handler import ImagePromptHandler
from core.handlers.bbox_prompt_handler import BBoxPromptHandler
from infrastructure.neural_network.annotation_node import AnnotationNode
from infrastructure.neural_network.encoders_manager import EncodersManager
from depthai_nodes.node import ImgDetectionsFilter


class HandlersManager:
    def __init__(
        self,
        encoders: EncodersManager,
        det_filter: ImgDetectionsFilter,
        annotation_node: AnnotationNode,
    ):
        self.label_manager = LabelManager(det_filter, annotation_node)
        self.encoders = encoders
        self.class_update_handler = self._class_update_handler()
        self.image_update_handler = self._image_update_handler()
        self.bbox_prompt_handler = self._bbox_prompt_handler()

    def _class_update_handler(self) -> TextPromptHandler:
        return TextPromptHandler(self.encoders.textual_encoder, self.label_manager)

    def _image_update_handler(self) -> ImagePromptHandler:
        return ImagePromptHandler(self.encoders.visual_encoder, self.label_manager)

    def _bbox_prompt_handler(self) -> BBoxPromptHandler:
        return BBoxPromptHandler(self.encoders.visual_encoder, self.label_manager)
