from core.controllers.nn_controller import YOLONNController
from infrastructure.frame_cache_node import FrameCacheNode
from infrastructure.neural_network.handlers_manager import HandlersManager
from services.class_update.class_update_service import ClassUpdateService
from services.threshold_update.threshold_update_service import ThresholdUpdateService
from services.image_upload.image_upload_service import ImageUploadService
from services.bbox_prompt.bbox_prompt_service import BBoxPromptService


class NNServiceManager:
    def __init__(
        self,
        controller: YOLONNController,
        handlers: HandlersManager,
        frame_cache: FrameCacheNode,
    ):
        self.controller = controller
        self.handlers = handlers
        self.frame_cache = frame_cache
        self.services = self._build_services()

    def _build_services(self):
        return [
            ClassUpdateService(self.controller, self.handlers.class_update_handler),
            ThresholdUpdateService(self.controller),
            ImageUploadService(self.controller, self.handlers.image_update_handler),
            BBoxPromptService(
                self.handlers.bbox_prompt_handler,
                self.frame_cache,
                self.controller,
            ),
        ]
