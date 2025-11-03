from services.base_service import BaseService
from services.bbox_prompt.bbox_prompt_payload import BBoxPromptPayload
from core.handlers.bbox_prompt_handler import BBoxPromptHandler
from core.controllers.nn_controller import YOLONNController
from infrastructure.frame_cache_node import FrameCacheNode
from services.service_name import ServiceName


class BBoxPromptService(BaseService[BBoxPromptPayload]):
    NAME = ServiceName.BBOX_PROMPT

    def __init__(
        self,
        handler: BBoxPromptHandler,
        frame_cache: FrameCacheNode,
        repository: YOLONNController,
    ):
        super().__init__()
        self.handler = handler
        self.frame_cache = frame_cache
        self.repository = repository

    def handle(self, payload: BBoxPromptPayload):
        image = self.frame_cache.get_last_frame()
        if image is None:
            return {"ok": False, "reason": "no_frame_available"}

        try:
            embeddings, class_names, dummy = self.handler.process(
                image, payload["bbox"]
            )
        except ValueError as e:
            return {"ok": False, "reason": "invalid_bbox"}

        self.repository.send_embeddings_pair(embeddings, dummy, class_names)

        return {"ok": True, "classes": class_names}
