from services.base_service import BaseService
from services.class_update.class_update_payload import ClassUpdatePayload
from core.handlers.text_prompt_handler import TextPromptHandler
from core.controllers.nn_controller import YOLONNController
from services.service_name import ServiceName


class ClassUpdateService(BaseService[ClassUpdatePayload]):
    """Coordinates text-based class updates across model, repository, and state."""

    NAME = ServiceName.CLASS_UPDATE

    def __init__(
        self,
        repository: YOLONNController,
        handler: TextPromptHandler,
    ):
        super().__init__()
        self.repository = repository
        self.handler = handler

    def handle(self, payload: ClassUpdatePayload) -> dict[str, any]:
        new_classes = payload["classes"]
        if not new_classes:
            return {"ok": False, "reason": "empty_class_list"}

        text_inputs, dummy = self.handler.process(new_classes)

        self.repository.send_embeddings_pair(dummy, text_inputs, new_classes)

        return {"ok": True, "classes": new_classes}
