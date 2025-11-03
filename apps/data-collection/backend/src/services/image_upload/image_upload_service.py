from services.base_service import BaseService
from services.image_upload.image_upload_payload import ImageUploadPayload
from core.handlers.image_upload_handler import ImagePromptHandler
from core.controllers.nn_controller import YOLONNController
from services.service_name import ServiceName


class ImageUploadService(BaseService[ImageUploadPayload]):
    """Coordinates image upload flow: decode → extract → send → update labels."""

    NAME = ServiceName.IMAGE_UPLOAD

    def __init__(self, repository: YOLONNController, handler: ImagePromptHandler):
        super().__init__()
        self.repository = repository
        self.handler = handler

    def handle(self, payload: ImageUploadPayload):
        image_inputs, class_names, dummy = self.handler.process(payload)

        self.repository.send_embeddings_pair(image_inputs, dummy, class_names)

        return {"ok": True, "class": class_names}
