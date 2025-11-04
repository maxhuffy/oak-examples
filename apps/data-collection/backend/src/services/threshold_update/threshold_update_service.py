from services.base_service import BaseService
from services.threshold_update.threshold_update_payload import ThresholdUpdatePayload
from core.controllers.nn_controller import YOLONNController
from services.service_name import ServiceName


class ThresholdUpdateService(BaseService[ThresholdUpdatePayload]):
    """Coordinates NN confidence threshold updates between handler, repository, and state."""

    NAME = ServiceName.THRESHOLD_UPDATE

    def __init__(
        self,
        repository: YOLONNController,
    ):
        super().__init__()
        self.repository = repository

    def handle(self, payload: ThresholdUpdatePayload) -> dict[str, any]:
        new_threshold = payload["threshold"]

        clamped = max(0.0, min(1.0, new_threshold))
        self.repository.set_confidence_threshold(clamped)

        return {"ok": True, "threshold": clamped}
