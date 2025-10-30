from .base_condition import ConditionBase


class NoDetectionsCondition(ConditionBase):
    """
    Triggers when no objects are detected in the current frame.

    Activates only when the detections list is empty and the
    cooldown period for this condition has expired.
    """

    def __init__(self, key="noDetections", name="No Detections"):
        super().__init__(key, name)

    def should_trigger(self, cond_manager, det_data=None, **_):
        if det_data is None or len(getattr(det_data, "detections", [])) > 0:
            return False
        return cond_manager.hit(self.key)
