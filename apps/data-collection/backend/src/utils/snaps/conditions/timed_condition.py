from .base_condition import ConditionBase


class TimedCondition(ConditionBase):
    """
    Triggers snaps at fixed time intervals.

    Fires independently of detection data, based solely on the
    configured cooldown period in the condition manager.
    """

    def __init__(self, key="timed", name="Timed Snap"):
        super().__init__(key, name)

    def should_trigger(self, cond_manager, **_):
        return cond_manager.hit(self.key)
