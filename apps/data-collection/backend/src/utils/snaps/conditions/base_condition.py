from abc import ABC, abstractmethod


class ConditionBase(ABC):
    """Interface for all snap trigger conditions."""

    def __init__(self, key: str, name: str):
        self.key = key
        self.name = name

    @abstractmethod
    def should_trigger(self, cond_man, **kwargs) -> bool:
        """Return True if this condition should fire."""
        pass

    def handle(self, **kwargs) -> dict:
        return {"reason": f"{self.key}_snap"}
