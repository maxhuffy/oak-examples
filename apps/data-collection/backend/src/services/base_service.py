from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from services.service_name import ServiceName


PayloadT = TypeVar("PayloadT", bound=dict)


class BaseService(ABC, Generic[PayloadT]):
    """
    Abstract base class for all backend services.
    Defines a single consistent interface for handling typed payloads.
    """

    NAME: ServiceName  # must be defined in subclasses

    def __init__(self):
        self.__name = self.NAME

    @abstractmethod
    def handle(self, payload: PayloadT) -> dict[str, any]:
        """Execute service logic and return a JSON-serializable response."""
        pass

    def get_name(self) -> ServiceName:
        return self.__name
