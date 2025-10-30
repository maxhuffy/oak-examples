import abc


class BaseService(abc.ABC):
    """
    Minimal abstract base for DepthAI backend services.
    Only defines common interface + registration logic.
    """

    def __init__(self, visualizer, name: str):
        self.visualizer = visualizer
        self.name = name

    @abc.abstractmethod
    def handle(self, payload: dict | None = None):
        """Process request from frontend."""
        raise NotImplementedError

    def register(self):
        """Register service with the DepthAI visualizer."""
        self.visualizer.registerService(self.name, self.handle)
