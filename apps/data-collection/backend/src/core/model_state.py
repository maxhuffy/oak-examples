class ModelState:
    """Holds model-related metadata and class definitions."""

    def __init__(self):
        self.__current_classes: list[str] = []
        self.__conf_threshold = 0.1

    def update_classes(self, new_classes: list[str]):
        self.__current_classes = list(new_classes)

    def update_threshold(self, value: float):
        self.__conf_threshold = max(0.0, min(1.0, value))

    def get_classes(self) -> list[str]:
        return list(self.__current_classes)

    def get_threshold(self) -> float:
        return self.__conf_threshold
