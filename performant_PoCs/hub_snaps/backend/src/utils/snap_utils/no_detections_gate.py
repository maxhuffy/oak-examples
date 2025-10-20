from __future__ import annotations


class NoDetectionsGate:
    def __init__(self) -> None:
        self.enabled: bool = False
        self.in_run: bool = False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.in_run = False

    def on_frame(self, num_detections: int) -> bool:
        if not self.enabled:
            self.in_run = False
            return False
        if num_detections == 0:
            if not self.in_run:
                self.in_run = True
                return True
            return False
        if self.in_run:
            self.in_run = False
        return False
