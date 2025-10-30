from .base_service import BaseService
from ..core.io import base64_to_cv2_image


class BBoxPromptService(BaseService):
    """
    Handles bounding box prompts from the frontend.

    Decodes an image (or retrieves the last cached frame) and
    converts a normalized or pixel-based bounding box to absolute
    pixel coordinates for further processing.
    """

    def __init__(self, visualizer, frame_cache):
        super().__init__(visualizer, "BBox Prompt Service")
        self.frame_cache = frame_cache

    def handle(self, payload: dict | None = None):
        image = base64_to_cv2_image(payload["data"]) if payload.get("data") else None
        if image is None:
            image = self.frame_cache.get_last_frame()
            if image is None:
                return {"ok": False, "reason": "no_image"}
        if image is None:
            return {"ok": False, "reason": "decode_failed"}

        bbox = payload.get("bbox", {})
        bx = float(bbox.get("x", 0.0))
        by = float(bbox.get("y", 0.0))
        bw = float(bbox.get("width", 0.0))
        bh = float(bbox.get("height", 0.0))

        H, W = image.shape[:2]
        is_pixel = payload.get("bboxType", "normalized") == "pixel"
        if is_pixel:
            x0 = int(round(bx))
            y0 = int(round(by))
            x1 = int(round(bx + bw))
            y1 = int(round(by + bh))
        else:
            x0 = int(round(bx * W))
            y0 = int(round(by * H))
            x1 = int(round((bx + bw) * W))
            y1 = int(round((by + bh) * H))

        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return {"ok": True}
