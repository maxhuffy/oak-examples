import math
from typing import List, Tuple
import numpy as np
import cv2
import depthai as dai

def _layout_rects(n: int) -> List[Tuple[float, float, float, float]]:
    R: List[Tuple[float, float, float, float]] = []
    def row(cols: int, y0: float, h: float, count: int | None = None):
        if count is None:
            count = cols
        w = 1.0 / cols
        for i in range(count):
            R.append((i * w, y0, w, h))
    if n <= 1:
        return [(0, 0, 1, 1)]
    if n == 2:
        row(2, 0, 1, 2)
    elif n == 3:
        row(2, 0, 0.5, 2); R.append((0, 0.5, 1, 0.5))
    elif n == 4:
        row(2, 0, 0.5, 2); row(2, 0.5, 0.5, 2)
    elif n == 5:
        row(3, 0, 0.5, 3); row(2, 0.5, 0.5, 2)
    elif n == 6:
        row(3, 0, 0.5, 3); row(3, 0.5, 0.5, 3)
    else:
        rows = math.ceil(n / 3)
        h = 1.0 / rows
        left = n; y = 0.0
        for _ in range(rows):
            c = min(3, left)
            row(c, y, h, c)
            left -= c; y += h
    return R

def _fit_letterbox(img: np.ndarray, W: int, H: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(W / iw, H / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x0 = (W - nw) // 2; y0 = (H - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

class GridLayoutNode(dai.node.HostNode):
    """
    Consumes GatherData.out (detections as reference):
      - msg.reference_data: ImgDetectionsExtended (has .detections, .getSequenceNum(), .getTimestamp())
      - msg.gathered: list[dai.ImgFrame] (crops for the same timestamp)

    Emits one mosaic per detector frame. No count buffer needed.
    """
    def __init__(self):
        super().__init__()
        self.gather_crops = self.createInput()
        self.output = self.createOutput()

        self._target_w = 1920
        self._target_h = 1080
        self.frame_type = dai.ImgFrame.Type.BGR888p

    def build(self, gather_crops: dai.Node.Output, target_size: Tuple[int, int]) -> "GridLayoutNode":
        self._target_w, self._target_h = map(int, target_size)
        print(f"[GridLayoutNode] target={self._target_w}x{self._target_h}, input=GatherData.out (detections as reference)")
        self.link_args(gather_crops)
        return self

    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img

    def process(self, msg) -> None:
        # Expect a GatherData bundle: has 'reference_data' and 'gathered'
        if not hasattr(msg, "reference_data") or not hasattr(msg, "gathered"):
            return

        ref = msg.reference_data
        crops = msg.gathered or []

        # expected = number of detections (preferred)
        try:
            expected = len(ref.detections)
        except Exception:
            expected = len(crops)

        n = len(crops)
        use = min(expected, n)

        # metadata for output
        ts = None
        seq = None
        try:
            ts = ref.getTimestamp()
        except Exception:
            pass
        try:
            seq = int(ref.getSequenceNum())
        except Exception:
            pass

        if use == 0:
            # emit a blank to keep cadence
            out_img = np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)
            out = dai.ImgFrame()
            out.setType(self.frame_type)
            out.setWidth(self._target_w)
            out.setHeight(self._target_h)
            out.setData(out_img.tobytes())
            if ts is not None: out.setTimestamp(ts)
            if seq is not None: out.setSequenceNum(seq)
            self.output.send(out)
            print(f"[GridLayoutNode] seq={seq} EMIT blank (0 tiles)")
            return

        cv_frames: List[np.ndarray] = []
        for fr in crops[:use]:
            try:
                cv_frames.append(self._to_bgr(fr.getCvFrame()))
            except Exception:
                pass
        if not cv_frames:
            return

        layout = _layout_rects(len(cv_frames))
        out_img = np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)
        for (x, y, w, h), img in zip(layout, cv_frames):
            W = max(1, int(w * self._target_w))
            H = max(1, int(h * self._target_h))
            X = int(x * self._target_w)
            Y = int(y * self._target_h)
            out_img[Y:Y+H, X:X+W] = _fit_letterbox(img, W, H)

        out = dai.ImgFrame()
        out.setType(self.frame_type)
        out.setWidth(self._target_w)
        out.setHeight(self._target_h)
        out.setData(out_img.tobytes())
        if ts is not None:
            try: out.setTimestamp(ts)
            except Exception: pass
        if seq is not None:
            try: out.setSequenceNum(seq)
            except Exception: pass

        self.output.send(out)
        print(f"[GridLayoutNode] seq={seq} EMIT mosaic with {len(cv_frames)} tiles")
