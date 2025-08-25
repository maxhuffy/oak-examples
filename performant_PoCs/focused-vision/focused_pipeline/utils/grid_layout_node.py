import math
from typing import List, Tuple
import cv2
import numpy as np
import depthai as dai


def _layout_rects(n: int) -> List[Tuple[float, float, float, float]]:
    """
    Returns a list of (x, y, w, h) tiles in normalized [0..1] coords.
    Simple, deterministic layouts for 1..N items with up to 3 columns.
    """
    R: List[Tuple[float, float, float, float]] = []

    def row(cols: int, y0: float, h: float, count: int | None = None):
        if count is None:
            count = cols
        w = 1.0 / cols
        for i in range(count):
            R.append((i * w, y0, w, h))

    if n <= 1:
        return [(0.0, 0.0, 1.0, 1.0)]
    if n == 2:
        row(2, 0.0, 1.0, 2)
    elif n == 3:
        row(2, 0.0, 0.5, 2)
        R.append((0.0, 0.5, 1.0, 0.5))
    elif n == 4:
        row(2, 0.0, 0.5, 2)
        row(2, 0.5, 0.5, 2)
    elif n == 5:
        row(3, 0.0, 0.5, 3)
        row(2, 0.5, 0.5, 2)
    elif n == 6:
        row(3, 0.0, 0.5, 3)
        row(3, 0.5, 0.5, 3)
    else:
        rows = math.ceil(n / 3)
        h = 1.0 / rows
        left = n
        y = 0.0
        for _ in range(rows):
            c = min(3, left)
            row(c, y, h, c)
            left -= c
            y += h
    return R


def _fit_letterbox(img: np.ndarray, W: int, H: int) -> np.ndarray:
    """Resize with preserved aspect and center into WxH canvas."""
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    s = min(W / max(1, iw), H / max(1, ih))
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x0 = (W - nw) // 2
    y0 = (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _mosaic_from_frames(
    frames: List[dai.ImgFrame],
    out_w: int,
    out_h: int,
    frame_type: dai.ImgFrame.Type
) -> dai.ImgFrame:
    """
    Compose a mosaic from a list of ImgFrames (assumed BGR/BGRA/GRAY).
    Does not set timestamp/seq — caller should stamp from the GatherData reference.
    """
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    if frames:
        mats: List[np.ndarray] = []
        for fr in frames:
            img = fr.getCvFrame()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            mats.append(img)

        layout = _layout_rects(len(mats))
        for (x, y, w, h), img in zip(layout, mats):
            W = max(1, int(w * out_w))
            H = max(1, int(h * out_h))
            X = int(x * out_w)
            Y = int(y * out_h)
            tile = _fit_letterbox(img, W, H)
            canvas[Y:Y + H, X:X + W] = tile

    out = dai.ImgFrame()
    out.setType(frame_type)
    out.setWidth(out_w)
    out.setHeight(out_h)
    out.setData(canvas.tobytes())
    return out


class GridLayoutNode(dai.node.HostNode):
    """
    Builds a single mosaic image from a GatherData bundle of crop frames.

    Input:
      - GatherData.out (bundle)
        * msg.reference_data : the stage-1 reference message (for ts/seq)
        * msg.gathered       : List[dai.ImgFrame] crop frames for this logical frame

    Output:
      - self.output : dai.ImgFrame (mosaic), stamped with the reference ts/seq
    """
    def __init__(self):
        super().__init__()
        self.crops_input = self.createInput()
        self.output = self.createOutput()

        self._target_w = 1920
        self._target_h = 1080
        self.frame_type = None

    def build(self, crops_input: dai.Node.Output, target_size: Tuple[int, int], frame_type: dai.ImgFrame.Type) -> "GridLayoutNode":
        self._target_w, self._target_h = map(int, target_size)
        self.link_args(crops_input)
        self.frame_type = frame_type
        return self

    def process(self, msg):
        """
        Expect a GatherData bundle:
          - msg.reference_data for timestamp/sequence (authoritative)
          - msg.gathered as List[dai.ImgFrame] for tiles
        """
        # Safely read gathered crops
        crops: List[dai.ImgFrame] = getattr(msg, "gathered", []) or []

        # Build mosaic image
        mosaic = _mosaic_from_frames(crops, self._target_w, self._target_h, self.frame_type)

        self.output.send(mosaic)
