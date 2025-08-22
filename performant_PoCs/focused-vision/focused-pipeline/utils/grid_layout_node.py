import math
from typing import Dict, List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np

GROUP_STRIDE = 1000

def _layout_rects(n: int) -> List[Tuple[float, float, float, float]]:
    R: List[Tuple[float, float, float, float]] = []
    def row(cols: int, y0: float, h: float, count: Optional[int] = None):
        if count is None: count = cols
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
    def __init__(self):
        super().__init__()
        self.crops_input = self.createInput()
        self.num_configs_input = self.createInput()
        self.output = self.createOutput()

        self._target_w = 1920
        self._target_h = 1080
        self.frame_type = dai.ImgFrame.Type.BGR888p

        self._expected: Dict[int, int] = {}
        self._frames: Dict[int, List[dai.ImgFrame]] = {}

    def build(self, crops_input: dai.Node.Output, num_configs_input: dai.Node.Output, target_size: Tuple[int, int]) -> "GridLayoutNode":
        self._target_w, self._target_h = map(int, target_size)
        print(f"[GridLayoutNode] target={self._target_w}x{self._target_h}, GROUP_STRIDE={GROUP_STRIDE}")
        self.link_args(crops_input, num_configs_input)
        return self

    def process(self, crops_msg, count_msg):
        # This is how you access the gathered data
        print(crops_msg.gathered)
        if isinstance(count_msg, dai.Buffer):
            gid = int(count_msg.getSequenceNum())
            N = len(bytearray(count_msg.getData()))
            self._expected[gid] = N
            self._frames.setdefault(gid, [])
            print(f"[GridLayoutNode] gid={gid} COUNT={N}")
            self._try_emit(gid)

        if isinstance(crops_msg, dai.ImgFrame):
            seq = int(crops_msg.getSequenceNum())
            gid = (seq // GROUP_STRIDE) * GROUP_STRIDE
            self._frames.setdefault(gid, []).append(crops_msg)
            got = len(self._frames[gid])
            exp = self._expected.get(gid, -1)
            print(f"[GridLayoutNode] gid={gid} CROP {got}/{exp if exp>=0 else '?'} (seq={seq})")
            self._try_emit(gid)

    def _try_emit(self, gid: int):
        exp = self._expected.get(gid)
        frs = self._frames.get(gid)
        if exp is None:
            return

        if exp == 0:
            out_img = np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)
            out = dai.ImgFrame()
            out.setType(self.frame_type)
            out.setWidth(self._target_w)
            out.setHeight(self._target_h)
            out.setData(out_img.tobytes())
            out.setSequenceNum(gid)
            self.output.send(out)
            del self._expected[gid]
            self._frames.pop(gid, None)
            print(f"[GridLayoutNode] gid={gid} EMIT blank (0 tiles)")
            return

        if frs is None or len(frs) < exp:
            return

        cv_frames = [fr.getCvFrame() for fr in frs[:exp]]
        for i, fr_img in enumerate(cv_frames):
            if fr_img.ndim == 2:
                cv_frames[i] = cv2.cvtColor(fr_img, cv2.COLOR_GRAY2BGR)
            elif fr_img.ndim == 3 and fr_img.shape[2] == 4:
                cv_frames[i] = cv2.cvtColor(fr_img, cv2.COLOR_RGBA2BGR)

        layout = _layout_rects(len(cv_frames))
        out_img = np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)

        for (x, y, w, h), img in zip(layout, cv_frames):
            W = max(1, int(w * self._target_w))
            H = max(1, int(h * self._target_h))
            X = int(x * self._target_w)
            Y = int(y * self._target_h)
            tile = _fit_letterbox(img, W, H)
            out_img[Y:Y+H, X:X+W] = tile

        out = dai.ImgFrame()
        out.setType(self.frame_type)
        out.setWidth(self._target_w)
        out.setHeight(self._target_h)
        out.setData(out_img.tobytes())
        try:
            out.setTimestamp(frs[-1].getTimestamp())
        except Exception:
            pass
        out.setSequenceNum(gid)

        self.output.send(out)
        print(f"[GridLayoutNode] gid={gid} EMIT mosaic with {len(cv_frames)} tiles")

        del self._expected[gid]
        del self._frames[gid]
