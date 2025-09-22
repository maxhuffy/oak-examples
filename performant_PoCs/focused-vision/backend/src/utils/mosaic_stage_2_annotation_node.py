from typing import List, Tuple
import depthai as dai

from depthai_nodes.message.gathered_data import GatheredData
from .mosaic_layout_node import _compute_mosaic_layout
from depthai_nodes.utils import AnnotationHelper

from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.message import Keypoints


class MosaicStage2AnnotationNode(dai.node.HostNode):
    """
    Merge detections from each crop into one message and REMAP their coordinates
    from crop-normalized -> mosaic-normalized.

    Input (via link_args): GatherData bundle
      - msg.reference_data : (for ts/seq)
      - msg.gathered       : List[ImgDetectionsExtended] OR List[Keypoints]

    Output:
      - self.out : Same message type as input (ImgDetectionsExtended or Keypoints),
                   with all items remapped to the mosaic canvas.
    """

    def __init__(self):
        super().__init__()
        self.mosaic_w = self.mosaic_h = None
        self.crop_w = self.crop_h = None

    def build(
            self,
            gathered_pair_out: dai.Node.Output,
            mosaic_size: Tuple[int, int],
            crop_size: Tuple[int, int],
    ) -> "MosaicStage2AnnotationNode":
        self.mosaic_w, self.mosaic_h = map(int, mosaic_size)
        self.crop_w, self.crop_h = map(int, crop_size)
        self.link_args(gathered_pair_out)
        return self

    @staticmethod
    def _letterbox(tile_w: int, tile_h: int, src_w: int, src_h: int):
        """Return (scale, off_x, off_y) in tile pixels for letterboxed fit."""
        s = min(tile_w / max(1, src_w), tile_h / max(1, src_h))
        x0 = (tile_w - int(src_w * s)) // 2
        y0 = (tile_h - int(src_h * s)) // 2
        return s, x0, y0

    def _remap_point_crop_to_mosaic(
            self, nx: float, ny: float, tile_x: int, tile_y: int, tile_w: int, tile_h: int
    ) -> Tuple[float, float]:
        """Remap a normalized crop-space point -> mosaic-normalized point."""
        s, x0, y0 = self._letterbox(tile_w, tile_h, self.crop_w, self.crop_h)
        cx = nx * self.crop_w
        cy = ny * self.crop_h
        tx = x0 + cx * s
        ty = y0 + cy * s
        mx = (tile_x + tx) / self.mosaic_w
        my = (tile_y + ty) / self.mosaic_h
        return float(mx), float(my)

    def _remap_rotated_rect(
            self, det, tile_x: int, tile_y: int, tile_w: int, tile_h: int
    ) -> None:
        """
        Remap det.rotated_rect (center/size/angle) from crop-normalized to mosaic-normalized.
        Mutates `det` in place (safe: call on a copy).
        """
        rr = getattr(det, "rotated_rect", None)
        if rr is None or not hasattr(rr, "center") or not hasattr(rr, "size"):
            return

        cx = float(rr.center.x) * self.crop_w
        cy = float(rr.center.y) * self.crop_h
        w = float(rr.size.width) * self.crop_w
        h = float(rr.size.height) * self.crop_h

        s, x0, y0 = self._letterbox(tile_w, tile_h, self.crop_w, self.crop_h)
        tx = x0 + cx * s
        ty = y0 + cy * s
        tw = max(1e-6, w * s)
        th = max(1e-6, h * s)

        mx = (tile_x + tx) / self.mosaic_w
        my = (tile_y + ty) / self.mosaic_h
        mw = tw / self.mosaic_w
        mh = th / self.mosaic_h
        angle = float(getattr(rr, "angle", 0.0))

        try:
            det.rotated_rect = (mx, my, mw, mh, angle)
        except Exception:
            pass

    def _remap_keypoints_in_place(
            self, keypoints_list: List, tile_x: int, tile_y: int, tile_w: int, tile_h: int
    ) -> None:
        """Remap a list of keypoints (each has .x/.y normalized to crop) to mosaic-normalized."""
        if not keypoints_list:
            return
        s, x0, y0 = self._letterbox(tile_w, tile_h, self.crop_w, self.crop_h)
        for kp in keypoints_list:
            if not hasattr(kp, "x") or not hasattr(kp, "y"):
                continue
            kx_px = float(kp.x) * self.crop_w
            ky_px = float(kp.y) * self.crop_h
            kx_tile = x0 + kx_px * s
            ky_tile = y0 + ky_px * s
            kp.x = (tile_x + kx_tile) / self.mosaic_w
            kp.y = (tile_y + ky_tile) / self.mosaic_h

    def process(self, msg) -> None:
        if isinstance(msg, GatheredData):
            ref = msg.reference_data
            gathered = msg.gathered
        else:
            ref = None
            gathered = []

        if not gathered:
            empty = AnnotationHelper().build(
                timestamp=ref.getTimestamp(), sequence_num=ref.getSequenceNum()
            ) if ref else AnnotationHelper().build()
            self.out.send(empty)
            return

        layout = _compute_mosaic_layout(len(gathered))

        out = gathered[0]

        merged_dets: List = []
        merged_kps: List = []
        merged_edges: List[Tuple[int, int]] = []

        for i, crop_msg in enumerate(gathered):
            if i >= len(layout) or crop_msg is None:
                continue

            x_n, y_n, w_n, h_n = layout[i]
            tile_x = int(x_n * self.mosaic_w)
            tile_y = int(y_n * self.mosaic_h)
            tile_w = max(1, int(w_n * self.mosaic_w))
            tile_h = max(1, int(h_n * self.mosaic_h))

            if isinstance(crop_msg, ImgDetectionsExtended):
                dets = getattr(crop_msg, "detections", None)
                if not dets:
                    continue
                for det in dets:
                    try:
                        d = det.copy()
                    except Exception:
                        d = det
                    self._remap_rotated_rect(d, tile_x, tile_y, tile_w, tile_h)
                    kps = getattr(d, "keypoints", None)
                    if kps:
                        self._remap_keypoints_in_place(kps, tile_x, tile_y, tile_w, tile_h)
                    merged_dets.append(d)

            elif isinstance(crop_msg, Keypoints):
                kps = getattr(crop_msg, "keypoints", None)
                if not kps:
                    continue

                idx_offset = len(merged_kps)

                for kp in kps:
                    try:
                        k = kp.copy()
                    except Exception:
                        k = kp
                    self._remap_keypoints_in_place([k], tile_x, tile_y, tile_w, tile_h)
                    merged_kps.append(k)

                crop_edges = list(getattr(crop_msg, "edges", [])) or []
                for a, b in crop_edges:
                    merged_edges.append((idx_offset + int(a), idx_offset + int(b)))

        if isinstance(out, ImgDetectionsExtended):
            out.detections = merged_dets
        elif isinstance(out, Keypoints):
            out.keypoints = merged_kps
            out.edges = merged_edges

        self.out.send(out)
