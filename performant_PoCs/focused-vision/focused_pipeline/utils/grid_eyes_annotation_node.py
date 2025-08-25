from typing import List, Tuple
import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.utils import AnnotationHelper
from .grid_layout_node import _layout_rects  # reuse the exact layout


class GridEyesAnnotationNode(dai.node.HostNode):
    """
    Draw eye squares for each crop onto a single mosaic-sized annotation.

    Input (via link_args): GatherData bundle
        - msg.reference_data : stage-1 dets (ImgDetectionsExtended)  [used for ts/seq only]
        - msg.gathered       : List[ImgDetectionsExtended] from stage-2 on crops
    Output:
        - self.out : annotation buffer, stamped with reference ts/seq
    """

    def __init__(self):
        super().__init__()

        self.mosaic_w = self.mosaic_h = None
        self.crop_w = self.crop_h = None
        self.EYE_SCALE = 0.35

    def build(
            self,
            gathered_pair_out: dai.Node.Output,
            mosaic_size: Tuple[int, int],
            crop_size: Tuple[int, int],
            eye_scale: float = 0.35,
    ) -> "GridEyesAnnotationNode":
        self.mosaic_w, self.mosaic_h = map(int, mosaic_size)
        self.crop_w, self.crop_h = map(int, crop_size)
        self.EYE_SCALE = float(eye_scale)
        self.link_args(gathered_pair_out)
        return self

    @staticmethod
    def _tile_letterbox_params(tile_w: int, tile_h: int, crop_w: int, crop_h: int):
        """
        Mirror utils/grid_layout_node._fit_letterbox:
        scale to fit, then center inside tile.
        Returns (s, x0, y0) where:
          - s  : scale factor
          - x0 : left offset inside tile
          - y0 : top offset inside tile
        """
        s = min(tile_w / max(1, crop_w), tile_h / max(1, crop_h))
        x0 = (tile_w - int(crop_w * s)) // 2
        y0 = (tile_h - int(crop_h * s)) // 2
        return s, x0, y0

    def _draw_eye_squares(
            self,
            ann: AnnotationHelper,
            face_w_norm: float,
            face_h_norm: float,
            kps: List[Tuple[float, float]],  # [(kx, ky) ...] in crop-normalized 0..1
            tile_x: int, tile_y: int, tile_w: int, tile_h: int,
    ):
        # Letterbox params for this tile
        s, x0, y0 = self._tile_letterbox_params(tile_w, tile_h, self.crop_w, self.crop_h)

        # Face size in mosaic pixels (crop-normalized -> crop px -> scaled into tile)
        face_w_px = face_w_norm * self.crop_w * s
        face_h_px = face_h_norm * self.crop_h * s
        side_px = max(6, int(min(face_w_px, face_h_px) * self.EYE_SCALE))

        half_w_n = (side_px / self.mosaic_w) / 2.0
        half_h_n = (side_px / self.mosaic_h) / 2.0

        # Draw for first two kps (left/right eyes)
        for kx, ky in kps[:2]:
            # crop-normalized -> crop px
            cx = kx * self.crop_w
            cy = ky * self.crop_h
            # crop px -> tile px with letterbox
            tx = x0 + int(cx * s)
            ty = y0 + int(cy * s)
            # tile px -> mosaic-normalized
            mx = (tile_x + tx) / self.mosaic_w
            my = (tile_y + ty) / self.mosaic_h
            ann.draw_rectangle(
                [mx - half_w_n, my - half_h_n],
                [mx + half_w_n, my + half_h_n]
            )

    def process(self, msg) -> None:
        ref = getattr(msg, "reference_data", None)
        if not isinstance(ref, ImgDetectionsExtended):
            return

        # Stage-2 results per crop (same order as crops/layout)
        gathered = getattr(msg, "gathered", []) or []

        ann = AnnotationHelper()

        # Decide tile layout from number of stage-2 results we have
        layout = _layout_rects(len(gathered))

        for i, crop_msg in enumerate(gathered):
            if i >= len(layout):
                break
            if not isinstance(crop_msg, ImgDetectionsExtended):
                continue
            if not crop_msg.detections:
                continue

            # Tile rectangle in mosaic pixels
            x, y, w, h = layout[i]
            tile_x = int(x * self.mosaic_w)
            tile_y = int(y * self.mosaic_h)
            tile_w = max(1, int(w * self.mosaic_w))
            tile_h = max(1, int(h * self.mosaic_h))

            # Use STAGE-2 crops face size for square sizing
            det2 = crop_msg.detections[0]  # expected 1 per crop
            face_w_norm = det2.rotated_rect.size.width
            face_h_norm = det2.rotated_rect.size.height

            # Keypoints in crop-normalized coords 0..1
            kps = [(kp.x, kp.y) for kp in det2.keypoints[:2]]

            # Draw two eye squares onto the mosaic canvas
            self._draw_eye_squares(
                ann, face_w_norm, face_h_norm, kps,
                tile_x, tile_y, tile_w, tile_h
            )

        out = ann.build(timestamp=ref.getTimestamp(), sequence_num=ref.getSequenceNum())
        self.out.send(out)
