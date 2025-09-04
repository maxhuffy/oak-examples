from typing import List, Tuple
import depthai as dai
from depthai_nodes.utils import AnnotationHelper
from .mosaic_layout_node import _compute_mosaic_layout
from depthai_nodes.message.gathered_data import GatheredData


class MosaicEyesAnnotationNode(dai.node.HostNode):
    """
    Draw boxes from each crop onto a single mosaic-sized annotation.

    Input (via link_args): GatherData bundle
        - msg.reference_data : stage-1 dets (used for ts/seq only)
        - msg.gathered       : List[...] from stage-2 on crops
          (expects each item to have `.detections` with d.xmin/ymin/xmax/ymax in 0..1 crop space)
    Output:
        - self.out : annotation buffer, stamped with reference ts/seq
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
    ) -> "MosaicEyesAnnotationNode":
        self.mosaic_w, self.mosaic_h = map(int, mosaic_size)
        self.crop_w, self.crop_h = map(int, crop_size)
        self.link_args(gathered_pair_out)
        return self

    @staticmethod
    def _letterbox(tile_w: int, tile_h: int, crop_w: int, crop_h: int):
        """
        Scale crop to fit inside tile and center (letterbox).
        Returns (s, x0, y0) where:
          s  : scale factor
          x0 : left offset inside tile
          y0 : top offset inside tile
        """
        s = min(tile_w / max(1, crop_w), tile_h / max(1, crop_h))
        x0 = (tile_w - int(crop_w * s)) // 2
        y0 = (tile_h - int(crop_h * s)) // 2
        return s, x0, y0

    def _draw_boxes_on_tile(
        self,
        ann: AnnotationHelper,
        boxes_crop_norm: List[Tuple[float, float, float, float]],
        tile_x: int, tile_y: int, tile_w: int, tile_h: int,
    ):
        """
        Map crop boxes → tile px (with letterbox) → mosaic-normalized, then draw.
        """
        s, x0, y0 = self._letterbox(tile_w, tile_h, self.crop_w, self.crop_h)

        for nx1, ny1, nx2, ny2 in boxes_crop_norm:
            # crop-normalized -> crop px
            cx1, cy1 = nx1 * self.crop_w, ny1 * self.crop_h
            cx2, cy2 = nx2 * self.crop_w, ny2 * self.crop_h
            # crop px -> tile px with letterbox
            tx1 = x0 + int(cx1 * s)
            ty1 = y0 + int(cy1 * s)
            tx2 = x0 + int(cx2 * s)
            ty2 = y0 + int(cy2 * s)
            # tile px -> mosaic-normalized
            mx1 = (tile_x + tx1) / self.mosaic_w
            my1 = (tile_y + ty1) / self.mosaic_h
            mx2 = (tile_x + tx2) / self.mosaic_w
            my2 = (tile_y + ty2) / self.mosaic_h
            ann.draw_rectangle([mx1, my1], [mx2, my2])

    def process(self, msg) -> None:
        if isinstance(msg, GatheredData):
            ref = msg.reference_data
            gathered = msg.gathered
        else:
            ref = None
            gathered = []

        if not gathered:
            out = AnnotationHelper().build(timestamp=ref.getTimestamp(), sequence_num=ref.getSequenceNum()) if ref else AnnotationHelper().build()
            self.out.send(out)
            return

        ann = AnnotationHelper()

        # Compute mosaic tiles
        layout = _compute_mosaic_layout(len(gathered))

        for i, crop_msg in enumerate(gathered):
            if i >= len(layout):
                break
            dets = getattr(crop_msg, "detections", None)
            if not dets:
                continue

            x, y, w, h = layout[i]
            tile_x = int(x * self.mosaic_w)
            tile_y = int(y * self.mosaic_h)
            tile_w = max(1, int(w * self.mosaic_w))
            tile_h = max(1, int(h * self.mosaic_h))

            boxes: List[Tuple[float, float, float, float]] = []
            for d in dets:
                boxes.append((float(d.xmin), float(d.ymin), float(d.xmax), float(d.ymax)))
            if not boxes:
                continue

            self._draw_boxes_on_tile(ann, boxes, tile_x, tile_y, tile_w, tile_h)

        if ref is not None:
            out = ann.build(timestamp=ref.getTimestamp(), sequence_num=ref.getSequenceNum())
        else:
            out = ann.build()
        self.out.send(out)
