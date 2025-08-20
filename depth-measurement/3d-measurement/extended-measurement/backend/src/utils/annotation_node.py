import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

Point = Tuple[float, float]

class AnnotationNode(dai.node.ThreadedHostNode):
    """ Interactive annotation / selection node

    Inputs:
      - in_detections: ImgDetectionsExtended produced by NN parser (includes masks)
      - in_rgb_stream: RGB ImgFrame 
      - in_depth_stream: depth ImgFrame (will be masked to create segmented PCL)

    Outputs:
      - out_det: filtered detections (optionally only the clicked instance)
      - out_segm: passthrough RGB (for overlay in the viewer)
      - out_segm_depth: masked depth (background set to 0) for PCL generation
    """
    def __init__(self, label_encoding: Dict[int, str] = {}) -> None:
        dai.node.ThreadedHostNode.__init__(self)

        self.in_detections = self.createInput()
        self.in_rgb_stream = self.createInput()
        self.in_depth_stream = self.createInput()

        self.out_det = self.createOutput()
        self.out_segm = self.createOutput()
        self.out_segm_depth = self.createOutput()

        self._label_encoding = label_encoding
        
        self._selection_point: Optional[Tuple[float, float]] = None  # normalized coords 
        self._keep_top_only: bool = True
        self._show_when_no_selection: bool = True  # False - show nothing until a click

        self._src_trans = None
        self._dst_trans = None

    def setLabelEncoding(self, label_encoding: Dict[int, str]) -> None:
        """Sets the label encoding.

        @param label_encoding: The label encoding with labels as keys and label names as
            values.
        @type label_encoding: Dict[int, str]
        """
        if not isinstance(label_encoding, Dict):
            raise ValueError("label_encoding must be a dictionary.")
        self._label_encoding = label_encoding

    def setSelectionPoint(self, nx: float, ny: float) -> None:
        nx = float(max(0.0, min(1.0, nx)))
        ny = float(max(0.0, min(1.0, ny)))
        self._selection_point = (nx, ny)
    
    def clearSelection(self) -> None:
        self._selection_point = None
    
    def setKeepTopOnly(self, keep_top: bool) -> None:
        """If True, keep only the highest-confidence detection under the click."""
        self._keep_top_only = bool(keep_top)
    
    def build(
        self,
        detections: dai.Node.Output,
        rgb_stream: Optional[dai.Node.Output] = None,
        depth_stream: Optional[dai.Node.Output] = None,
        label_encoding: Optional[Dict[int, str]] = None,
    ) -> "AnnotationNode":
        if label_encoding is not None:
            self.setLabelEncoding(label_encoding)

        detections.link(self.in_detections)

        if rgb_stream is not None:
            rgb_stream.link(self.in_rgb_stream)

        if depth_stream is not None:
            depth_stream.link(self.in_depth_stream)

        return self
    
    def point_in_convex_quad(self, px: float, py: float, corners, eps: float = 1e-9) -> bool:
        """
        corners: list[(x,y)] 
        Returns True if (px, py) lies inside/on a convex quad defined by 'corners'.
        """
        def cross(o, a, b):
            return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

        s = []
        for i in range(4):
            o = corners[i]
            a = corners[(i+1) % 4]
            s.append(cross(o, a, (px, py)))

        return all(v >= -eps for v in s) or all(v <= eps for v in s)
    
    def _contains_point(self, det, sx: float, sy: float) -> bool:
        """
        Check if normalized point (sx, sy) lies inside a detection.
        """
        # for rotated rectangles provided by parser
        rr = det.rotated_rect
        
        corners = self._get_rotated_rect_points(
        center=(rr.center.x, rr.center.y),
        size=(rr.size.width, rr.size.height),
        angle=rr.angle
        ) 

        return self.point_in_convex_quad(sx, sy, corners)
        # For axis-aligned normalized bbox
        #return (xmin <= sx <= xmax) and (ymin <= sy <= ymax)
    
    def _get_rotated_rect_points(
        self, center: Point, size: Tuple[float, float], angle: float
    ) -> List[Point]:
        """
        Return 4 corner points (normalized coords) of a rotated rect.
        """
        cx, cy = center
        width, height = size
        angle_rad = np.radians(angle)

        dx = width / 2
        dy = height / 2
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])

        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad),  np.cos(angle_rad)]], dtype=np.float32)
        rotated = corners @ R.T
        translated = rotated + np.array([cx, cy], dtype=np.float32)
        return translated.tolist()
    
    
    def _prepare_mask_transform(self, src_T: dai.ImgTransformation, dst_T: dai.ImgTransformation):
        """Precompute homography to warp NN-space masks into RGB-space.

        Args:
          src_T: ImgTransformation for the NN input space (sensor→NN)
          dst_T: ImgTransformation for the RGB image space (sensor→RGB)
        """
        A = np.array(src_T.getMatrix(), dtype=np.float32)
        B = np.array(dst_T.getMatrix(), dtype=np.float32)

        # NN -> RGB
        try:
            self._H_nn_to_rgb = B @ np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self._H_nn_to_rgb = B @ np.linalg.pinv(A)

        dw, dh = dst_T.getSize()
        self._dst_size = (int(dw), int(dh))

    def _mask_nn_to_rgb(self, mask_src_2d: np.ndarray) -> np.ndarray:
        """
        Warp a NN-space mask to RGB-space using the precomputed homography.
        """
        dst_w, dst_h = self._dst_size

        mask = mask_src_2d
        if mask.dtype not in (np.int16, np.float32, np.uint8, np.uint16):
            mask = mask.astype(np.int16, copy=False)

        return cv2.warpPerspective(
            mask, self._H_nn_to_rgb, (dst_w, dst_h),
            flags=cv2.INTER_NEAREST, borderValue=-1
        )
    
    def _clear_dets_and_mask(self, det) -> None:
        """
        Clears detections and masks
        """
        det.detections = []
        m = getattr(det, "masks", None)
        if m is not None and getattr(m, "size", 0):
            # -1 - background 
            det.masks = np.full_like(m, -1)

    def run(self) -> None:

        while self.isRunning():
        
            detections_message = self.in_detections.get()
            img_msg = self.in_rgb_stream.get()
            depth_msg = self.in_depth_stream.get()

            depth_map = depth_msg.getFrame()
            rgb = img_msg.getCvFrame()

            H, W = rgb.shape[:2] 

            if self._src_trans is None or self._dst_trans is None:
                self._src_trans = detections_message.getTransformation()
                self._dst_trans = img_msg.getTransformation()
                self._prepare_mask_transform(self._src_trans, self._dst_trans)      # To transform mask from NN space to rgb space 
        
            assert isinstance(detections_message, ImgDetectionsExtended)
            for detection in detections_message.detections:
                detection.label_name = self._label_encoding.get(detection.label, "unknown")

            # No click/selection
            if self._selection_point is None:
                if not self._show_when_no_selection:
                    self._clear_dets_and_mask(detections_message)
                self.out_det.send(detections_message)
                self.out_segm.send(img_msg)
                self.out_segm_depth.send(depth_msg)
                continue
            
            # Filter detections by click location
            sx, sy = self._selection_point
            orig_dets = detections_message.detections
            kept_idx = [i for i, d in enumerate(orig_dets) if self._contains_point(d, sx, sy)]

            # If more detections contain click pick the most confident only 
            if self._keep_top_only and kept_idx:
                best_i = max(kept_idx, key=lambda i: getattr(orig_dets[i], "confidence", 0.0))
                kept_idx = [best_i]

            # If the click falls in no detection clear all 
            if not kept_idx:
                self._clear_dets_and_mask(detections_message)
                self.out_det.send(detections_message)
                self.out_segm.send(img_msg)
                self.out_segm_depth.send(depth_msg)
                continue

            detections_message.detections = [orig_dets[i] for i in kept_idx]
            m = detections_message.masks
            h_mask, w_mask = m.shape[:2]

            # Note: quick fix for the selection of the mask, need to modify 
            px = int(np.clip(sx * w_mask, 0, w_mask - 1))
            py = int(np.clip(sy * h_mask, 0, h_mask - 1))
            selected_mid = int(m[py, px])

            if m is not None and selected_mid != -1:
                new_mask = np.full_like(m, -1)
                new_mask[m == selected_mid] = 0
                detections_message.masks = new_mask   
            else:
                self.out_det.send(detections_message)
                self.out_segm.send(img_msg)
                self.out_segm_depth.send(depth_msg)
                continue
            
            # Segment PCL using mask 
            # convert mask from nn to rgb space 
            mask_rgb = self._mask_nn_to_rgb(new_mask)
            keep = (mask_rgb >= 0)
            
            depth_u16 = np.ascontiguousarray(depth_map, dtype=np.uint16)
            depth_u16[~keep] = 0         # mask background to zero depth

            # Send masked depth for PCL generation 
            depthF = dai.ImgFrame()
            depthF.setType(dai.ImgFrame.Type.RAW16)
            depthF.setWidth(W); depthF.setHeight(H)
            depthF.setData(depth_u16.tobytes())                 
            depthF.setTimestamp(depth_msg.getTimestamp())
            depthF.setSequenceNum(depth_msg.getSequenceNum())
            depthF.setTransformation(depth_msg.getTransformation())  

            self.out_segm.send(img_msg)
            self.out_segm_depth.send(depthF)
            self.out_det.send(detections_message)

            
