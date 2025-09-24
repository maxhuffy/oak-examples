import depthai as dai
import numpy as np
import traceback
import json
import time
from depthai_nodes.utils import AnnotationHelper
from collections import deque

from .PointCloudMeasurement import PointCloudMeasurement

IMG_WIDTH, IMG_HEIGHT = 640, 400

BOX_EDGES = (
    (0,1),(1,2),(2,3),(3,0),  # bottom face
    (4,5),(5,6),(6,7),(7,4),  # top face
    (0,4),(1,5),(2,6),(3,7)   # verticals
)

class MeasurementNode(dai.node.ThreadedHostNode):
    """
    Node for point cloud volume and dimensions measurement

    Inputs:
      - in_pcl            : PointCloudData (RGBD PCL)
      - in_enable_measure : Buffer, holds measuring mode value (0 - disabled, 1 - enabled)

    Outputs:
      - out_result : Buffer(JSON)  -> {"dims":[L,W,H], "vol": V}
      - out_ann    : Box outline annotation 
    """
    MODE_NOMEASURE = 0
    MODE_MEASURE   = 1
    MODE_PLANE     = 2  

    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)

        self.in_pcl = self.createInput()
        self.in_enable_measure = self.createInput()

        self.in_imu = self.createInput()

        self.out_result = self.createOutput()
        self.out_ann    = self.createOutput()

        self.pcl_measure = PointCloudMeasurement()
        self.have_plane = False
        self.measurement_mode = "obb"  # "obb" | "heightgrid"

        # intrinsics for box outline overlay 
        self.intrinsics = None
        self.imgW = IMG_WIDTH
        self.imgH = IMG_HEIGHT

        self._last_dims = deque(maxlen=30)
        self._last_vol = deque(maxlen=30)

        self.an_node = None 

    def build(self, pcl: dai.Node.Output, measure: dai.Node.Output, imu: dai.Node.Output):
        pcl.link(self.in_pcl)
        measure.link(self.in_enable_measure)
        imu.link(self.in_imu)
        return self
    
    def setIntrinsics(self, fx, fy, cx, cy, imgW, imgH):
        self.intrinsics = (float(fx), float(fy), float(cx), float(cy))
        self.imgW = int(imgW)
        self.imgH = int(imgH)
    
    def reset_plane(self):
        self.pcl_measure.clear_plane()
        self.have_plane = False
    
    def reset_measurements(self):
        self._last_dims.clear()
        self._last_vol.clear()

    def push_measurements(self, dims_cm, vol_cm3):
        if dims_cm.size == 3 and np.all(np.isfinite(dims_cm)):
            self._last_dims.append(tuple(float(x) for x in dims_cm))
        if isinstance(vol_cm3, (int, float)) and np.isfinite(vol_cm3):
            self._last_vol.append(float(vol_cm3))

    def _emit_median(self, pcl_msg, dims_cm, vol_cm3):
        self.push_measurements(dims_cm, vol_cm3)

        dims_out = None
        vol_out = None

        if self._last_dims:
            arr = np.asarray(self._last_dims, dtype=np.float64)  # shape (n,3)
            dims_out = np.median(arr, axis=0).round(2).tolist()
        
        if self._last_vol:
            v = np.asarray(self._last_vol, dtype=np.float64)
            vol_out = float(np.median(v).round(2))
        
        self._emit_result_min(pcl_msg, dims_out, vol_out)


    # ---------------- helpers for box outline annotations ---------------
    def _emit_result_min(self, pcl_msg: dai.PointCloudData, dims_cm, volume_cm3):
        payload = {
            "dims": [round(float(x), 2) for x in dims_cm] if dims_cm is not None else None,
            "vol":  round(float(volume_cm3), 2) if volume_cm3 is not None else None,
        }
        b = dai.Buffer()
        b.setData(json.dumps(payload).encode("utf-8"))
        b.setTimestamp(pcl_msg.getTimestamp())
        b.setSequenceNum(pcl_msg.getSequenceNum())
        self.out_result.send(b)

    def _proj_cam_to_rgb_norm(self, pts_cam):
        if not self.intrinsics:
            return [None] * len(pts_cam)
        fx, fy, cx, cy = self.intrinsics
        out = []
        for x, y, z in np.asarray(pts_cam, float):
            if z <= 0:
                out.append(None); continue
            u = fx * (x / z) + cx
            v = fy * (y / z) + cy
            out.append((u / self.imgW, v / self.imgH))
        return out
    
    def _emit_overlay(self,
                    pcl_msg: dai.PointCloudData,
                    pts3d: np.ndarray,
                    edges: np.ndarray | None = None,
                    color=(0.0, 1.0, 0.0, 1.0),
                    thickness: int = 3) -> None:
        """
        Draw a wireframe overlay.

        - If 'edges' is provided: draws segments defined by (i,j) pairs over pts3d (OBB case).
        - If 'edges' is None and len(pts3d) == 8: assumes a box and uses _BOX_EDGES (height-grid case).
        """
        helper = AnnotationHelper()

        if pts3d is None or len(pts3d) == 0:
            self.out_ann.send(helper.build(pcl_msg.getTimestamp(), pcl_msg.getSequenceNum()))
            return

        segs = edges
        if segs is None:
            if len(pts3d) == 8:
                segs = BOX_EDGES
            else:
                self.out_ann.send(helper.build(pcl_msg.getTimestamp(), pcl_msg.getSequenceNum()))
                return

        pts2d = self._proj_cam_to_rgb_norm(pts3d)

        for i, j in segs:
            pi, pj = pts2d[i], pts2d[j]
            if pi and pj:  
                helper.draw_line(pi, pj, color=color, thickness=thickness)

        self.out_ann.send(helper.build(pcl_msg.getTimestamp(), pcl_msg.getSequenceNum()))

    def run(self):
        # small pairing buffers keyed by (seq, ts_ns)
        pcl_buf  = {}
        mode_buf = {}

        MAX_UNMATCHED = 30  
        
        def key_for(msg):
            return msg.getSequenceNum()

        while self.isRunning():
            
            # Note: temporary sync logic for mode and pcl messgaes (to do)
            while True:
                mb = self.in_enable_measure.tryGet()
                if not mb:
                    break
                mode_buf[key_for(mb)] = mb
                if len(mode_buf) > MAX_UNMATCHED:
                    # drop oldest
                    first_k = next(iter(mode_buf))
                    mode_buf.pop(first_k, None)

            # 2) drain PCL queue
            while True:
                imu_msg = self.in_imu.tryGet()
                if imu_msg is None:
                    time.sleep(0.001)
                    continue 
                pc = self.in_pcl.tryGet()
                if not pc:
                    break
                pcl_buf[key_for(pc)] = pc
                if len(pcl_buf) > MAX_UNMATCHED:
                    first_k = next(iter(pcl_buf))
                    pcl_buf.pop(first_k, None)

            # 3) match by (seq, ts_ns)
            common = set(pcl_buf.keys()) & set(mode_buf.keys())
            if not common:
                #print('not matched')
                time.sleep(0.005)
                continue

            # process matched pairs in order
            for k in sorted(common):

                pcl_msg  = pcl_buf.pop(k)
                mode_msg = mode_buf.pop(k)

                assert isinstance(pcl_msg, dai.PointCloudData)

                mode_val = int(bytes(mode_msg.getData())[0]) if mode_msg else self.MODE_NOMEASURE
   
                print("MODE:", mode_val)

                if mode_val == self.MODE_PLANE:
                    try:
                        points, colors = pcl_msg.getPointsRGB()
                    except Exception as e:
                        print("MeasurementNode: getPointsRGB() failed:", e)
                        #self._emit_result_min(pcl_msg, None, None)
                        continue

                    if points is None:
                        print("AnnotationNode: Empty PCL points array.")
                        #self._emit_result_min(pcl_msg, None, None)
                        continue

                    bgr = colors[:, :3]
                    rgb = bgr[:, ::-1].astype(np.float64) / 255.0

                    for pkt in imu_msg.packets:
                        if hasattr(pkt, "acceleroMeter") and pkt.acceleroMeter is not None:
                            acc = pkt.acceleroMeter  # IMUReportAccelerometer
                            g_imu = np.array([acc.x, acc.y, acc.z], dtype=np.float64)
                    self.pcl_measure.set_point_cloud_plane(points)
                    plane, _, ok = self.pcl_measure.fit_plane(g_imu)
                    if ok:
                        self.pcl_measure.plane_eq = plane
                        self.have_plane = True
                        self.an_node.requestPlaneCapture(False)
                        print("Plane set!")
                    else:
                        self.have_plane = False
                        print("Plane fit failed.")
                        time.sleep(0.005)
                        self.an_node.requestPlaneCapture(True)
                    continue

                if mode_val != self.MODE_MEASURE:
                    #self.reset_measurements()
                    continue

                try:
                    points, colors = pcl_msg.getPointsRGB()
                except Exception as e:
                    print("MeasurementNode: getPointsRGB() failed:", e)
                    self._emit_result_min(pcl_msg, None, None)
                    continue

                if points is None:
                    print("AnnotationNode: Empty PCL points array.")
                    continue

                bgr = colors[:, :3]
                rgb = bgr[:, ::-1].astype(np.float64) / 255.0

                if self.measurement_mode == "obb":
                    self.pcl_measure.set_point_cloud(points, rgb)
                    self.pcl_measure.get_measurement_OBB()

                    self._emit_overlay(
                        pcl_msg,
                        self.pcl_measure.obb_pts,
                        self.pcl_measure.obb_edges
                    )

                    self._emit_median(pcl_msg, self.pcl_measure.dimensions, self.pcl_measure.volume)

                elif self.measurement_mode == "heightgrid":
                        
                    if not self.have_plane or self.pcl_measure.plane_eq is None:
                        self.an_node.requestPlaneCapture(True)
                        self._emit_result_min(pcl_msg, None, None)
                        continue

                    for pkt in imu_msg.packets:
                        if hasattr(pkt, "acceleroMeter") and pkt.acceleroMeter is not None:
                            acc = pkt.acceleroMeter  # IMUReportAccelerometer
                            g_imu = np.array([acc.x, acc.y, acc.z], dtype=np.float64)

                    if not self.pcl_measure.is_ground_plane(g_imu, self.pcl_measure.plane_eq):
                        self.an_node.requestPlaneCapture(True)
                        self.have_plane = False
                        continue

                    self.pcl_measure.set_point_cloud(points, rgb)
                    self.pcl_measure.get_measurement_groundHG()
                    self._emit_overlay(
                        pcl_msg,
                        self.pcl_measure.corners3d  
                    )
                        
                    self._emit_median(pcl_msg, self.pcl_measure.dimensions, self.pcl_measure.volume)
                
                 