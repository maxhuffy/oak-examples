import cv2
import depthai as dai
import imutils
import numpy as np
from multiprocessing import Process, Queue
import traceback

class Stitch(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input_frame1 = self.createInput()
        self.input_frame2 = self.createInput()
        self.first_loop = True
        self.out = self.createOutput()

        self.stitch_err_msg = {cv2.STITCHER_ERR_NEED_MORE_IMGS: "stitcher does not have enough images",
                               cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL: "error estimating stitching homography",
                               cv2.STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL: "failed adjusting parameters"}
    
    def _stitch_process(self, q, stitcher, images):
        """Runs stitcher.stitch(images) in a separate process."""
        try:
            result = stitcher.stitch(images)  # (status, stitched)
            q.put(("ok", result))
        except Exception as e:
            q.put(("error", traceback.format_exc()))

    def stitch_with_timeout(self, stitcher, images, timeout=1):
        """
        Runs stitcher.stitch(images) with a hard timeout.
        Returns (status, stitched) if it finishes in time,
        otherwise returns (None, None) and kills the process.
        """
        q = Queue()
        p = Process(target=self._stitch_process, args=(q, stitcher, images))
        p.start()
        try:
            msg_type, payload = q.get(timeout=timeout)
        except:
            p.terminate()
            print("timeout")
            return None, None
            
        
        if msg_type == "ok":
            (status, stitched) = payload
            print("✅ Stitch completed successfully before timeout.")
            return status, stitched
        else:
            print("❌ Error while stitching:\n", payload)
            return None, None
        

    def run(self):
        stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
        non_stitched_frames_nr = 0
        prev_stitched = None
        self.first_loop = True
        while self.isRunning():
            """TODO add description"""

            input_frame1 = self.input_frame1.get()
            input_frame2 = self.input_frame2.get()

            assert isinstance(input_frame1, dai.ImgFrame)
            assert isinstance(input_frame2, dai.ImgFrame)

            images = [input_frame1.getCvFrame(), input_frame2.getCvFrame()]
    

            status, stitched = self.stitch_with_timeout(stitcher, images, timeout=1)

            if status is None:
                print("Stitching did not complete successfully.")
                continue

            if status == 0:
                prev_stitched = stitched
                non_stitched_frames_nr = 0
            else:
                print(f"Error stitching because: {self.stitch_err_msg[status]}")
                non_stitched_frames_nr += 1
                if non_stitched_frames_nr > 5 or self.first_loop:
                    stitched = cv2.hconcat(images)
                    self.first_loop = False
                else:
                    stitched = prev_stitched

            img_frame = dai.ImgFrame()
            stitched = cv2.resize(stitched, (512,288))
            img_frame.setCvFrame(stitched, dai.ImgFrame.Type.BGR888p)
            self.out.send(img_frame)
            