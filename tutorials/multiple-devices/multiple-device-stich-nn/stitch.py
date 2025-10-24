import cv2
import depthai as dai
import imutils
import numpy as np

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

    def run(self):
        stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
        non_stitched_frames_nr = 0
        prev_stitched = None
        while self.isRunning():
            """TODO add description"""

            input_frame1 = self.input_frame1.get()
            input_frame2 = self.input_frame2.get()

            assert isinstance(input_frame1, dai.ImgFrame)
            assert isinstance(input_frame2, dai.ImgFrame)

            images = [input_frame1.getCvFrame(), input_frame2.getCvFrame()]
            
            try:
                (status, stitched) = stitcher.stitch(images)
            except:
                print(f"Error stitching")
                continue
            # if the status is '0', then OpenCV successfully performed image
            # stitching
            if status == 0:
                prev_stitched = stitched
                non_stitched_frames_nr = 0
            else:
                print(f"Error stitching because: {self.stitch_err_msg[status]}")
                non_stitched_frames_nr += 1
                if non_stitched_frames_nr > 10 or self.first_loop:
                    stitched = cv2.hconcat(images)
                    self.first_loop = False
                else:
                    stitched = prev_stitched

            img_frame = dai.ImgFrame()
            stitched = cv2.resize(stitched, (512,288))
            img_frame.setCvFrame(stitched, dai.ImgFrame.Type.BGR888p)
            self.out.send(img_frame)
            