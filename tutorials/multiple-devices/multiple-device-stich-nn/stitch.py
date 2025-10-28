import cv2
import depthai as dai
from stitching import Stitcher
from stitching.images import Images

class VideoStitcher(Stitcher):

    def initialize_stitcher(self, **kwargs):
        super().initialize_stitcher(**kwargs)
        self.cameras = None
        self.cameras_registered = False

    def unregister_cameras(self):
        self.cameras_registered = False
        return
		
    def stitch(self, images, feature_masks=[]):
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )

        if not self.cameras_registered:
            imgs = self.resize_medium_resolution()
            features = self.find_features(imgs, feature_masks)
            matches = self.match_features(features)
            imgs, features, matches = self.subset(imgs, features, matches)
            cameras = self.estimate_camera_parameters(features, matches)
            cameras = self.refine_camera_parameters(features, matches, cameras)
            cameras = self.perform_wave_correction(cameras)
            self.estimate_scale(cameras)
            self.cameras = cameras
            self.cameras_registered = True

        imgs = self.resize_low_resolution()
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, self.cameras)
        self.prepare_cropper(imgs, masks, corners, sizes)
        imgs, masks, corners, sizes = self.crop_low_resolution(
            imgs, masks, corners, sizes
        )
        self.estimate_exposure_errors(corners, imgs, masks)
        seam_masks = self.find_seam_masks(imgs, corners, masks)

        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, self.cameras)
        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )
        self.set_masks(masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        return self.create_final_panorama()

class Stitch(dai.node.ThreadedHostNode):
    def __init__(self, nr_inputs: int) -> None:
        super().__init__()
        self.inputs = []
        for _ in range(0, nr_inputs):
            self.inputs.append(self.createInput())
        self.out = self.createOutput()
        self.stitcher = VideoStitcher()

    def recalculate_homography(self):
        self.stitcher.unregister_cameras()
        return

    def run(self):
        while self.isRunning():
            """TODO add description"""
            images = []
            for input in self.inputs:
                input_frame = input.get()  # get the frame from input
                assert isinstance(input_frame, dai.ImgFrame)  # assert that it has correct type
                images.append(input_frame.getCvFrame())

            if len(images) < 2:
                print(f"There should be at lest 2 frames for stitching.")
                break
    
            try:
                stitched = self.stitcher.stitch(images)
            except Exception as e:
                print(f"Failed stitching because: {e}")
                stitched = cv2.hconcat(images)

            img_frame = dai.ImgFrame()
            stitched = cv2.resize(stitched, (512,288))
            img_frame.setCvFrame(stitched, dai.ImgFrame.Type.BGR888p)
            self.out.send(img_frame)