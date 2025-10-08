from typing import List, Optional

import depthai as dai
from depthai_nodes.node.utils import generate_script_content


def create_manip(pipeline: dai.Pipeline, frame_type: dai.ImgFrame.Type, width, height, mode: dai.ImageManipConfig.ResizeMode = dai.ImageManipConfig.ResizeMode.LETTERBOX):
    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(width * height * 3)
    manip.initialConfig.setOutputSize(width, height, mode=mode)
    manip.initialConfig.setFrameType(frame_type)
    return manip


def build_roi_cropper(
    pipeline: dai.Pipeline,
    *,
    preview_stream: dai.Node.Output,
    det_stream: dai.Node.Output,
    out_size: tuple[int, int],
    frame_type: dai.ImgFrame.Type,
    padding: float = 0.05,
    pool_size: int = 10,
    cfg_queue_size: int = 10,
) -> dai.Node.Output:
    """
    Build: Script(det_in+preview) -> ImageManip(crop+resize to out_size, frame_type) -> out
    Returns cropper.out (frames of size out_size)
    """
    resize_w, resize_h = out_size
    script = pipeline.create(dai.node.Script)
    script.setScript(generate_script_content(resize_width=resize_w, resize_height=resize_h, resize_mode="LETTERBOX", padding=padding))
    det_stream.link(script.inputs["det_in"])
    preview_stream.link(script.inputs["preview"])

    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(resize_w * resize_h * 3)
    manip.initialConfig.setOutputSize(resize_w, resize_h)
    manip.initialConfig.setFrameType(frame_type)
    manip.inputConfig.setMaxSize(cfg_queue_size)
    manip.inputImage.setMaxSize(cfg_queue_size)
    manip.setNumFramesPool(pool_size)
    manip.inputConfig.setWaitForMessage(True)

    script.outputs["manip_cfg"].link(manip.inputConfig)
    script.outputs["manip_img"].link(manip.inputImage)

    return manip.out


def create_rgb(
        pipeline: dai.Pipeline,
        high_res: tuple[int, int],
        low_res: tuple[int, int],
        fps: int,
        frame_type: dai.ImgFrame.Type = dai.ImgFrame.Type.BGR888i,
) -> tuple[dai.Node.Output, dai.Node.Output]:
    rgb = pipeline.create(dai.node.Camera).build()
    rgb.initialControl.setManualExposure(30_000, 110)
    if high_res[0] > 4000 or high_res[1] > 3000:
        high_res_out = rgb.requestOutput(size=high_res, type=frame_type, fps=fps)
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(*low_res, mode=dai.ImageManipConfig.ResizeMode.LETTERBOX)
        manip.setMaxOutputFrameSize(low_res[0] * low_res[1] * 3)
        high_res_out.link(manip.inputImage)
        low_res_out = manip.out
    else:
        high_res_out = rgb.requestOutput(size=high_res, type=frame_type, fps=fps)
        low_res_out = rgb.requestOutput(size=low_res, type=frame_type, fps=fps)
    return high_res_out, low_res_out
