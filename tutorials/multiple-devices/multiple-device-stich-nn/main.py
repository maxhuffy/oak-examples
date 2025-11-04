#!/usr/bin/env python3

from depthai_nodes.node import ParsingNeuralNetwork
from stitch import Stitch
import contextlib
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, TilesPatcher, Tiling

def createPipeline(pipeline):
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = camRgb.requestOutput((640, 400), dai.ImgFrame.Type.NV12 ,dai.ImgResizeMode.CROP, 20)
    return pipeline, output


with contextlib.ExitStack() as stack:
    visualizer = dai.RemoteConnection(httpPort=8082)
    deviceInfos = dai.Device.getAllAvailableDevices()
    print("=== Found devices: ", deviceInfos)
    outputs = []
    pipelines = []

    for deviceInfo in deviceInfos:
        pipeline = stack.enter_context(dai.Pipeline())
        device = pipeline.getDefaultDevice()
        
        print("===Connected to ", deviceInfo.getDeviceId())
        mxId = device.getDeviceId()
        cameras = device.getConnectedCameras()
        usbSpeed = device.getUsbSpeed()
        eepromData = device.readCalibration2().getEepromData()
        print("   >>> Device ID:", mxId)
        print("   >>> Num of cameras:", len(cameras))
        if eepromData.boardName != "":
            print("   >>> Board name:", eepromData.boardName)
        if eepromData.productName != "":
            print("   >>> Product name:", eepromData.productName)
        
        pipeline, output = createPipeline(pipeline)
        pipelines.append(pipeline)

        outputs.append(output)

    # Create threaded node pipeline with Stitch class, setting nr on inputs and output resolution 
    # set to NN input resolution
    stitch_pl = pipeline.create(Stitch, nr_inputs=len(outputs), output_resolution = (512,288))
    for i, output in enumerate(outputs):
        # Link each output of a camera to stitching inputs
        output.link(stitch_pl.inputs[i])
        # Do not block stream if image queue gets full - less delay in output detection stream
        stitch_pl.inputs[i].setBlocking(False)  

    # stitch_queue = stitch_pl.out.createOutputQueue()
    # stitched_img_frame: dai.ImgFrame = stitch_queue.tryGet()
    grid_size = (2, 1)

    tile_manager = pipeline.create(Tiling).build(
        img_output=stitch_pl.out,
        img_shape=(768,288),
        overlap=0.2,
        grid_size=grid_size,
        grid_matrix=None,
        global_detection=False,
        nn_shape=(512, 288),
    )

    interleaved_manip = pipeline.create(dai.node.ImageManip)
    interleaved_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    tile_manager.out.link(interleaved_manip.inputImage)
    
    # Run NN detection on stitched output 
    nn = pipeline.create(ParsingNeuralNetwork).build(tile_manager.out, "luxonis/yolov6-nano:r2-coco-512x288")

    nn.input.setMaxSize(len(tile_manager.tile_positions))

    patcher = pipeline.create(TilesPatcher).build(
        tile_manager=tile_manager, nn=nn.out, conf_thresh=0.1, iou_thresh=0.2
    )

    # Show stitched image on visualizer overlayed with nn detections
    visualizer.addTopic("Stitched", stitch_pl.out)
    # visualizer.addTopic("NN detections", nn.out)
    visualizer.addTopic("Patcher", patcher.out)

    # Start all of the pipelines
    for p in pipelines:
        p.start()

    # Register visualizer with the first pipeline
    visualizer.registerPipeline(pipelines[0])
    
    print("Press 'r' in visualizer to recalculate homography")
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
        if key == ord("r"):
            print("Got r key from the remote connection, recalculating homography")
            stitch_pl.recalculate_homography()

