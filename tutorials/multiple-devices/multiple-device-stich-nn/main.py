#!/usr/bin/env python3

from depthai_nodes.node import ParsingNeuralNetwork
from stitch import Stitch
import contextlib
import depthai as dai


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

    stitched = pipeline.create(Stitch)
    outputs[0].link(stitched.input_frame1)
    outputs[1].link(stitched.input_frame2)

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(stitched.out, "luxonis/yolov6-nano:r2-coco-512x288")

    visualizer.addTopic("Background blur", stitched.out)
    visualizer.addTopic("NN detections", nn_with_parser.out)

    for p in pipelines:
        p.start()
    visualizer.registerPipeline(pipelines[0])
    
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
