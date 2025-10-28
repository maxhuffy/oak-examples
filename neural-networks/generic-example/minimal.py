import cv2
import depthai as dai
import numpy as np
import time
import os

os.environ["DEPTHAI_HUB_API_KEY"] = "tapi.o0M6yGp4pI_JSUKjdBTzKg.DptpxXC_19X2jjv2vtU5qeKibM4awLXYcANDlOqzxFMHS48rp1WHV1XCsKkWWBnEhJsaDAPNeJI5f1OQkv9FHQ"

device = dai.Device(dai.DeviceInfo("10.12.143.173"))

# Create pipeline
with dai.Pipeline(device) as pipeline:
    cameraNode = pipeline.create(dai.node.Camera).build()
    # Longer form - useful in case of a local NNArchive
    modelDescription = dai.NNModelDescription("luxonis-ml-team/rf-detr:working-simplified", platform=pipeline.getDefaultDevice().getPlatformAsString())
    archive = dai.NNArchive(dai.getModelFromZoo(modelDescription))
    neuralNetwork = pipeline.create(dai.node.NeuralNetwork).build(cameraNode, archive)
    # neuralNetwork = pipeline.create(dai.node.NeuralNetwork).build(cameraNode, dai.NNModelDescription("yolov6-nano"))

    qNNData = neuralNetwork.out.createOutputQueue()

    pipeline.start()


    while pipeline.isRunning():
        inNNData: dai.NNData = qNNData.get()
        tensor = inNNData.getFirstTensor()
        assert(isinstance(tensor, np.ndarray))
        print(f"Received NN data: {tensor.shape}")