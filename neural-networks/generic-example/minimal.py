import depthai as dai
import numpy as np
import os
from dotenv import load_dotenv

from utils.arguments import initialize_argparser



load_dotenv(override=True)


os.environ["DEPTHAI_HUB_API_KEY"] = "redacted"

device = dai.Device(dai.DeviceInfo("10.12.143.165"))
platform = device.getPlatformAsString()
print(f"Platform: {platform}")

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    cameraNode = pipeline.create(dai.node.Camera).build()
    model_description = dai.NNModelDescription("luxonis-ml-team/bean-classification-dinov3:224-x-224-classification-head-classification-parser", platform=platform)
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
    neuralNetwork = pipeline.create(dai.node.NeuralNetwork).build(cameraNode, nn_archive)

    qNNData = neuralNetwork.out.createOutputQueue()

    print("Pipeline created")
    pipeline.start()


    while pipeline.isRunning():
        inNNData: dai.NNData = qNNData.get()
        tensor = inNNData.getFirstTensor()