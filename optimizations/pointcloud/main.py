import depthai as dai


visualizer = dai.RemoteConnection()
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    color_stream = cam.requestOutput((1280, 800), dai.ImgFrame.Type.NV12, fps=30)

    left_cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B
    )
    left_stream = left_cam.requestOutput((1280, 800), dai.ImgFrame.Type.NV12, fps=30)

    right_cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C
    )
    right_stream = right_cam.requestOutput((1280, 800), dai.ImgFrame.Type.NV12, fps=30)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left_stream,
        right_stream,
        dai.node.StereoDepth.PresetMode.DEFAULT
    )

    encoder = pipeline.create(dai.node.VideoEncoder).build(
        color_stream,
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )
    encoder.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.VBR)
    encoded_output = encoder.out

    benchmark = pipeline.create(dai.node.BenchmarkIn)
    stereo.depth.link(benchmark.input)
    benchmark.logReportsAsWarnings(True)

    image_align = pipeline.create(dai.node.ImageAlign)
    stereo.depth.link(image_align.input)
    color_stream.link(image_align.inputAlignTo)

    visualizer.addTopic("_Point Cloud Color", encoded_output)
    visualizer.addTopic("Point Cloud", image_align.outputAligned)
    pipeline.run()
    