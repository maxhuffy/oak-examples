import depthai as dai


visualizer = dai.RemoteConnection()
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    raw_stream = cam.requestOutput((4000, 3000), dai.ImgFrame.Type.NV12, fps=30)

    encoder = pipeline.create(dai.node.VideoEncoder).build(
        raw_stream,
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )
    encoder.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.VBR)
    encoded_output = encoder.out

    benchmark = pipeline.create(dai.node.BenchmarkIn)
    encoded_output.link(benchmark.input)
    benchmark.logReportsAsWarnings(True)

    visualizer.addTopic("H.264 Stream", encoded_output)
    pipeline.run()
    