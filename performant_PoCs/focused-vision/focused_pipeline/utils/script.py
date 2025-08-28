try:
    while True:
        # Pull one source frame (full-resolution) and its crop count
        frame = node.inputs["frame_input"].get()
        num_msg = node.inputs["num_configs_input"].get()

        num_configs = len(bytearray(num_msg.getData()))

        for i in range(max(0, num_configs)):
            cfg = node.inputs["config_input"].get()

            node.outputs["output_frame"].send(frame)
            node.outputs["output_config"].send(cfg)

except Exception as e:
    node.warn(f"[Script] Exception: {str(e)}")
