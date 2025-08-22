# DepthAI Script node:
#  - Read one source frame + COUNT Buffer (to know gid and num_configs)
#  - For each crop i:
#       * get one config (its seq must be gid + i)
#       * set frame.seq = gid + i
#       * send frame, then send config
#  - This produces N (image,config) pairs -> N outputs from ImageManip.

try:
    GROUP_STRIDE = 1000

    while True:
        # Pull one source frame (full-resolution) and its crop count
        frame = node.inputs["frame_input"].get()
        num_msg = node.inputs["num_configs_input"].get()

        gid = int(num_msg.getSequenceNum())  # group id for this detector frame
        num_configs = len(bytearray(num_msg.getData()))
        node.warn(f"[Script] gid={gid} num_configs={num_configs}")

        for i in range(max(0, num_configs)):
            cfg = node.inputs["config_input"].get()
            seq_for_this_pair = gid + i
            frame.setSequenceNum(seq_for_this_pair)

            node.outputs["output_frame"].send(frame)
            node.outputs["output_config"].send(cfg)

            node.warn(f"[Script] sent pair i={i} seq={seq_for_this_pair}")

except Exception as e:
    node.warn(f"[Script] Exception: {str(e)}")
