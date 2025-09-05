import depthai as dai
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode
import cv2
import time

# Optional fallback decoder
try:
    import zxingcpp
except ImportError:
    zxingcpp = None

class BarcodeDecoder(dai.node.ThreadedHostNode):
    """
    Custom host node that receives ImgFrame messages,
    runs pyzbar (plus optional fallbacks), and emits raw bytes
    in dai.Buffer messages.
    """
    def __init__(self):
        super().__init__()

        # Input queue: receive ImgFrame
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        # Output queue: send Buffer containing barcode bytes
        self.output = self.createOutput()
        self.output.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])

    def run(self):
        while self.isRunning():
            # Use tryGet with timeout to prevent blocking
            in_msg = self.input.tryGet()
            if in_msg is None:
                time.sleep(0.001)  # Short sleep if no message available
                continue
                
            cv_frame = in_msg.getCvFrame()
            pil_img = Image.fromarray(cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB))

            barcodes = decode(pil_img)

            if not barcodes:
                for angle in (90, 180, 270):
                    rotated = pil_img.rotate(angle, expand=True)
                    barcodes = decode(rotated)
                    if barcodes:
                        break

            if not barcodes:
                inverted = ImageOps.invert(pil_img)
                barcodes = decode(inverted)

            if not barcodes and zxingcpp is not None:
                try:
                    import numpy as np
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    zx_results = zxingcpp.read_barcodes(cv_img)
                    barcodes = [type("Z", (), {"data": r.text.encode()}) for r in zx_results]
                except Exception:
                    barcodes = []

            for bc in barcodes:
                buf = dai.Buffer()
                buf.setData(bc.data)
                self.output.send(buf)
                

            if not barcodes:
                time.sleep(0.001)