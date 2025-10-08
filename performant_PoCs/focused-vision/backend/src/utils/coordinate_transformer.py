import depthai as dai


def transform_ymax(detection: dai.ImgDetection):
    return min(1., detection.ymin + ((detection.ymax - detection.ymin) * 0.15))


def transform_ymin(detection: dai.ImgDetection):
    return max(0., detection.ymin - 0.03)
