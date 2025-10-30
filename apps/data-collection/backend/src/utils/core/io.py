import os
import base64
import numpy as np
import cv2
import requests
from pathlib import Path


def download_file(url: str, path: str) -> Path:
    if not os.path.exists(path):
        print(f"Downloading tokenizer config from {url}...")
        with open(path, "wb") as f:
            f.write(requests.get(url).content)
    return path


def base64_to_cv2_image(base64_data_uri: str):
    if "," in base64_data_uri:
        header, base64_data = base64_data_uri.split(",", 1)
    else:
        base64_data = base64_data_uri

    binary_data = base64.b64decode(base64_data)
    np_arr = np.frombuffer(binary_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img
