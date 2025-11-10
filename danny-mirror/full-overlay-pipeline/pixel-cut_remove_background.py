import base64
import json
import os
from pathlib import Path

import requests

# 1. Put your Pixelcut API key here (from https://developer.pixelcut.ai/)
PIXELCUT_API_KEY = os.environ.get("PIXELCUT_API_KEY", "sk_d9c026c715664ae4955fe09a4a1b48c6")

# 2. Local image path to process
BASE_DIR = Path(__file__).resolve().parent
PERSON_IMAGE_PATH = BASE_DIR / "input_image.png"

REMOVE_BG_URL = "https://api.developer.pixelcut.ai/v1/remove-background"
OUTPUT_PATH = BASE_DIR / "person_no_background.png"


def main():
    if PIXELCUT_API_KEY == "YOUR_PIXELCUT_API_KEY_HERE":
        raise SystemExit("Set PIXELCUT_API_KEY env var or edit the file with your real key.")

    # Pixelcut docs specify header key: X-API-KEY (case sensitive on some gateways)
    headers = {
        "Accept": "application/json",
        "X-API-KEY": PIXELCUT_API_KEY,
    }

    form_data = {
        "format": "png",
        "shadow": json.dumps(
            {
                "enabled": False,
                "opacity": 0,
                "light_source": {
                    "size": 0,
                    "position": {"x": 0, "y": 0, "z": 0},
                },
            }
        ),
    }

    if not PERSON_IMAGE_PATH.exists():
        raise SystemExit(f"Input image not found at {PERSON_IMAGE_PATH}")

    with PERSON_IMAGE_PATH.open("rb") as person_image:
        # According to API: multipart field key must be 'image' (binary) OR 'image_url'
        files = {
            "image": (PERSON_IMAGE_PATH.name, person_image, "image/png"),
        }

        resp = requests.post(
            REMOVE_BG_URL,
            headers=headers,
            data=form_data,
            files=files,
            timeout=60,
        )

    if resp.status_code != 200:
        print("Request failed:", resp.status_code, resp.text)
        return

    # API returns JSON with 'result_url'. No base64 in current docs, keep backward compatibility check.
    json_resp = resp.json()
    result_url = json_resp.get("result_url") or json_resp.get("image_url")
    result_base64 = json_resp.get("image_base64")

    if result_base64:
        try:
            decoded = base64.b64decode(result_base64)
        except Exception as e:
            print("Failed to decode base64 image:", e)
        else:
            with open(OUTPUT_PATH, "wb") as out_file:
                out_file.write(decoded)
            print(f"Saved result (base64) to {OUTPUT_PATH}")
            return

    if result_url:
        try:
            img_resp = requests.get(result_url, timeout=60)
        except requests.RequestException as e:
            print("Error downloading result image:", e)
            return
        if img_resp.status_code == 200:
            with open(OUTPUT_PATH, "wb") as out_file:
                out_file.write(img_resp.content)
            print(f"Saved result to {OUTPUT_PATH}")
        else:
            print("Got the URL but failed to download image:", img_resp.status_code)
        return

    print("API response did not include an image (no result_url / image_base64)")


if __name__ == "__main__":
    main()
