import os
import requests

# 1. Put your Pixelcut API key here (from https://developer.pixelcut.ai/)
PIXELCUT_API_KEY = os.environ.get("PIXELCUT_API_KEY", "sk_d9c026c715664ae4955fe09a4a1b48c6")

# 2. Local image paths
PERSON_IMAGE_PATH = "oak-examples\danny-mirror\try-on-clothes\calibration_images\oak_capture_20251107_212523_886212.png"         # your photo
GARMENT_IMAGE_PATH = "oak-examples\danny-mirror\try-on-clothes\pixel-cut-input\puffer_jacket.png"     # photo of the clothes

API_URL = "https://api.developer.pixelcut.ai/v1/try-on"


def main():
    if PIXELCUT_API_KEY == "YOUR_PIXELCUT_API_KEY_HERE":
        raise SystemExit("Set PIXELCUT_API_KEY env var or edit the file with your real key.")

    # open files in a context so they get closed
    with open(PERSON_IMAGE_PATH, "rb") as person_file, open(GARMENT_IMAGE_PATH, "rb") as garment_file:
        headers = {
            "X-API-Key": PIXELCUT_API_KEY,
            # Pixelcut will figure out the content type from multipart
            "Accept": "application/json",
        }

        # these names come from the docs: person_image / garment_image
        # and they are mutually exclusive with the *_url fields :contentReference[oaicite:1]{index=1}
        files = {
            "person_image": ("person.jpg", person_file, "image/png"),
            "garment_image": ("garment.png", garment_file, "image/png"),
        }

        # optional form fields
        data = {
            "preprocess_garment": "true",  # let Pixelcut cut out the garment
            "remove_background": "false",  # keep background of the final image
            "wait_for_result": "true",     # so we get the result URL right away
        }

        resp = requests.post(API_URL, headers=headers, files=files, data=data, timeout=60)

    if resp.status_code != 200:
        print("Request failed:", resp.status_code, resp.text)
        return

    json_resp = resp.json()
    result_url = json_resp.get("result_url")
    print("Pixelcut returned:", result_url)

    # result_url is valid for ~1 hour, so let's download it immediately :contentReference[oaicite:2]{index=2}
    if result_url:
        img_resp = requests.get(result_url, timeout=60)
        if img_resp.status_code == 200:
            out_path = "tryon_result.jpg"
            with open(out_path, "wb") as f:
                f.write(img_resp.content)
            print(f"Saved result to {out_path}")
        else:
            print("Got the URL but failed to download image:", img_resp.status_code)


if __name__ == "__main__":
    main()
