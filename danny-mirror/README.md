# danny-mirror

This folder mirrors the `depth-measurement/calc-spatial-on-host` example so you can iterate on features here without touching the original.

## How to run

1) (Optional) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows bash
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the script

```bash
python main.py            # auto-detect device
# or specify a device by name/ID/IP
python main.py --device <DEVICE_ID_OR_IP>
```

Controls (from the remote UI):
- w/a/s/d: move the ROI
- r/f: increase/decrease ROI size
- q: quit

Notes:
- Requires a connected OAK device and the `depthai` and `depthai-nodes` packages.
- The app opens a remote visualizer on port 8082.
