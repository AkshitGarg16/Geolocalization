# Installation

Tested on Ubuntu 22.04 with Python 3.10. Anything with Python ≥ 3.9 and OpenCV
should work; the type hints use `X | None` syntax, so 3.9 needs the
`from __future__ import annotations` already present in the sources.

## 1. Get the code

```bash
git clone https://github.com/botvik155/geolocalization.git
cd geolocalization
```

The model weights (`models/best1.pt`, `models/yolov8n.pt`, ~28 MB total) are
committed to the repo, so no separate download is needed.

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `ultralytics`   | YOLOv8 person detection (pulls in `torch`) |
| `opencv-python` | Video decode, drawing, preview window |
| `numpy`         | Vector/matrix maths in the projection chain |
| `scipy`         | Rotation utilities |
| `scikit-learn`  | DBSCAN clustering |
| `pymavlink`     | **`person_tracker.py` only** — live flight control |

`ultralytics` installs a CPU build of PyTorch by default. That is enough to run
the pipeline; it is just slower. For NVIDIA GPUs, install the matching CUDA
build first and let `ultralytics` reuse it:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check which device you got:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Headless machines

If there is no display, `cv2.imshow` will fail. Either set
`'show_preview': False` in the `CONFIG` dict in `src/geolocalization.py`, or
install the headless OpenCV build instead:

```bash
pip uninstall -y opencv-python && pip install opencv-python-headless
```

## 4. Provide the input data

The offline pipeline needs a recorded flight video and its matching per-frame
telemetry CSV — one CSV row per video frame, joined on the `frame` column. See
the *Inputs* section of [README.md](README.md) for the expected columns.

## 5. Point the config at your files

The defaults in `CONFIG` at the top of `src/geolocalization.py` are absolute
paths from the original machine. Edit them:

```python
CONFIG = {
    'video_source':       '/path/to/fcb-YYYYMMDD-HHMMSS.mp4',
    'telemetry_csv':      '/path/to/fcb-YYYYMMDD-HHMMSS.csv',
    'yolo_model':         '/path/to/geolocalization/models/best1.pt',
    'csv_output_path':    '/path/to/geolocalization/outputs/detections.csv',
    'clustered_csv_path': '/path/to/geolocalization/outputs/clustered_detections.csv',
    ...
}
```

Also replace the hard-coded intrinsic matrix in `PersonLocalisationPipeline`
with your own camera calibration, and set `zoom_baseline` to the optical zoom
you calibrated at — otherwise every coordinate carries a systematic error. See
the README's *Camera intrinsics* section.

## 6. Run

The output directory is created automatically if it is missing.

```bash
cd src
python3 geolocalization.py
```

The modules import each other by bare module name, so they must be run from
inside `src/`.

## 7. Verify

On a good start you should see, in order:

```
Telemetry CSV loaded: N row(s) from … (frames 0–N-1)
YOLO model loaded ✓
Video opened: … (N frames)
Detected 1 person(s) at t=…
  Person 1: lat=…, lon=…
```

and `outputs/detections.csv` growing. On exit, `Flushing all detection
buffers …` writes `outputs/clustered_detections.csv`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: Telemetry CSV not found` | `telemetry_csv` path is wrong |
| `No usable rows in telemetry CSV` | CSV has no numeric `frame` column |
| `Cannot open video source` | Bad path, or missing codec — try `pip install opencv-python` over a headless build, or re-encode the file |
| Every frame logs `No telemetry row for frame N` | Video and CSV are not the matched pair, or frame numbering is offset |
| `Incomplete telemetry at frame N` | Blank `lat_deg`/`alt_rel_m`/attitude fields in that row — frame is skipped by design |
| `Could not project GPS` | Ray points at or above the horizon; expected for near-horizon detections |
| `cv2.error: … not implemented … GTK` | Headless machine — see step 3 |
| Coordinates are plausible but consistently offset | Wrong intrinsics or `zoom_baseline`; re-calibrate |

## Live tracker (optional)

`src/person_tracker.py` additionally needs a live camera and a MAVLink link, and
commands the aircraft. Set `mavlink_connection` in its own `CONFIG`, give it a
UDP port no ground station is already holding, and test in SITL before flying.
