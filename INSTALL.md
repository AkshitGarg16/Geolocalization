# Installation

Tested on Ubuntu 22.04 with Python 3.10. Anything with Python ≥ 3.9 and OpenCV
should work; the type hints use `X | None` syntax, so 3.9 needs the
`from __future__ import annotations` already present in the sources.

## 1. Get the code

```bash
git clone https://github.com/botvik155/geolocalization.git
cd geolocalization
```

The model weights (`models/yoloe-26x-seg.pt`, `models/yoloe-26l-seg.pt`, `models/best1.pt`, `models/yolov8n.pt`, ~280 MB total) are
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
| `ultralytics`   | YOLOE person detection (pulls in `torch`) |
| `opencv-python` | Drawing, preview window |
| `av`            | H.265 decode of the bag's `FFMPEGPacket` stream (PyAV) |
| `numpy`         | Vector/matrix maths in the projection chain |
| `scipy`         | Rotation utilities and attitude SLERP |
| `scikit-learn`  | DBSCAN clustering |
| `pymavlink`     | **`person_tracker.py` only** — live flight control |

### ROS 2

Reading the bag needs a ROS 2 install on the `PYTHONPATH` — `rosbag2_py`,
`rclpy`, `rosidl_runtime_py` and the `sensor_msgs` interfaces. Humble was used
here:

```bash
source /opt/ros/humble/setup.bash
python3 -c "import rosbag2_py, rclpy; print('ok')"
```

Two message packages the bag references are **not** required:
`ffmpeg_image_transport_msgs` (the image packets are unpacked from CDR
directly by `bag_source.parse_ffmpeg_packet`) and `fcb_interfaces` (`/fcb/state`
is skipped entirely — see the README on optical zoom). Build them only if you
want per-frame zoom.

Note that a venv does not see system ROS packages unless it is created with
`--system-site-packages`.

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

One rosbag2 directory holding both the imagery and the MAVROS telemetry:

```
~/rosbags/25m_1x_2ms/
    metadata.yaml
    fcb-20260912-091108_0.db3
```

See the *Inputs* section of [README.md](README.md) for the topics it must
carry. Check what a bag actually has with:

```bash
ros2 bag info ~/rosbags/25m_1x_2ms
```

## 5. Point the config at your files

The defaults in `CONFIG` at the top of `src/geolocalization.py` are absolute
paths from the original machine. Edit them:

```python
CONFIG = {
    'bag_path':           '/path/to/rosbags/25m_1x_2ms',
    'zoom_ratio':         'auto',   # from the bag name; or a number to force it
    'yolo_model':         '/path/to/geolocalization/models/yoloe-26x-seg.pt',
    'csv_output_path':    '/path/to/geolocalization/outputs/detections.csv',
    'clustered_csv_path': '/path/to/geolocalization/outputs/clustered_detections.csv',
    ...
}
```

Or pass a bag on the command line, which overrides `bag_path`:

```bash
python3 geolocalization.py ~/rosbags/25m_1x_2ms
python3 geolocalization.py ~/rosbags/30m_2x_2ms --zoom 2   # force the zoom
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
YOLO model loaded ✓
Telemetry loaded: 27391 GPS, 27391 alt, 66006 IMU over 548.8s
Video: 1920x1080, codec hevc, topic /fcb/image/ffmpeg
Streaming /home/…/25m_1x_2ms — press q in the window to stop
```

and a preview window playing the flight with boxes drawn on it. One
`libav.hevc: PPS id out of range` line at the very start is normal — it is the
decoder waiting for the stream's first parameter sets, and it recovers on the
first keyframe.

On exit the detection buffers are flushed to
`outputs/clustered_detections.csv` and a one-line summary reports how many
frames were shown, how many the detector ran on, and how many had no
telemetry.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: rosbag2_py` | ROS 2 not on the `PYTHONPATH` — `source /opt/ros/humble/setup.bash`, or recreate the venv with `--system-site-packages` |
| `ModuleNotFoundError: av` | `pip install av` |
| `FileNotFoundError: Bag directory not found` | `bag_path` must be the **directory** holding `metadata.yaml`, not the `.db3` file |
| `Bag … is missing telemetry topic(s)` | Topic names differ — override them in `CONFIG['topics']` |
| `Bag has no image topic` | Same, for `CONFIG['topics']['image']`; the error lists what the bag does carry |
| `libav.hevc: PPS id out of range` at startup | Normal — the decoder has not reached the first keyframe yet |
| Window plays but the HUD says `No telemetry` | The camera started before MAVROS did; the first second or so of a bag legitimately has none. It also appears across a telemetry dropout wider than `max_telemetry_gap_s`, where projecting would mean inventing a position |
| `header stamp(s) unusable … fell back to bag receive-time` | That topic's stamps are unset or in a different clock domain. Sync is degraded — see the README's *Time sync* section |
| `N message(s) before the first GPS fix had no stamp to inherit` | Normal: `rel_alt` has no header and borrows the GPS stamp, so any that precede the first fix are dropped |
| Playback is behind the bag clock | Expected at 1080p — see the README on keeping up; raise `detect_every_n` or set `realtime: False` |
| `Could not project GPS` | Ray points at or above the horizon; expected for near-horizon detections |
| `cv2.error: … not implemented … GTK` | Headless machine — see step 3 |
| Coordinates are plausible but consistently offset | Wrong intrinsics or `zoom_baseline`; re-calibrate |

## Live tracker (optional)

`src/person_tracker.py` additionally needs a live camera and a MAVLink link, and
commands the aircraft. Set `mavlink_connection` in its own `CONFIG`, give it a
UDP port no ground station is already holding, and test in SITL before flying.
