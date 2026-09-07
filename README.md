# geolocalization

**Air-to-ground person geolocalisation for an ArduPilot multirotor.**
A downward-facing camera detects people with YOLOv8; every detection's pixel
centre is ray-cast through the camera model, rotated into the world frame using
live MAVLink attitude, intersected with the ground plane, and converted to a
WGS-84 latitude/longitude. Repeated fixes on the same person are consolidated
into a single coordinate with DBSCAN.

No ROS 2, no MAVROS, no `cv_bridge` — the pipeline talks to the flight
controller directly over `pymavlink` and to the camera through OpenCV, so it
runs on a bare onboard computer.

| Script | Role |
|---|---|
| `src/geolocalization.py`        | **Main pipeline** — detect → project to GPS → log → cluster |
| `src/coordinate_transformer.py` | Pixel → camera ray → body → NED → lat/lon maths |
| `src/yolo_detector.py`          | YOLOv8 person detector + bounding-box/GPS overlay |
| `src/person_tracker.py`         | **Separate mode** — visual servoing: keep the largest person centred under the drone |

---

## How a pixel becomes a coordinate

`CoordinateTransformer.pixel_to_gps()` is the whole geometry chain:

1. **Pixel → camera ray.** Back-project `(u, v)` through the intrinsic matrix
   `K`, giving a unit direction in the camera frame.
2. **Camera → body.** Fixed 90° rotation for the downward-facing mount
   (camera `+X` → body `−Y`, camera `+Y` → body `+X`).
3. **Body → world (NED).** Apply `R_yaw · R_pitch · R_roll` from the live
   `ATTITUDE` message. Yaw is outermost, which is the correct order for NED.
4. **Ray → ground plane.** Scale the ray until it reaches
   `drone_altitude − ground_altitude`. A ray pointing at or above the horizon
   returns `None` and the detection is dropped rather than projected to infinity.
5. **NED offset → lat/lon.** Convert the north/east offset with the
   latitude-dependent metres-per-degree series (not a flat-Earth constant), and
   add it to the drone's own GPS position.

**Assumption to be aware of:** the ground is modelled as a flat plane at
`ground_altitude = 0` relative to the drone's AGL altitude. Terrain relief under
the target translates directly into horizontal error, roughly
`error ≈ relief × tan(off-nadir angle)`. Detections near the image centre are
therefore substantially more accurate than detections near the edge.

### DBSCAN consolidation

A person walking through frame produces one fix per frame, each with its own
noise. `DetectionBuffer` accumulates fixes per detection slot and flushes them
through DBSCAN (in a local metric XY projection, not in degrees) whenever the
slot jumps more than `cluster_jump_metres` — i.e. when the slot has almost
certainly latched onto a different person. Each surviving cluster is written out
as one averaged coordinate with its point count, mean confidence, and time span.
Remaining buffers are flushed on shutdown, so no detections are lost on Ctrl-C.

---

## Telemetry

A background thread (`MAVLinkReader`) consumes the MAVLink stream and keeps a
single `TelemetryState` snapshot fresh:

| Message | Fields used |
|---|---|
| `GLOBAL_POSITION_INT` | `lat`, `lon`, `relative_alt` (AGL), `hdg` (compass) |
| `ATTITUDE`            | `roll`, `pitch`, `yaw` |

Frames are only processed when **every** field is present *and* the snapshot is
younger than `max_cache_age` (0.5 s default). Stale telemetry skips the frame
instead of projecting a detection with an out-of-date attitude — a 0.5 s-old
roll angle at 30 m altitude is worth tens of metres of ground error.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python numpy scipy scikit-learn pymavlink
```

CUDA is optional — `ultralytics` falls back to CPU, at a lower frame rate.

## Run

The modules import each other by bare name (`from yolo_detector import ...`),
so **run them from inside `src/`**:

```bash
cd src
python3 geolocalization.py
```

Point `yolo_model` at the weights in `models/` (or copy them next to the
scripts) before the first run — see the config table below.

Outputs, written to the working directory:

| File | Contents |
|---|---|
| `detections.csv`            | One row per person per frame: person lat/lon, drone lat/lon/alt, roll/pitch/yaw (degrees), confidence |
| `clustered_detections.csv`  | One row per DBSCAN cluster: consolidated lat/lon, point count, mean confidence, first/last timestamp |

A live preview window shows bounding boxes annotated with each target's
coordinates, plus an altitude/attitude/compass HUD. Set `show_preview: False`
for headless flight.

### Configuration

Edit the `CONFIG` dict at the top of `geolocalization.py`:

| Key | Default | Notes |
|---|---|---|
| `camera_source`        | `0` | Device index, or `'rtsp://…'` / GStreamer pipeline |
| `mavlink_connection`   | `'udpin:0.0.0.0:14550'` | Or `'/dev/ttyUSB0,57600'` for serial |
| `yolo_model`           | `'best1.pt'` | `models/best1.pt` (fine-tuned) or `models/yolov8n.pt` (stock) |
| `confidence_threshold` | `0.5` | |
| `max_cache_age`        | `0.5` s | Telemetry freshness gate |
| `cluster_eps_metres`   | `2.0` | DBSCAN neighbourhood radius |
| `cluster_min_samples`  | `3` | Minimum fixes to form a cluster |
| `cluster_jump_metres`  | `3.0` | Slot-reassignment threshold that triggers a flush |

> **Use its own MAVLink endpoint.** Two processes bound to the same UDP port
> steal each other's packets. If a GCS already holds 14550, forward the link to
> a second port with MAVProxy or mavlink-router and point
> `mavlink_connection` there.

### Camera intrinsics — calibrate before flying

`PersonLocalisationPipeline` hard-codes a 640×480 intrinsic matrix:

```
fx = 393.13   fy = 394.76
cx = 321.48   cy = 241.58
```

These belong to **one specific camera at one specific resolution**. A different
lens, sensor, or capture resolution makes every coordinate wrong by a
systematic, hard-to-spot amount. Re-calibrate with OpenCV's chessboard
routine and replace the matrix.

---

## Person tracker (visual servoing)

`person_tracker.py` is a separate mode, not part of the geolocalisation
pipeline. It locks onto the **largest** detected person and commands
body-frame velocities so the drone hovers over them:

- pixel `+X` (target right of centre) → body `+Y` (strafe right)
- pixel `+Y` (target below centre) → body `+X` (nose forward)

Proportional control with a dead zone, sent as
`SET_POSITION_TARGET_LOCAL_NED` velocity set-points at 20 Hz. Defaults:
`gain_xy = 0.0025`, `max_linear_speed = 0.4 m/s`, `dead_zone_px = 20`. It
publishes an explicit hover when no target is visible, so loss of detection
stops the drone rather than letting the last set-point run on.

```bash
cd src
python3 person_tracker.py
```

> **Flight safety.** The drone must already be armed, airborne, and in GUIDED
> before either script is started. Test in SITL first, then fly with a pilot on
> the sticks and a mode switch ready — a velocity set-point stream overrides
> position hold.

## Repository layout

```
src/
  geolocalization.py           main pipeline
  coordinate_transformer.py    projection maths
  yolo_detector.py             detection + visualisation
  person_tracker.py            visual servoing mode
models/
  best1.pt                     fine-tuned person detector
  yolov8n.pt                   stock YOLOv8-nano
```
