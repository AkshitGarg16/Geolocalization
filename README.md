# geolocalization

**Air-to-ground person geolocalisation from recorded drone footage.**
A downward-facing camera records video while the flight controller logs one row
of GPS/IMU telemetry per video frame. This pipeline detects people with YOLOv8,
ray-casts each detection's pixel centre through the camera model, rotates it
into the world frame using that frame's attitude, intersects it with the ground
plane, and converts the result to a WGS-84 latitude/longitude. Repeated fixes on
the same person are consolidated into a single coordinate with DBSCAN.

No ROS 2, no MAVROS, no `cv_bridge`. The main pipeline is pure OpenCV + NumPy +
`ultralytics`, so it runs offline on any machine that can open the video.

| Script | Role |
|---|---|
| `src/geolocalization.py`        | **Main pipeline** — detect → project to GPS → log → cluster |
| `src/coordinate_transformer.py` | Pixel → camera ray → body → NED → lat/lon maths |
| `src/yolo_detector.py`          | YOLOv8 person detector + bounding-box/GPS overlay |
| `src/person_tracker.py`         | **Separate live mode** — visual servoing over MAVLink; keeps the largest person centred under the drone |

See [INSTALL.md](INSTALL.md) for setup.

---

## Inputs

The recorder writes an `.mp4` and a `.csv` together, **frame for frame**, so
telemetry is looked up by frame index — there is no clock synchronisation step
and no timestamp matching to get wrong. A video frame with no matching CSV row,
or with any missing field, is skipped rather than projected with a guess.

Expected CSV columns (extras are ignored):

| Column | Meaning |
|---|---|
| `frame`        | Video frame index — the join key |
| `t_wall_utc`   | ISO-8601 wall clock; falls back to `t_mono_s`, then `0.0` |
| `lat_deg`, `lon_deg` | Drone position |
| `alt_rel_m`    | Altitude above launch (AGL) |
| `roll_deg`, `pitch_deg`, `yaw_deg` | Attitude, degrees (converted to radians on load) |
| `heading_deg`  | Raw compass heading, HUD display only |
| `zoom_ratio`   | Camera optical zoom for that frame |

### Optical zoom

`zoom_ratio` is divided by `zoom_baseline` (the zoom the intrinsics were
calibrated at) and the result scales `fx`/`fy` for that frame. The principal
point is left alone. Ignoring this on a zooming camera is a silent,
systematically wrong projection, not a noisy one.

---

## How a pixel becomes a coordinate

`CoordinateTransformer.pixel_to_gps()` is the whole geometry chain:

1. **Pixel → camera ray.** Back-project `(u, v)` through the intrinsic matrix
   `K` (focal length scaled by this frame's zoom), giving a unit direction in
   the camera frame.
2. **Camera → body.** Fixed 90° rotation for the downward-facing mount
   (camera `+X` → body `−Y`, camera `+Y` → body `+X`).
3. **Body → world (NED).** Apply `R_yaw · R_pitch · R_roll` from the frame's
   attitude. Yaw is outermost, which is the correct order for NED.
4. **Ray → ground plane.** Scale the ray until it reaches the drone's AGL
   altitude. A ray pointing at or above the horizon returns `None` and the
   detection is dropped rather than projected to infinity.
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

> Detection slots are just the per-frame YOLO output order, not identity
> tracking. The jump threshold is what keeps two different people from being
> averaged into one coordinate.

---

## Run

The modules import each other by bare name (`from yolo_detector import ...`),
so **run them from inside `src/`**:

```bash
cd src
python3 geolocalization.py
```

The output directory is created automatically if it does not exist.

Outputs, at the paths set in `CONFIG`:

| File | Contents |
|---|---|
| `outputs/detections.csv`            | One row per person per frame: person lat/lon, drone lat/lon/alt, roll/pitch/yaw (degrees), confidence |
| `outputs/clustered_detections.csv`  | One row per DBSCAN cluster: consolidated lat/lon, point count, mean confidence, first/last timestamp |

A preview window shows bounding boxes annotated with each target's coordinates,
plus an altitude/attitude/compass/zoom HUD. Press `q` or `Esc` to stop early
(buffers are still flushed). Set `show_preview: False` for headless runs.

### Configuration

Edit the `CONFIG` dict at the top of `geolocalization.py`:

| Key | Default | Notes |
|---|---|---|
| `video_source`         | `…/fcb-*.mp4` | Recorded flight video |
| `telemetry_csv`        | `…/fcb-*.csv` | Matching per-frame telemetry |
| `zoom_baseline`        | `1.0` | Zoom ratio the intrinsics were calibrated at |
| `yolo_model`           | `models/best1.pt` | Fine-tuned, or `models/yolov8n.pt` for stock |
| `confidence_threshold` | `0.5` | |
| `csv_output_path`      | `outputs/detections.csv` | |
| `clustered_csv_path`   | `outputs/clustered_detections.csv` | |
| `show_preview`         | `True` | |
| `cluster_eps_metres`   | `2.0` | DBSCAN neighbourhood radius |
| `cluster_min_samples`  | `3` | Minimum fixes to form a cluster |
| `cluster_jump_metres`  | `3.0` | Slot-reassignment threshold that triggers a flush |

The defaults are absolute paths from the original machine — point them at your
own video, CSV, weights, and output directory before the first run.

### Camera intrinsics — calibrate before trusting the numbers

`PersonLocalisationPipeline` hard-codes a 640×480 intrinsic matrix:

```
fx = 393.13   fy = 394.76
cx = 321.48   cy = 241.58
```

These belong to **one specific camera at one specific resolution**. A different
lens, sensor, or capture resolution makes every coordinate wrong by a
systematic, hard-to-spot amount. Re-calibrate with OpenCV's chessboard routine,
replace the matrix, and set `zoom_baseline` to the zoom you calibrated at.

---

## Person tracker (visual servoing)

`person_tracker.py` is a separate **live** mode, not part of the offline
geolocalisation pipeline — it needs a camera and a MAVLink link to the flight
controller (`pymavlink`). It locks onto the **largest** detected person and
commands body-frame velocities so the drone hovers over them:

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
> before this script is started. Test in SITL first, then fly with a pilot on
> the sticks and a mode switch ready — a velocity set-point stream overrides
> position hold.

> **Use its own MAVLink endpoint.** Two processes bound to the same UDP port
> steal each other's packets. If a GCS already holds 14550, forward the link to
> a second port with MAVProxy or mavlink-router and point
> `mavlink_connection` there.

## Repository layout

```
src/
  geolocalization.py           main offline pipeline
  coordinate_transformer.py    projection maths
  yolo_detector.py             detection + visualisation
  person_tracker.py            live visual-servoing mode
models/
  best1.pt                     fine-tuned person detector
  yolov8n.pt                   stock YOLOv8-nano
outputs/                       created automatically on first run (git-ignored)
```
