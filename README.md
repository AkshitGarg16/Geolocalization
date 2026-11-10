# geolocalization

**Air-to-ground person geolocalisation, played live off a ROS 2 bag.**
A downward-facing camera streams H.265 into a rosbag2 alongside MAVROS
GPS/IMU telemetry. This pipeline decodes that bag frame by frame, detects
people with YOLOE, ray-casts each detection's pixel centre through the camera
model, rotates it into the world frame using that frame's attitude, intersects
it with the ground plane — at the height a downward-facing 2D lidar measures,
not the height the barometer guesses — and converts the result to a WGS-84
latitude/longitude. Repeated fixes on the same person are consolidated into a
single coordinate with DBSCAN.

Everything comes out of **one rosbag2 directory** — imagery and telemetry
both. There is no sidecar video file and no per-frame telemetry CSV. Frames are
decoded, detected on and displayed as they stream, paced against the bag's own
clock, so you watch the detector work at the speed the flight happened.

| Script | Role |
|---|---|
| `src/geolocalization.py`        | **Main pipeline** — detect → project to GPS → log → cluster |
| `src/coordinate_transformer.py` | Pixel → camera ray → body → NED → lat/lon maths |
| `src/bag_source.py`             | **Bag I/O** — FFMPEGPacket decode, telemetry interpolation, frame-number ↔ bag-time clock |
| `src/yolo_detector.py`          | YOLOE person detector + bounding-box/GPS overlay |
| `src/play_video.py`             | Plays the bag's imagery with frame number + telemetry on screen; SPACE saves a clean screenshot of the current frame |
| `src/click_points.py`           | Click the target on saved frames; writes one `frame,u,v,label` CSV per person, beside their frames (no image editor needed) |
| `src/extract_annotations.py`    | Diffs your marked-up frame copies against the originals to recover the pixel you marked, as a `frame,u,v,label` CSV |
| `src/geolocate_pixels.py`       | Projects **manually annotated pixels** to GPS, bypassing the detector — for checking the projection against surveyed ground truth |
| `src/person_tracker.py`         | **Separate live mode** — visual servoing over MAVLink; keeps the largest person centred under the drone |

See [INSTALL.md](INSTALL.md) for setup.

---

## Inputs

One rosbag2 directory:

```
~/rosbags/25m_1x_2ms/
    metadata.yaml
    fcb-20260912-091108_0.db3     <- imagery AND telemetry
    fcb-20260912-091108.mp4       <- sidecar, not used (see below)
```

| Topic | Type | Used for |
|---|---|---|
| `/fcb/image/ffmpeg`              | `ffmpeg_image_transport_msgs/FFMPEGPacket` | H.265 imagery, one packet per frame |
| `/mavros/global_position/global` | `sensor_msgs/NavSatFix`   | Drone lat/lon |
| `/mavros/global_position/rel_alt`| `std_msgs/Float64`        | Barometric altitude above launch — the fallback height |
| `/scan`                          | `sensor_msgs/LaserScan`   | Downward-facing 2D lidar — the measured height above ground |
| `/mavros/imu/data`               | `sensor_msgs/Imu`         | Attitude quaternion |
| `/fcb/camera_info`               | `sensor_msgs/CameraInfo`  | Optional intrinsics (off by default — see below) |

Telemetry is **interpolated to each frame's capture time** — linear for
position and altitude, SLERP for orientation — rather than joined by index. A
frame the bag cannot support is displayed but not projected.

### Time sync — everything is keyed on the header stamp

Bag receive-time is when the recorder *wrote* a message, not when the data was
captured, and the topics do not lag by the same amount. Measured over
`25m_1x_2ms` (header stamp minus bag receive-time):

| Topic | mean | sd | min |
|---|---|---|---|
| `/fcb/image/ffmpeg`              | −51.9 ms | 163.2 | −20537 ms |
| `/fcb/camera_info`               | −51.9 ms | 163.2 | −20537 ms |
| `/mavros/global_position/global` |  −0.2 ms |   0.2 |    −4.3 ms |
| `/mavros/imu/data`               |  −0.2 ms |   0.2 |    −6.8 ms |

The camera is ~52 ms behind the flight controller. Keying both off bag time —
which is what this pipeline used to do — matches every frame against telemetry
~52 ms in its own future: a systematic attitude and position lead, not noise.
Everything is now keyed on the **header stamp**, the capture clock, and all
stamps share one UTC epoch domain so they are directly comparable.

That `−20537 ms` is the worst case and it is not a tail: it is the recorder's
20.5 s stall at t = 374 s. Frames captured before the stall were flushed to
disk after it, so their bag timestamps land 20.5 s late:

```
  pkt  capture t   bag t     skew    alt capture   alt bag   pos err
22428     374.16   374.21     51ms         0.35m     0.36m     0.01m
22429     374.18   394.72  20537ms         0.35m    16.54m     2.31m
22430     374.20   394.72  20520ms         0.35m    16.54m     2.31m
22440     394.85   394.90     51ms        16.61m    16.63m     0.00m
```

Those two frames were being projected as though the drone were 16 m up when it
was sitting on the ground. Across the whole flight the correction is worth a
mean of 0.076° in yaw and 0.013 m in position, but p99 2.5° / max 3.4° in yaw
and max 2.27° in pitch — around a metre of ground displacement at 25 m AGL
during the fast manoeuvres, which is where it matters.

**`rel_alt` is the one topic with no header.** It is a bare `Float64`. It is
published back-to-back with `global_position/global` from the same MAVLink
`GLOBAL_POSITION_INT`, so it inherits that message's stamp. Verified on this
bag: all 27391 `rel_alt` messages are immediately preceded by a GPS message
(mean bag gap 0.045 ms, max 3.2 ms), and `rel_alt` is bit-identical to the
headered `global_position/local` z — max difference 0.000 m over the flight.

**Dropouts are refused, not interpolated.** `BagTelemetry.at()` returns `None`
rather than a fabricated value when it would have to interpolate across a hole
wider than `max_gap_s` (default 0.25 s), and the pipeline skips projecting that
frame. Observed spacing on this bag: GPS median 19.7 ms / max 241 ms, IMU
median 8.2 ms / max 96 ms.

**If a bag has a broken clock**, a stamp that is unset or more than 60 s from
bag time is rejected per-message and that message falls back to bag
receive-time, with a warning naming the topic and the count. Nothing is
silently mis-synced.

### Why the bag and not the `.mp4` beside it

The recorder writes an `.mp4` alongside the `.db3`, and it is tempting to use
it. Don't: it holds **31490 frames against the bag's 31507 packets**. The first
17 packets precede the stream's first parameter sets and produce no decodable
picture, so mp4 frame *N* is bag packet *N+17*. Reading imagery from the bag
keeps every frame married to the timestamp it was actually captured at.

That same fact shapes the decoder loop in `bag_source.BagVideo`: each packet is
stamped with its index before being handed to the decoder and the decoded frame
is matched back by that stamp. Pairing them FIFO would silently skew the whole
flight by ~0.29 s — about 0.6 m of ground error at 2 m/s.

Capture stamps are also not strictly monotonic — 155 packets on this bag carry
a stamp earlier than their predecessor's, by up to 67 ms — so the player
rebases its clock on a backward step instead of trying to sleep backwards.

### Height above ground — the lidar, not the barometer

The projection scales every ray by the drone's height, so the height is a
multiplier on every ground offset: get it 6% wrong and a person 15 m from the
nadir point lands ~0.9 m away from where they are.

`rel_alt` is not a measurement of that height. It is a **barometric** altitude
above the point the drone armed at: it drifts with pressure and it is
referenced to the launch pad rather than to the ground the drone is now over.
The drone also carries a **downward-facing 2D lidar** on `/scan`, which
measures the distance to the ground directly, and that is what `alt_agl` now
carries. `altitude_source: 'baro'` (or `--alt-source baro`) goes back to the
barometer alone.

Measured over the bags here — `lidar-baro` is the height error the barometer
was contributing, `coverage` the fraction of in-flight frames the lidar can
answer for after interpolation:

| bag | scans with a height | lidar − baro | p16–p84 | coverage | fit σ |
|---|---|---|---|---|---|
| `20m_1x_1.5ms` |  845/963  | **+1.23 m** | 1.38 | 100%  | 0.012 m |
| `20m_1x_1ms`   | 1111/1335 | **+1.61 m** | 0.92 | 100%  | 0.015 m |
| `25m_1x_2ms`   |  867/2633 | **+0.94 m** | 1.20 | 100%  | 0.008 m |
| `25m_2x_1.5ms` |  972/1243 | **+1.60 m** | 1.24 | 99.6% | 0.016 m |
| `25m_2x_1ms`   | 1191/1425 | **+1.82 m** | 1.04 | 100%  | 0.016 m |
| `25m_2x_2ms`   |  612/755  | **+1.35 m** | 1.44 | 100%  | 0.011 m |

The barometer reads 0.9–1.8 m low on every one of these flights, and **not by
the same amount** — so it is not a bias you can calibrate out once. At 25 m AGL
that is a 4–7% scale error on every ground offset. `20m_2x_1.5ms`,
`20m_2x_1ms` and `20m_2x_1ms_occluded` were recorded without a `/scan` topic
at all; those fall back to the barometer and say so in the log.

Run `probe_lidar.py` to get that table for your own bags:

```bash
cd src
python3 probe_lidar.py ~/rosbags/*
python3 probe_lidar.py ~/rosbags/25m_1x_2ms --detail   # walk the flight
```

#### What the scans look like, and how the height comes out of them

`sensor_msgs/LaserScan`, `frame_id: laser`, ~4.8 Hz, 100 points per message,
`range_max` 50 m, misses written as `inf` or `-1.0`. The fan lies within ±45°
of nadir, but **each message carries a different slice of it**: the window runs
29° to 82° wide, `angle_increment` varies from 0.30° to 0.83° between messages,
and the window does not always contain the nadir direction. Anything that
assumes a fixed beam layout is wrong on this sensor, so every scan is read
through its own `angle_min` / `angle_increment`.

Over flat ground the profile is a V — slant range is shortest along the nadir
ray and grows as `h / cos(off-nadir angle)` towards both edges. The height is
the range at the bottom of that V, which is what makes it **independent of
attitude**: a roll or pitch slides the bottom of the V along the fan, it does
not change its depth. `lidar_altimeter.height_from_scan` finds that bottom by
fitting rather than by picking the smallest sample:

1. Keep the valid returns, then take the beams within `half_window_deg` (10°)
   of the shortest one and fit range against angle with a quadratic.
2. Reject outliers and re-fit, twice. The rejection is **asymmetric**: a return
   more than 3σ *shorter* than the fit is something between the drone and the
   ground — a person, a vehicle, a tree — and must not be allowed to pull the
   height down. A longer one is usually just noise.
3. The vertex of the fit is the height. The scan is refused if the parabola
   opens downward or its vertex falls outside the window, which is what happens
   when that message's slice of the fan misses nadir.

Step 2 is why a person standing under the drone does not corrupt the height,
and step 3 is why a partial scan pointing off to one side never reports a slant
range as a height.

Why a fit and not the smallest return, or a plane fitted to the whole fan:
measured against the barometer on `25m_1x_2ms`, the high-frequency spread is
0.19 m for the ±10° vertex fit, 0.27 m for the raw minimum, and 1.0–1.7 m for
anything fitted across the full ±45° fan — the wide fan is dominated by ground
that is not flat and by the message's shifting angular window, not by sensor
noise, which is only ~0.02 m.

#### Dropouts, and what stands in

About 10% of in-flight scans are refused (almost all of them the
nadir-not-in-window case), leaving ~0.4 s holes. `LidarAltimeter.at()`
interpolates across those but **refuses a hole wider than `max_gap_s`**
(0.6 s), the same way telemetry dropouts are refused rather than invented.
Where the lidar is genuinely silent, the barometer stands in **shifted onto the
lidar by the offset measured at the surrounding scans** — not by the flight
median, which is only the last resort for a query the per-scan series cannot
reach. The local offset is what keeps the height continuous: at the edge of a
gap the shifted barometer equals the lidar reading exactly, and it drifts from
there only at the rate the barometer does. Forcing 588 fallback samples on
`25m_1x_2ms` (by tightening `max_gap_s` to 0.25 s), the height steps by a mean
of **0.02 m and at most 0.07 m** across the 85 source changes, against 0.19 m
of genuine movement between neighbouring samples; with the flight median
instead it was a mean of 0.50 m and up to 1.84 m.

Every row of `detections.csv` records which of the three it was
(`altitude_source`: `lidar`, `lidar-bias`, `baro`) along with both raw values,
so a run can be re-scored either way after the fact. On a 995-detection run
over `25m_1x_2ms` every row came out `lidar`.

> **The fitted extrinsics predate this.** `R_BODY_CAM` and the
> `GROUND_OFFSET_*` translation in `coordinate_transformer.py` were fitted
> against surveyed points using **barometric** heights. Feeding the projection a
> height that is ~1 m larger scales the fitted offsets by a few per cent, and
> the constant translation absorbed some of the old height bias. Re-fit both
> against the lidar heights before quoting the 0.55 m figure again.

### Attitude convention

MAVROS publishes orientation in **ENU/FLU** (ROS REP-103). The projection was
calibrated against the old recorder CSV, whose `yaw_deg` matched its
`heading_deg` — i.e. **NED/FRD**, clockwise from north. `bag_source` converts
(`roll` unchanged, `pitch` negated, `yaw → 90° − yaw`). Verified against GPS
course-over-ground on `25m_1x_2ms`: course 179.6° vs converted yaw 178.6°, and
course 0.4° vs 358.7°.

Getting this wrong does not produce an obvious failure — it produces
confidently wrong coordinates — so re-check it if you change the attitude
source.

### Optical zoom

`zoom_ratio` is divided by `zoom_baseline` (the zoom the intrinsics were
calibrated at) and the result scales `fx`/`fy`. The principal point is left
alone. Ignoring this on a zooming camera is a silent, systematically wrong
projection, not a noisy one.

Because zoom varies from bag to bag, it is **not hardcoded**. `zoom_ratio`
accepts:

| Value | Behaviour |
|---|---|
| `'auto'` (default) | Read the zoom out of the bag directory name — `25m_1x_2ms` → 1x, `30m_2.5x_3ms` → 2.5x |
| a number | Use it verbatim; always wins |
| `--zoom 2` / `--zoom auto` | Same, per run, from the command line |

So a folder of bags at different zooms needs no per-bag edit, provided the
names carry the zoom. A name with no zoom token, or with two of them, is
treated as unknown: the run falls back to 1.0 **and logs a warning**, rather
than guessing silently.

#### Why not just read it off `/fcb/state`

The camera does publish its zoom there, but the topic is a
`fcb_interfaces/msg/CameraState` and that package is not installed, so the
message cannot be deserialised. `zoom_from_camera_state` can find the field
heuristically — it scans every 4-byte-aligned offset for the one float32 that
stays in [1, 60] and takes only a handful of distinct values — and on both bags
available here it lands on offset 88 and reads 1.0.

That is **not** proof it works. Both of those bags were shot at 1x, and a field
that is genuinely the zoom is indistinguishable from a field that happens to be
1.0 when every sample reads 1.0. So `use_camera_state_zoom` defaults to
`False`. To settle it, run the probe against a bag you know is not 1x:

```bash
cd src
python3 probe_zoom.py ~/rosbags/*
```

```
bag                               from name  from /fcb/state   detail
25m_1x_2ms                               1x               1x   offset 88, 1 distinct value(s)  [agree]
```

If a 2x bag makes that column read `2x`, the decoder is finding the real field
and `use_camera_state_zoom: True` becomes safe — and would then also track zoom
changes *within* a flight, which the name cannot. If it still reads `1x` there,
leave it off and keep the zoom in the bag name. When both sources are available
and disagree, the name wins and the run logs a warning.

The durable fix is to build and source `fcb_interfaces`, after which the topic
can be read properly.

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
4. **Ray → ground plane.** Scale the ray until it reaches the drone's height
   above ground — measured by the downward lidar, not the barometer; see
   *Height above ground*. A ray pointing at or above the horizon returns `None` and the
   detection is dropped rather than projected to infinity.
5. **NED offset → lat/lon.** Convert the north/east offset with the
   latitude-dependent metres-per-degree series (not a flat-Earth constant), and
   add it to the drone's own GPS position.

**Assumption to be aware of:** the ground is modelled as a flat plane at
`ground_altitude = 0` relative to the drone's height above ground. The lidar
measures that height under the *drone*, so the plane is now pinned to the
terrain the drone is over rather than to the launch pad — but it is still a
plane. Terrain relief under
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
python3 geolocalization.py                        # bag from CONFIG
python3 geolocalization.py ~/rosbags/25m_1x_2ms   # or name one
```

The output directory is created automatically if it does not exist.

Outputs, at the paths set in `CONFIG`:

| File | Contents |
|---|---|
| `outputs/detections.csv`            | One row per person per frame: person lat/lon, drone lat/lon/alt, which height source was used and both raw heights, roll/pitch/yaw (degrees), confidence |
| `outputs/clustered_detections.csv`  | One row per DBSCAN cluster: consolidated lat/lon, point count, mean confidence, first/last timestamp |

### Watching it run

The preview window shows the detection ROI, every box the model returns
annotated with the coordinate it projects to, and a telemetry HUD, all paced
against the bag clock so playback matches the real flight. The bottom-right
readout gives processing FPS, playback speed, and — when the detector cannot
keep up — how far behind the bag clock it has fallen.

| Key | Action |
|---|---|
| `space` | pause / resume |
| `,` `.` | step one frame (while paused) |
| `[` `]` | slower / faster |
| `r`     | back to 1x |
| `d`     | toggle the detector (video keeps playing) |
| `h`     | toggle the HUD |
| `s`     | save the current frame to `outputs/snapshots/` |
| `q`     | quit (buffers are still flushed) |

Set `show_preview: False` for headless runs.

**Keeping up with the clock.** The bag runs at ~58 FPS; YOLOE-26X on an RTX
3050 manages ~20 FPS at 1080p. Playback stays in real time regardless — what
gives is detection coverage, and the HUD says so. To cover every frame instead,
set `realtime: False` and let it run slower than the flight, or raise
`detect_every_n` to detect on a subset while still displaying everything.

**Seeking.** `start_offset_s` skips ahead, but an inter-coded stream has no
random access: earlier frames are still decoded, just not colour-converted, so
jumping to 400 s takes a couple of minutes of decode before the window opens.

### Configuration

Edit the `CONFIG` dict at the top of `geolocalization.py`:

| Key | Default | Notes |
|---|---|---|
| `bag_path`             | `~/rosbags/25m_1x_2ms` | The rosbag2 directory — the only input |
| `topics`               | see `CONFIG` | Topic names, if your recorder differs |
| `altitude_source`      | `'lidar'` | `'lidar'` measures the height with the downward 2D lidar on `/scan`; `'baro'` uses `rel_alt` alone. See *Height above ground* |
| `lidar`                | see `CONFIG` | `max_gap_s` 0.6 (longest lidar hole to interpolate across), `half_window_deg` 10 (beams around nadir that are fitted), `min_beams` 8, `mount_offset_m` 0.0 (lidar-to-camera height offset, if you have measured it) |
| `max_telemetry_gap_s`  | `0.25` | Refuse to interpolate telemetry across a hole wider than this; the frame is shown but not projected |
| `zoom_ratio`           | `'auto'` | Optical zoom: `'auto'` reads it from the bag name, or give a number. See *Optical zoom* |
| `use_camera_state_zoom`| `False` | Also try `/fcb/state`. Experimental — validate with `probe_zoom.py` first |
| `zoom_baseline`        | `1.0` | Zoom ratio the intrinsics were calibrated at |
| `use_camera_info`      | `False` | Take intrinsics from `/fcb/camera_info` instead of the calibrated matrix |
| `realtime`             | `True` | Pace to the bag clock; `False` runs as fast as it can |
| `playback_speed`       | `1.0` | Multiplier on the bag clock |
| `detect_every_n`       | `1` | Detect on every Nth frame, display all of them |
| `start_offset_s`       | `0.0` | Skip this many seconds into the flight |
| `display_width`        | `1280` | Preview downscale; `None` for native 1920 |
| `yolo_model`           | `models/yoloe-26x-seg.pt` | YOLOE v26-X, prompted with the single class `person`. `models/yoloe-26l-seg.pt` is ~1.9x faster at slightly lower recall; `models/best1.pt` is the fine-tuned alternative |
| `roi_fraction`         | `0.8` | Detection runs on the centred crop only: 0.8 = central 80% of width and height. Boxes are mapped back to full-frame pixels. See *Choosing the ROI* |
| `confidence_threshold` | `0.5` | |
| `csv_output_path`      | `outputs/detections.csv` | |
| `clustered_csv_path`   | `outputs/clustered_detections.csv` | |
| `show_preview`         | `True` | |
| `snapshot_dir`         | `outputs/snapshots` | Where `s` writes frames |
| `cluster_eps_metres`   | `2.0` | DBSCAN neighbourhood radius |
| `cluster_min_samples`  | `3` | Minimum fixes to form a cluster |
| `cluster_jump_metres`  | `3.0` | Slot-reassignment threshold that triggers a flush |

Set `csv_output_path` or `clustered_csv_path` to `None` to turn that output
off entirely. The defaults are absolute paths from the original machine —
point them at your own bag, weights, and output directory before the first run.

### Choosing the ROI

`roi_fraction` crops the frame before detection. It is **not** a speed control:
Ultralytics letterboxes whatever it is given to 640 px, so every setting costs
the same ~50 ms/frame. What it changes is scale — a tighter crop means more
output pixels per person — traded against coverage of the frame edges.

Measured on 75 frames from `25m_1x_2ms` around t = 425-450 s, where people are
in view:

| `roi_fraction` | frames with a detection | total detections | ms/frame |
|---|---|---|---|
| 0.5 | 35/75 | 39 | 50.2 |
| 0.6 | 41/75 | 56 | 50.1 |
| 0.7 | 45/75 | 60 | 50.3 |
| **0.8** | **49/75** | **68** | **50.7** |
| 0.9 | 61/75 | 95 | 50.9 |
| 1.0 | 50/75 | 74 | 50.4 |

Two things worth knowing. Coverage matters more than magnification on this
footage — subjects drift to the frame edges, so the old 0.6 was throwing away
detections for no compute saving. And the best setting is not the largest:
0.9 beat full-frame here. Treat that peak as indicative rather than exact —
it is one 75-frame sample from one flight.

### Camera intrinsics — calibrate before trusting the numbers

`PersonLocalisationPipeline.CALIBRATED_K` hard-codes a 1920×1080 intrinsic
matrix:

```
fx = 1584.03   fy = 1581.52
cx =  953.28   cy =  518.97
```

These belong to **one specific camera at one specific resolution**. A different
lens, sensor, or capture resolution makes every coordinate wrong by a
systematic, hard-to-spot amount. Re-calibrate with OpenCV's chessboard routine,
replace the matrix, and set `zoom_baseline` to the zoom you calibrated at.

**The bag's `/fcb/camera_info` is not a substitute.** It reports
`fx = fy = 1545`, `cx = 960`, `cy = 540` — a round focal length and a
dead-centre principal point, constant for the whole flight: placeholders, not a
calibration. `use_camera_info` is `False` for that reason, and the fitted
extrinsics in `coordinate_transformer.py` are matched to the calibrated matrix
above, so the two must be changed together.

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
  geolocalization.py           main pipeline, live off a bag
  bag_source.py                rosbag2 I/O: decode, telemetry, frame clock
  lidar_altimeter.py           height above ground from the downward 2D lidar
  coordinate_transformer.py    projection maths
  yolo_detector.py             detection + visualisation
  play_video.py                bag player, SPACE captures frames
  click_points.py              click targets on saved frames
  extract_annotations.py       recover marked pixels from edited frames
  geolocate_pixels.py          geolocalise hand-annotated pixels
  probe_lidar.py               what the lidar says per bag, vs the barometer
  probe_zoom.py                check what zoom a bag reports, and from where
  person_tracker.py            live visual-servoing mode
models/
  yoloe-26x-seg.pt             YOLOE v26-X open-vocabulary detector (default)
  yoloe-26l-seg.pt             YOLOE v26-L, faster; used by person_tracker.py
  best1.pt                     fine-tuned person detector
  yolov8n.pt                   stock YOLOv8-nano
outputs/                       created automatically on first run (git-ignored)
  detections.csv               one row per person per frame
  clustered_detections.csv     one row per DBSCAN cluster
  snapshots/                   frames saved with `s` during a live run
```

The annotation tools (`click_points.py`, `extract_annotations.py`) and the
cluster map (`plot_clusters.py`) work on saved PNGs and `detections.csv`, so
they were unaffected by the move to bags.
