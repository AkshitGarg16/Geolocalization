# Command reference

Every runnable script and every argument it accepts.

**Run everything from inside `src/`.** The modules import each other by bare
name (`from bag_source import ...`), so they will not resolve from the repo
root.

```bash
cd ~/geolocalization/src
source /opt/ros/humble/setup.bash     # rosbag2_py, rclpy, sensor_msgs
```

Anything that opens a bag needs ROS 2 on the `PYTHONPATH`. `click_points.py`,
`extract_annotations.py` and `plot_clusters.py` work on saved PNGs and CSVs
only, so they do not.

---

## 1. `geolocalization.py` — the main pipeline

Detect people in the bag's imagery, project each detection to a GPS
coordinate, log every fix and consolidate repeated fixes with DBSCAN.
Opens a live preview paced to the bag's own clock.

```bash
python3 geolocalization.py
python3 geolocalization.py ~/rosbags/25m_1x_2ms
python3 geolocalization.py ~/rosbags/30m_2x_2ms --zoom 2 --roi 0.9
python3 geolocalization.py ~/rosbags/25m_1x_2ms --start 400 --no-preview
python3 geolocalization.py ~/rosbags/25m_1x_2ms --alt-source baro
```

| Argument | Default | What it does |
|---|---|---|
| `bag` (positional, optional) | `CONFIG['bag_path']` | The rosbag2 **directory** (the one holding `metadata.yaml`), not the `.db3` |
| `--zoom VALUE` | `CONFIG['zoom_ratio']` = `auto` | Optical zoom ratio. A number forces it; `auto` reads it from the bag name (`25m_1x_2ms` -> 1x). A wrong zoom scales every coordinate |
| `--roi FLOAT` | `0.8` | Detection ROI as a fraction of the frame. 0.8 = central 80% of width and height. Not a speed control — it trades magnification against edge coverage |
| `--start SECONDS` | `0.0` | Skip this far into the flight. No random access in H.265, so earlier frames are still decoded (just not colour-converted) |
| `--alt-source lidar\|baro` | `lidar` | Height above ground from the downward 2D lidar on `/scan`, or from barometric `rel_alt` alone |
| `--no-preview` | off | Headless run, no OpenCV window |

Everything else is edited in the `CONFIG` dict at the top of the file: topic
names, `max_telemetry_gap_s`, `use_camera_info`, `yolo_model`,
`confidence_threshold`, `realtime`, `playback_speed`, `detect_every_n`,
`display_width`, the three output paths, the lidar sub-dict and the three
DBSCAN parameters.

**Preview window keys**

| Key | Action |
|---|---|
| `space` | pause / resume |
| `,` `.` | step one frame (while paused) |
| `[` `]` | slower / faster |
| `r` | back to 1x |
| `d` | toggle the detector (video keeps playing) |
| `h` | toggle the HUD |
| `s` | save the current frame to `outputs/snapshots/` |
| `q` / `Esc` | quit (buffers are still flushed) |

**Writes:** `outputs/detections.csv` (one row per person per frame) and
`outputs/clustered_detections.csv` (one row per DBSCAN cluster).

---

## 2. `play_video.py` — bag player, saves clean frames

Plays the bag's imagery with frame number, bag time, UTC and telemetry burned
on screen. `SPACE` saves a screenshot **without** the overlay, so the pixel
coordinates you read off it are the real full-frame coordinates.

```bash
python3 play_video.py
python3 play_video.py ~/rosbags/25m_1x_2ms
python3 play_video.py ~/rosbags/25m_1x_2ms -o ../outputs/frames/surya -f 3000
python3 play_video.py ~/rosbags/25m_1x_2ms --cache 200 --zoom 2
```

| Argument | Default | What it does |
|---|---|---|
| `bag` (positional, optional) | `CONFIG['bag_path']` | The rosbag2 directory |
| `-o`, `--out-dir DIR` | `outputs/frames` | Where screenshots are written. Point it at a per-person folder — the folder name becomes that person's label downstream |
| `-f`, `--start-frame N` | `0` | Frame to open at |
| `--cache N` | `60` | Frames held in memory for backward stepping (~370 MB at 1080p). Stepping back beyond the cache reopens the bag and re-decodes forward |
| `--zoom VALUE` | `CONFIG['zoom_ratio']` | Optical zoom, for the telemetry overlay only |

**Keys**

| Key | Action |
|---|---|
| `SPACE` | save a clean screenshot of the current frame |
| `p` | pause / resume |
| `.` `,` | step forward / back one frame (pauses) |
| `>` `<` | step forward / back 10 frames |
| `g` | jump to a frame number (typed in the terminal) |
| `+` `-` | faster / slower |
| `q` / `Esc` | quit |

Screenshots are named `frame_NNNNNN.png` after the image packet's real
position in the bag, which is what `geolocate_pixels.py` turns back into a
capture time.

---

## 3. `click_points.py` — click the target on saved frames

Opens each saved frame; you click the target and the pixel goes straight into
a `frame,u,v,label` CSV. No image editor needed. Frames are never modified.

```bash
python3 click_points.py
python3 click_points.py --dir ../outputs/frames/surya
python3 click_points.py --dir ../outputs/frames --csv-name points.csv
python3 click_points.py --dir ../outputs/frames/surya --label surya_left
```

| Argument | Default | What it does |
|---|---|---|
| `--dir DIR` | `outputs/frames` | Directory of saved frames, searched recursively |
| `--csv-name NAME` | `<dirname>_points.csv` | CSV filename written **inside each frame directory** |
| `--label TEXT` | the sub-directory name | Override the label written to every row |

**Keys**

| Key | Action |
|---|---|
| left click | mark the target (one per image; clicking again moves it) |
| right click | remove this image's mark |
| `n` / `space` | next image |
| `p` | previous image |
| `m` | toggle the 8x magnifier |
| `q` / `Esc` | save and quit |

Saves on every click, so quitting and restarting resumes where you left off.
Writes one CSV per directory, beside that directory's frames.

---

## 4. `extract_annotations.py` — recover pixels from marked-up frames

The alternative to `click_points.py`. Mark a copy of a frame in any image
editor (dot, X, circle, any colour), and this diffs it against the pristine
original to recover the centre of what you drew.

```bash
python3 extract_annotations.py
python3 extract_annotations.py --preview
python3 extract_annotations.py --annotated ../outputs/annotated \
                               --originals ../outputs/frames \
                               -o ../outputs/points.csv
python3 extract_annotations.py --all-marks --label tower
```

| Argument | Default | What it does |
|---|---|---|
| `--annotated DIR` | `outputs/annotated` | Root of your marked copies |
| `--originals DIR` | `outputs/frames` | Root of the pristine frames |
| `-o`, `--output CSV` | `outputs/points.csv` | Annotation CSV to write |
| `--label TEXT` | the sub-directory name | Override the label |
| `--preview` | off | Also write check images showing what was detected |
| `--all-marks` | off | Keep every mark per image, not just the largest |

Save marked copies as **PNG** — JPEG re-compression adds noise everywhere and
wrecks the diff. Layout expected:

```
outputs/frames/surya/frame_003360.png       <- pristine, from play_video.py
outputs/annotated/surya/frame_003360.png    <- your marked copy
```

---

## 5. `geolocate_pixels.py` — project hand-annotated pixels

Runs the same pixel -> GPS projection on annotated points instead of YOLO
detections. This is how the projection gets checked against surveyed ground
truth and how the extrinsics get re-fitted, with the detector out of the loop.

```bash
python3 geolocate_pixels.py ../outputs/points.csv
python3 geolocate_pixels.py ../outputs/frames/*/*_points.csv -o ../outputs/all.csv
python3 geolocate_pixels.py points.csv --ground-truth 28.7537558,77.1155393
python3 geolocate_pixels.py points.csv --extrinsics nominal    # A/B the old matrix
python3 geolocate_pixels.py points.csv --alt-source baro       # A/B the height source
python3 geolocate_pixels.py points.csv --bag ~/rosbags/25m_1x_2ms
```

| Argument | Default | What it does |
|---|---|---|
| `points_csv` (positional, 1+) | required | One or more `frame,u,v[,label]` CSVs. `u,v` are FULL-frame pixels, not ROI-local |
| `-o`, `--output CSV` | one `<input>_geolocated.csv` per input | Combine every input into this one CSV instead |
| `--bag DIR` | `CONFIG['bag_path']` | The rosbag2 directory the frame numbers refer to |
| `--ground-truth LAT,LON` | none | Surveyed position; adds an `error_m` column and a mean/median/min/max/RMSE summary |
| `--extrinsics calibrated\|nominal` | `calibrated` | Which camera->body matrix to use. `nominal` is the un-fitted 90 deg mount, kept for comparison |
| `--alt-source lidar\|baro` | `CONFIG['altitude_source']` | Height above ground from `/scan` or from barometric `rel_alt` |

---

## 6. `plot_clusters.py` — two-stage re-clustering + Bokeh satellite map

Reads `detections.csv`, collapses all detections of the same person into one
GPS point, and renders a standalone interactive map on satellite imagery.
Independent of the pipeline's live DBSCAN — this is the offline pass.

Needs `bokeh` (`pip install bokeh`). Does **not** need ROS 2.

```bash
python3 plot_clusters.py
python3 plot_clusters.py --input ../outputs/detections.csv \
                         --output ../outputs/cluster_map.html \
                         --csv_output ../outputs/clustered.csv
python3 plot_clusters.py --eps 1.5 --min_samples 5 --merge_dist 4
python3 plot_clusters.py --min_cluster_size 40 --min_conf 0.6 --tiles osm
python3 plot_clusters.py --merge_dist 0          # skip stage 2 entirely
```

| Argument | Default | What it does |
|---|---|---|
| `--input CSV` | `../outputs/detections.csv` | Raw per-frame detections to cluster |
| `--output HTML` | `../outputs/cluster_map.html` | Standalone Bokeh map to write |
| `--csv_output CSV` | `../outputs/clustered.csv` | One row per final person |
| `--eps METRES` | `1.0` | Stage-1 DBSCAN radius. Tight, to find dense cores |
| `--min_samples N` | `3` | Minimum detections to form a stage-1 core |
| `--merge_dist METRES` | `5.0` | Stage-2 absorption radius: a below-median cluster within this distance of a larger one is absorbed. `0` skips stage 2 |
| `--min_alt METRES` | `5.0` | Drop detections taken below this drone AGL. Near the ground the ray runs almost parallel to the ground plane and the fix collapses onto the drone's own position |
| `--min_cluster_size N` | `20` | Drop clusters with fewer than N detections |
| `--min_conf FLOAT` | `0.0` | Drop clusters whose best confidence is below this |
| `--tiles esri\|osm` | `esri` | Basemap: Esri World Imagery (satellite, no key) or OpenStreetMap |

---

## 7. `probe_lidar.py` — what the lidar says, bag by bag

Per bag: how many `/scan` messages produced a height, how those heights
compare with the barometric `rel_alt`, and how much of the flight the lidar
can answer for. Read `lidar-baro` as the error the barometer was contributing
— the projection divides by that height, so +1.5 m at 25 m AGL is a 6% scale
error on every ground offset.

```bash
python3 probe_lidar.py ~/rosbags/25m_1x_2ms
python3 probe_lidar.py ~/rosbags/*
python3 probe_lidar.py ~/rosbags/25m_1x_2ms --detail
```

| Argument | Default | What it does |
|---|---|---|
| `bags` (positional, 1+) | required | One or more rosbag2 directories. Non-directories are skipped, so a glob is safe |
| `--detail` | off | Also walk the flight in 10 s steps — the quickest way to spot a stretch where the lidar drops out or saturates (`range_max` is 50 m) |

---

## 8. `probe_zoom.py` — check what zoom a bag reports, and from where

Shows zoom-from-the-bag-name beside zoom-decoded-from-`/fcb/state`, with
`[agree]` / `[DISAGREE]`, so you can decide whether to trust the experimental
`/fcb/state` decoder.

```bash
python3 probe_zoom.py ~/rosbags/25m_1x_2ms
python3 probe_zoom.py ~/rosbags/*
python3 probe_zoom.py ~/rosbags/* --topic /fcb/state
```

| Argument | Default | What it does |
|---|---|---|
| `bags` (positional, 1+) | required | One or more rosbag2 directories |
| `--topic NAME` | `/fcb/state` | Camera state topic to probe |

**Run it on a bag you know is NOT 1x.** A decoder that reads 1.0 on a 1x bag
proves nothing — a field that is always 1.0 is indistinguishable from a field
that is coincidentally 1.0. Only if a 2x bag prints `2x` is it safe to set
`use_camera_state_zoom: True` in `CONFIG`.

---

## 9. `lidar_altimeter.py` — self-check on synthetic scans

No arguments. Runs the height estimator against synthetic scans with a known
25.000 m truth: flat ground, a person at nadir, a person at +8 deg, a window
that misses nadir, a narrow window, 5x the measured noise, and an all-misses
scan. Run it after changing `height_from_scan`.

```bash
python3 lidar_altimeter.py
```

Expect the two "person" cases to still report ~25 m (the outlier rejection is
asymmetric so an object under the drone cannot pull the height down), and the
"window misses nadir" and "every return a miss" cases to be **refused**.

---

## 10. `person_tracker.py` — live visual servoing (separate mode)

No arguments — everything is in its own `CONFIG` dict at the top of the file.
Not part of the offline pipeline: it needs a live camera and a MAVLink link,
and it **commands the aircraft**. Locks onto the largest detected person and
sends body-frame velocity set-points at 20 Hz to hover over them.

```bash
python3 person_tracker.py
```

Edit in `CONFIG`: `camera_source` (int index, `rtsp://…`, or a GStreamer
string), `camera_width`/`camera_height`, `mavlink_connection`
(`udpin:0.0.0.0:14550` or `serial:/dev/ttyUSB0,57600`), `mavlink_dialect`,
`model_path`, `confidence_thresh`, `dead_zone_px`, `max_linear_speed`,
`gain_xy`, `publish_rate`, `show_preview`.

Press `q` or `Esc` in the preview window to stop; it sends an explicit hover
on the way out.

> **Flight safety.** The drone must already be armed, airborne and in GUIDED
> before this starts. Test in SITL first, then fly with a pilot on the sticks
> and a mode switch ready — a velocity set-point stream overrides position
> hold.

> **Use its own MAVLink endpoint.** Two processes bound to the same UDP port
> steal each other's packets. If a GCS already holds 14550, forward the link
> to a second port with MAVProxy or mavlink-router.

---

## Not runnable — imported as libraries

| File | Role |
|---|---|
| `coordinate_transformer.py` | The pixel -> camera ray -> body -> NED -> lat/lon projection chain, plus the fitted extrinsics |
| `bag_source.py` | rosbag2 I/O: FFMPEGPacket decode, telemetry interpolation, frame-number <-> capture-time clock, zoom resolution |
| `yolo_detector.py` | YOLOE person detector, ROI cropping, bounding-box/GPS overlay |

---

## Typical end-to-end run

```bash
cd ~/geolocalization/src
source /opt/ros/humble/setup.bash

# 0. Sanity-check the bag before trusting anything it produces
python3 probe_lidar.py ~/rosbags/25m_1x_2ms
python3 probe_zoom.py  ~/rosbags/25m_1x_2ms

# 1. Detect and geolocalise
python3 geolocalization.py ~/rosbags/25m_1x_2ms

# 2. Consolidate and map
python3 plot_clusters.py
xdg-open ../outputs/cluster_map.html
```

## Typical ground-truth / calibration run

```bash
cd ~/geolocalization/src
source /opt/ros/humble/setup.bash

# 1. Grab clean frames of the surveyed person (SPACE saves)
python3 play_video.py ~/rosbags/25m_1x_2ms -o ../outputs/frames/surya -f 2800

# 2. Mark the target on each one
python3 click_points.py --dir ../outputs/frames/surya
#    ...or mark copies in an image editor and recover them instead:
#    python3 extract_annotations.py --preview

# 3. Project them and score against the survey
python3 geolocate_pixels.py ../outputs/frames/surya/surya_points.csv \
        --ground-truth 28.7537558,77.1155393

# 4. A/B the two things most likely to be wrong
python3 geolocate_pixels.py ../outputs/frames/surya/surya_points.csv \
        --ground-truth 28.7537558,77.1155393 --extrinsics nominal
python3 geolocate_pixels.py ../outputs/frames/surya/surya_points.csv \
        --ground-truth 28.7537558,77.1155393 --alt-source baro
```
