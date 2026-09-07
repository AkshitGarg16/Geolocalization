"""
Person Geolocalisation — standalone (no ROS)
=============================================
Data sources
  - Video   : recorded fcb-*.mp4 (cv2.VideoCapture)
  - GPS/IMU : matching fcb-*.csv, one row per video frame

The recorder writes the .mp4 and the .csv together, frame-for-frame, so
telemetry is looked up by frame index — no clock synchronisation needed.
The CSV also carries the camera's optical zoom_ratio per frame, which is
applied to the focal length during the pixel -> GPS projection.

All ROS2 / MAVROS / cv_bridge / message_filters / pymavlink dependencies
have been removed.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from yolo_detector import YOLODetector
from coordinate_transformer import CoordinateTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('geolocalization')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0

CONFIG = {
    # --- Recorded video + telemetry (one CSV row per video frame) ---
    'video_source':          '/home/akshit/Downloads/fcb-20260903-175559.mp4',
    'telemetry_csv':         '/home/akshit/Downloads/fcb-20260903-175559.csv',

    # --- Optical zoom ---
    # Zoom ratio the intrinsic matrix was calibrated at. The CSV's zoom_ratio
    # is divided by this to get the focal-length scale factor for each frame.
    'zoom_baseline':         1.0,

    # --- YOLO ---
    'yolo_model':            '/home/akshit/geolocalization/models/best1.pt',
    'confidence_threshold':  0.5,

    # --- Output ---
    'csv_output_path':       '/home/akshit/geolocalization/outputs/detections.csv',
    'clustered_csv_path':    '/home/akshit/geolocalization/outputs/clustered_detections.csv',
    'show_preview':          True,        # display cv2 window

    # --- DBSCAN clustering ---
    'cluster_eps_metres':    2.0,
    'cluster_min_samples':   3,
    'cluster_jump_metres':   3.0,
}

# ---------------------------------------------------------------------------
# Helpers – geo maths
# ---------------------------------------------------------------------------

def haversine_metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def latlon_to_xy(lats, lons):
    lats  = np.asarray(lats, dtype=np.float64)
    lons  = np.asarray(lons, dtype=np.float64)
    mlat  = lats.mean()
    mlon  = lons.mean()
    x     = np.radians(lons - mlon) * EARTH_RADIUS_M * math.cos(math.radians(mlat))
    y     = np.radians(lats - mlat) * EARTH_RADIUS_M
    return np.column_stack([x, y]), mlat, mlon


def xy_to_latlon(x: float, y: float, mean_lat: float, mean_lon: float):
    lat = mean_lat + math.degrees(y / EARTH_RADIUS_M)
    lon = mean_lon + math.degrees(
        x / (EARTH_RADIUS_M * math.cos(math.radians(mean_lat)))
    )
    return lat, lon

# ---------------------------------------------------------------------------
# Detection buffer + DBSCAN
# ---------------------------------------------------------------------------

class DetectionBuffer:

    def __init__(self, eps_metres: float = 2.0, min_samples: int = 3):
        self.eps_metres  = eps_metres
        self.min_samples = min_samples
        self._lats:  list[float] = []
        self._lons:  list[float] = []
        self._times: list[float] = []
        self._confs: list[float] = []

    def append(self, lat: float, lon: float, timestamp: float, confidence: float):
        self._lats.append(lat)
        self._lons.append(lon)
        self._times.append(timestamp)
        self._confs.append(confidence)

    @property
    def last_point(self) -> tuple[float, float] | None:
        if not self._lats:
            return None
        return self._lats[-1], self._lons[-1]

    def __len__(self):
        return len(self._lats)

    def flush(self) -> list[dict]:
        results = []
        if len(self._lats) >= self.min_samples:
            xy, mlat, mlon = latlon_to_xy(self._lats, self._lons)
            db     = DBSCAN(eps=self.eps_metres, min_samples=self.min_samples,
                            metric='euclidean').fit(xy)
            labels = db.labels_
            for label in set(labels):
                if label == -1:
                    continue
                mask       = labels == label
                cx         = xy[mask, 0].mean()
                cy         = xy[mask, 1].mean()
                clat, clon = xy_to_latlon(cx, cy, mlat, mlon)
                confs      = np.asarray(self._confs)[mask]
                times      = np.asarray(self._times)[mask]
                results.append({
                    'lat':             clat,
                    'lon':             clon,
                    'count':           int(mask.sum()),
                    'mean_confidence': float(confs.mean()),
                    'first_ts':        float(times.min()),
                    'last_ts':         float(times.max()),
                })
        self._clear()
        return results

    def _clear(self):
        self._lats.clear()
        self._lons.clear()
        self._times.clear()
        self._confs.clear()

# ---------------------------------------------------------------------------
# CSV telemetry reader  (replaces the MAVLink stream)
# ---------------------------------------------------------------------------

@dataclass
class TelemetrySample:
    """One row of the recorder's per-frame telemetry CSV."""
    frame:    int
    ts:       float                 # seconds (UTC epoch, or t_mono_s as fallback)
    lat:      float | None = None   # degrees
    lon:      float | None = None   # degrees
    alt_agl:  float | None = None   # metres above launch (alt_rel_m)
    roll:     float | None = None   # radians
    pitch:    float | None = None   # radians
    yaw:      float | None = None   # radians
    compass:  float | None = None   # degrees (raw heading)
    zoom:     float        = 1.0    # optical zoom ratio reported by the camera

    def is_complete(self) -> bool:
        return not any(v is None for v in
                       (self.lat, self.lon, self.alt_agl,
                        self.roll, self.pitch, self.yaw))


def _as_float(row: dict, key: str) -> float | None:
    """Parse a CSV field, returning None when it is missing/blank/non-numeric."""
    raw = row.get(key)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return None if math.isnan(value) else value


def _parse_wall_time(raw: str | None) -> float | None:
    """'2026-09-03T12:25:59.300+00:00' -> UTC epoch seconds."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.strip()).timestamp()
    except ValueError:
        return None


class CSVTelemetryReader:
    """
    Loads the recorder's fcb-*.csv into memory, indexed by frame number.

    Expected columns (extra columns are ignored):
      frame, t_wall_utc, t_mono_s,
      lat_deg, lon_deg, alt_rel_m,
      roll_deg, pitch_deg, yaw_deg, heading_deg,
      zoom_ratio
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._samples: dict[int, TelemetrySample] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f'Telemetry CSV not found: {self.csv_path}')

        with open(self.csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = _as_float(row, 'frame')
                if frame is None:
                    continue
                frame = int(frame)

                ts = _parse_wall_time(row.get('t_wall_utc'))
                if ts is None:
                    ts = _as_float(row, 't_mono_s')
                if ts is None:
                    ts = 0.0

                roll_deg  = _as_float(row, 'roll_deg')
                pitch_deg = _as_float(row, 'pitch_deg')
                yaw_deg   = _as_float(row, 'yaw_deg')
                zoom      = _as_float(row, 'zoom_ratio')

                self._samples[frame] = TelemetrySample(
                    frame=frame,
                    ts=ts,
                    lat=_as_float(row, 'lat_deg'),
                    lon=_as_float(row, 'lon_deg'),
                    alt_agl=_as_float(row, 'alt_rel_m'),
                    roll=None  if roll_deg  is None else math.radians(roll_deg),
                    pitch=None if pitch_deg is None else math.radians(pitch_deg),
                    yaw=None   if yaw_deg   is None else math.radians(yaw_deg),
                    compass=_as_float(row, 'heading_deg'),
                    zoom=1.0 if zoom is None or zoom <= 0 else zoom,
                )

        if not self._samples:
            raise ValueError(f'No usable rows in telemetry CSV: {self.csv_path}')

        log.info(f'Telemetry CSV loaded: {len(self._samples)} row(s) from '
                 f'{self.csv_path} (frames {min(self._samples)}–{max(self._samples)})')

    def __len__(self):
        return len(self._samples)

    def get(self, frame_index: int) -> TelemetrySample | None:
        """Telemetry for a video frame, or None if that frame has no row."""
        return self._samples.get(frame_index)

# ---------------------------------------------------------------------------
# CSV helpers  (unchanged logic, removed ROS logger)
# ---------------------------------------------------------------------------

def _ensure_parent_dir(path: str):
    """Create the directory the CSV lives in, so a fresh clone can just run."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def init_csv(path: str):
    _ensure_parent_dir(path)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'timestamp_sec', 'person_idx',
            'person_lat', 'person_lon',
            'drone_lat', 'drone_lon', 'drone_altitude_agl',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'confidence', 'num_detections_in_frame',
        ])


def init_clustered_csv(path: str):
    _ensure_parent_dir(path)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'person_slot', 'cluster_id',
            'cluster_lat', 'cluster_lon',
            'num_points', 'mean_confidence',
            'first_detection_ts', 'last_detection_ts',
        ])


def log_detection(path, timestamp_sec, person_idx, person_lat, person_lon,
                  drone_lat, drone_lon, drone_alt, roll, pitch, yaw,
                  confidence, num_detections):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([
            f'{timestamp_sec:.9f}', person_idx,
            f'{person_lat:.9f}',    f'{person_lon:.9f}',
            f'{drone_lat:.9f}',     f'{drone_lon:.9f}', f'{drone_alt:.3f}',
            f'{np.rad2deg(roll):.4f}', f'{np.rad2deg(pitch):.4f}',
            f'{np.rad2deg(yaw):.4f}',
            f'{confidence:.4f}', num_detections,
        ])


def log_clusters(path: str, slot: int, clusters: list[dict]):
    if not clusters:
        return
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        for cid, c in enumerate(clusters):
            writer.writerow([
                slot, cid,
                f"{c['lat']:.9f}",           f"{c['lon']:.9f}",
                c['count'],                   f"{c['mean_confidence']:.4f}",
                f"{c['first_ts']:.9f}",       f"{c['last_ts']:.9f}",
            ])
    log.info(f'[DBSCAN] slot={slot}: flushed {len(clusters)} cluster(s)')

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class PersonLocalisationPipeline:

    def __init__(self, cfg: dict = CONFIG):
        self.cfg            = cfg
        self.zoom_baseline  = cfg.get('zoom_baseline', 1.0) or 1.0
        self.jump_thresh_m  = cfg['cluster_jump_metres']
        self.csv_path       = cfg['csv_output_path']
        self.clustered_csv  = cfg['clustered_csv_path']

        self.intrinsic_matrix = np.array([
            [393.12779563, 0.0,          321.48263787],
            [0.0,          394.76440916, 241.58044155],
            [0.0,          0.0,          1.0         ]
        ])

        self.detector    = YOLODetector(
            model_name=cfg['yolo_model'],
            confidence_threshold=cfg['confidence_threshold'],
        )
        self.transformer = CoordinateTransformer(self.intrinsic_matrix)
        log.info('YOLO model loaded ✓')

        init_csv(self.csv_path)
        init_clustered_csv(self.clustered_csv)

        self._buffers: dict[int, DetectionBuffer] = {}
        self._dbscan_eps         = cfg['cluster_eps_metres']
        self._dbscan_min_samples = cfg['cluster_min_samples']

        # --- Recorded telemetry (one row per video frame) ---
        self._telem_reader = CSVTelemetryReader(cfg['telemetry_csv'])

    # ------------------------------------------------------------------ #
    # Telemetry helpers                                                    #
    # ------------------------------------------------------------------ #

    def _telemetry_for_frame(self, frame_index: int) -> TelemetrySample | None:
        """Telemetry row for this video frame, or None if unusable."""
        telem = self._telem_reader.get(frame_index)
        if telem is None:
            log.warning(f'No telemetry row for frame {frame_index} — skipping frame')
            return None
        if not telem.is_complete():
            log.warning(f'Incomplete telemetry at frame {frame_index} — skipping frame')
            return None
        return telem

    # ------------------------------------------------------------------ #
    # Per-frame processing                                                 #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray, frame_index: int):
        telem = self._telemetry_for_frame(frame_index)
        if telem is None:
            return frame    # return unmodified frame

        timestamp_sec = telem.ts
        drone_lat  = telem.lat
        drone_lon  = telem.lon
        altitude   = telem.alt_agl
        roll       = telem.roll
        pitch      = telem.pitch
        yaw        = telem.yaw
        compass    = telem.compass

        # Focal-length scale factor for this frame's optical zoom.
        zoom_factor = telem.zoom / self.zoom_baseline

        detections = self.detector.detect_persons(frame)
        if not detections:
            return self._annotate_hud(frame, altitude, roll, pitch, yaw, compass,
                                      telem.zoom, 0)

        log.info(f'Detected {len(detections)} person(s) at t={timestamp_sec:.3f}s')

        gps_coords = []

        for idx, det in enumerate(detections):
            cx, cy     = det['center']
            confidence = det['confidence']

            result = self.transformer.pixel_to_gps(
                u=cx, v=cy,
                drone_lat=drone_lat, drone_lon=drone_lon,
                drone_altitude=altitude,
                roll=roll, pitch=pitch, yaw=yaw,
                ground_altitude=0.0,
                zoom=zoom_factor,
            )

            if result is None:
                gps_coords.append((None, None))
                log.warning(f'  Could not project GPS for person {idx + 1}')
                continue

            target_lat, target_lon = result
            gps_coords.append((target_lat, target_lon))

            log_detection(
                self.csv_path, timestamp_sec, idx + 1,
                target_lat, target_lon,
                drone_lat, drone_lon, altitude,
                roll, pitch, yaw, confidence, len(detections),
            )

            log.info(f'  Person {idx + 1}: lat={target_lat:.6f}, lon={target_lon:.6f}')

            # --- DBSCAN buffer ---
            buf = self._buffers.get(idx)
            if buf is None:
                buf = DetectionBuffer(
                    eps_metres=self._dbscan_eps,
                    min_samples=self._dbscan_min_samples,
                )
                self._buffers[idx] = buf

            last = buf.last_point
            if last is not None:
                dist = haversine_metres(last[0], last[1], target_lat, target_lon)
                if dist > self.jump_thresh_m:
                    log.info(
                        f'  Slot {idx}: jump of {dist:.1f}m — '
                        f'flushing {len(buf)} point(s) to DBSCAN'
                    )
                    clusters = buf.flush()
                    log_clusters(self.clustered_csv, idx, clusters)

            buf.append(target_lat, target_lon, timestamp_sec, confidence)

        vis = self.detector.draw_detections(frame, detections, gps_coords)
        return self._annotate_hud(vis, altitude, roll, pitch, yaw, compass,
                                  telem.zoom, len(detections))

    @staticmethod
    def _annotate_hud(frame, altitude, roll, pitch, yaw, compass, zoom, n_det):
        overlay = frame.copy()
        lines = [
            f"Alt: {altitude:.1f}m AGL",
            f"Roll:  {np.rad2deg(roll):.1f} deg",
            f"Pitch: {np.rad2deg(pitch):.1f} deg",
            f"Yaw:   {np.rad2deg(yaw):.1f} deg",
            f"Hdg:   {compass:.1f} deg",
            f"Zoom:  {zoom:.1f}x",
            f"Detections: {n_det}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(overlay, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return overlay

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        src = self.cfg['video_source']
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error(f'Cannot open video source: {src}')
            return

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info(f'Video opened: {src} ({n_frames} frames)')
        if n_frames > 0 and n_frames != len(self._telem_reader):
            log.warning(
                f'Video has {n_frames} frames but the telemetry CSV has '
                f'{len(self._telem_reader)} row(s) — frames without a row are skipped'
            )

        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.info('End of video — exiting loop')
                    break

                vis = self.process_frame(frame, frame_index)
                frame_index += 1

                if self.cfg.get('show_preview', True):
                    cv2.imshow('Person Geolocalisation', vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        log.info('User requested quit.')
                        break

        except KeyboardInterrupt:
            log.info('KeyboardInterrupt — shutting down.')
        finally:
            self._shutdown(cap)

    def _shutdown(self, cap: cv2.VideoCapture):
        log.info('Flushing all detection buffers …')
        for slot, buf in self._buffers.items():
            if len(buf) > 0:
                clusters = buf.flush()
                log_clusters(self.clustered_csv, slot, clusters)

        cap.release()
        cv2.destroyAllWindows()
        log.info('Shutdown complete.')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pipeline = PersonLocalisationPipeline(CONFIG)
    pipeline.run()
