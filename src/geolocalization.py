"""
Person Geolocalisation — standalone (no ROS)
=============================================
Data sources
  - Camera  : cv2.VideoCapture  (USB / RTSP / GStreamer)
  - GPS/IMU : pymavlink MAVLink stream (serial, UDP, or TCP)

All ROS2 / MAVROS / cv_bridge / message_filters dependencies have been removed.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
from pymavlink import mavutil

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
# Constants / default config
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0

CONFIG = {
    # --- Camera ---
    'camera_source':         0,           # int index, or 'rtsp://...' / 'udp://' etc.
    'camera_width':          640,
    'camera_height':         480,
    'camera_fps':            30,

    # --- MAVLink ---
    'mavlink_connection':    'udpin:0.0.0.0:14550',   # serial: '/dev/ttyUSB0,57600'
    'mavlink_dialect':       'ardupilotmega',

    # --- YOLO ---
    'yolo_model':            'best1.pt',
    'confidence_threshold':  0.5,

    # --- Output ---
    'csv_output_path':       'detections.csv',
    'clustered_csv_path':    'clustered_detections.csv',
    'show_preview':          True,        # display cv2 window

    # --- Synchronisation ---
    'max_cache_age':         0.5,         # seconds – discard stale telemetry

    # --- DBSCAN clustering ---
    'cluster_eps_metres':    2.0,
    'cluster_min_samples':   3,
    'cluster_jump_metres':   3.0,
}

# ---------------------------------------------------------------------------
# Helpers – geo maths (unchanged from original)
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
# Detection buffer + DBSCAN (unchanged logic)
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
# MAVLink telemetry reader  (replaces all ROS subscribers)
# ---------------------------------------------------------------------------

@dataclass
class TelemetryState:
    """Thread-safe snapshot of the latest MAVLink telemetry."""
    lat:      float | None = None   # degrees
    lon:      float | None = None   # degrees
    alt_agl:  float | None = None   # metres above ground
    roll:     float | None = None   # radians
    pitch:    float | None = None   # radians
    yaw:      float | None = None   # radians (from compass heading)
    compass:  float | None = None   # degrees (raw heading)
    ts:       float        = field(default_factory=time.monotonic)
    _lock:    threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.ts = time.monotonic()

    def snapshot(self) -> 'TelemetryState':
        with self._lock:
            import copy
            return copy.copy(self)


class MAVLinkReader(threading.Thread):
    """
    Background thread that consumes MAVLink messages and fills a TelemetryState.

    Messages used:
      GLOBAL_POSITION_INT  → lat, lon, relative altitude, yaw (compass)
      ATTITUDE             → roll, pitch  (used for perspective correction)
    """

    def __init__(self, connection_string: str, state: TelemetryState,
                 dialect: str = 'ardupilotmega'):
        super().__init__(daemon=True, name='mavlink-reader')
        self.connection_string = connection_string
        self.state             = state
        self.dialect           = dialect
        self._stop_event       = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        log.info(f'MAVLink connecting → {self.connection_string}')
        try:
            mav = mavutil.mavlink_connection(
                self.connection_string,
                dialect=self.dialect,
            )
            mav.wait_heartbeat(timeout=10)
            log.info('MAVLink heartbeat received ✓')
        except Exception as exc:
            log.error(f'MAVLink connection failed: {exc}')
            return

        # Request the two message streams we need at ~10 Hz
        mav.mav.request_data_stream_send(
            mav.target_system,
            mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,   # GLOBAL_POSITION_INT
            10, 1,
        )
        mav.mav.request_data_stream_send(
            mav.target_system,
            mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,     # ATTITUDE
            10, 1,
        )

        while not self._stop_event.is_set():
            msg = mav.recv_match(
                type=['GLOBAL_POSITION_INT', 'ATTITUDE'],
                blocking=True,
                timeout=1.0,
            )
            if msg is None:
                continue

            mtype = msg.get_type()

            if mtype == 'GLOBAL_POSITION_INT':
                # lat/lon are in 1e-7 degrees; relative_alt in mm
                lat     = msg.lat  * 1e-7
                lon     = msg.lon  * 1e-7
                alt_agl = msg.relative_alt * 1e-3          # mm → m
                hdg_deg = msg.hdg * 1e-2                   # cdeg → deg
                self.state.update(
                    lat=lat,
                    lon=lon,
                    alt_agl=alt_agl,
                    compass=hdg_deg,
                    yaw=math.radians(hdg_deg),
                )

            elif mtype == 'ATTITUDE':
                self.state.update(
                    roll=msg.roll,     # already radians
                    pitch=msg.pitch,   # already radians
                )

# ---------------------------------------------------------------------------
# CSV helpers  (unchanged logic, removed ROS logger)
# ---------------------------------------------------------------------------

def init_csv(path: str):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'timestamp_sec', 'person_idx',
            'person_lat', 'person_lon',
            'drone_lat', 'drone_lon', 'drone_altitude_agl',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'confidence', 'num_detections_in_frame',
        ])


def init_clustered_csv(path: str):
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
        self.max_cache_age  = cfg['max_cache_age']
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

        # --- MAVLink telemetry ---
        self._telem = TelemetryState()
        self._mav_reader = MAVLinkReader(
            connection_string=cfg['mavlink_connection'],
            state=self._telem,
            dialect=cfg['mavlink_dialect'],
        )
        self._mav_reader.start()

    # ------------------------------------------------------------------ #
    # Telemetry helpers                                                    #
    # ------------------------------------------------------------------ #

    def _check_cache_age(self, label: str) -> bool:
        """Return True if the telemetry timestamp is fresh enough."""
        age = time.monotonic() - self._telem.ts
        if age > self.max_cache_age:
            log.warning(f'Telemetry is {age:.2f}s old (max {self.max_cache_age}s) — skipping frame')
            return False
        return True

    def _telemetry_ready(self) -> bool:
        t = self._telem
        if any(v is None for v in [t.lat, t.lon, t.alt_agl, t.roll, t.pitch, t.yaw]):
            log.warning('Waiting for full telemetry …', )
            return False
        return self._check_cache_age('telemetry')

    # ------------------------------------------------------------------ #
    # Per-frame processing                                                 #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray, timestamp_sec: float):
        if not self._telemetry_ready():
            return frame    # return unmodified frame

        telem = self._telem.snapshot()
        drone_lat  = telem.lat
        drone_lon  = telem.lon
        altitude   = telem.alt_agl
        roll       = telem.roll
        pitch      = telem.pitch
        yaw        = telem.yaw
        compass    = telem.compass

        detections = self.detector.detect_persons(frame)
        if not detections:
            return self._annotate_hud(frame, altitude, roll, pitch, yaw, compass, 0)

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
                                  len(detections))

    @staticmethod
    def _annotate_hud(frame, altitude, roll, pitch, yaw, compass, n_det):
        overlay = frame.copy()
        lines = [
            f"Alt: {altitude:.1f}m AGL",
            f"Roll:  {np.rad2deg(roll):.1f} deg",
            f"Pitch: {np.rad2deg(pitch):.1f} deg",
            f"Yaw:   {np.rad2deg(yaw):.1f} deg",
            f"Hdg:   {compass:.1f} deg",
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
        src = self.cfg['camera_source']
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error(f'Cannot open camera source: {src}')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg['camera_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg['camera_height'])
        cap.set(cv2.CAP_PROP_FPS,          self.cfg['camera_fps'])
        log.info(f'Camera opened: {src}')

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.error('Camera read failed — exiting loop')
                    break

                timestamp_sec = time.time()
                vis = self.process_frame(frame, timestamp_sec)

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

        self._mav_reader.stop()
        cap.release()
        cv2.destroyAllWindows()
        log.info('Shutdown complete.')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pipeline = PersonLocalisationPipeline(CONFIG)
    pipeline.run()
