"""
Person Geolocalisation — live, straight off a ROS 2 bag
=======================================================
Single input: a rosbag2 directory. Imagery and telemetry both come out of the
.db3; there is no sidecar video file and no per-frame telemetry CSV any more.

    python3 geolocalization.py                        # bag from CONFIG
    python3 geolocalization.py ~/rosbags/25m_1x_2ms   # or name one

Frames are decoded, detected on and shown as they stream, paced against the
bag's own clock so playback runs at the speed the flight did. What the model
sees is what you see: the detection ROI, every box it returns, and the ground
position each box projects to.

    space  pause / resume          , .  step one frame (while paused)
    [ ]    slower / faster         r    back to 1x
    d      toggle detection        h    toggle the HUD
    s      save the frame to outputs/snapshots/
    q      quit

Why the bag and not the .mp4 sitting next to it: see bag_source.BagVideo.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from bag_source import (DEFAULT_MAX_GAP_S, BagSource, TelemetrySample,
                        camera_info_matrix, resolve_zoom)
from coordinate_transformer import CoordinateTransformer
from yolo_detector import YOLODetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('geolocalization')

EARTH_RADIUS_M = 6_371_000.0

CONFIG = {
    # --- Input: one rosbag2 directory, nothing else ---
    'bag_path':              '/home/akshit/rosbags/25m_1x_2ms',
    'topics': {
        'image':        '/fcb/image/ffmpeg',
        'camera_info':  '/fcb/camera_info',
        'camera_state': '/fcb/state',
        'gps':          '/mavros/global_position/global',
        'rel_alt':      '/mavros/global_position/rel_alt',
        'imu':          '/mavros/imu/data',
        'scan':         '/scan',
    },

    # --- Height above ground ---
    # 'lidar' takes the height from the downward-facing 2D lidar on /scan,
    # which measures the distance to the ground; 'baro' uses rel_alt alone,
    # which is a barometric height above the launch point and on these bags
    # reads 0.9-1.8 m low. The projection divides by this number, so the
    # difference is a few per cent on every ground offset. Where the lidar is
    # silent, the barometer stands in shifted by the flight's measured
    # lidar-barometer offset. See lidar_altimeter.py.
    'altitude_source':       'lidar',   # 'lidar' | 'baro'
    'lidar': {
        'max_gap_s':       0.6,   # refuse to interpolate a longer lidar hole
        'half_window_deg': 10.0,  # beams around nadir that are fitted
        'min_beams':       8,     # fewer than this in the window: no estimate
        'mount_offset_m':  0.0,   # lidar-to-camera height offset, if measured
    },

    # --- Optical zoom ---
    # 'auto' reads the zoom out of the bag directory name (`25m_1x_2ms` -> 1x),
    # so a folder of bags shot at different zooms needs no per-bag edit. Give a
    # number instead to force it, or pass --zoom on the command line. See
    # bag_source.resolve_zoom for where the value comes from and why
    # /fcb/state is not simply read.
    'zoom_ratio':            'auto',
    'zoom_baseline':         1.0,
    # Also try to recover the zoom from /fcb/state. Experimental and unverified
    # against a non-1x bag — see bag_source.zoom_from_camera_state.
    'use_camera_state_zoom': False,

    # --- Time sync ---
    # Every topic is keyed on its header stamp (the capture clock), not on bag
    # receive-time; see bag_source's module docstring. Telemetry is never
    # interpolated across a hole wider than this — the frame is shown but not
    # projected, rather than projected against an invented position.
    'max_telemetry_gap_s':   0.25,

    # --- Intrinsics ---
    # The bag's /fcb/camera_info carries fx=fy=1545, cx=960, cy=540 — nominal
    # placeholders (dead-centre principal point, round focal length), constant
    # for the whole flight. The matrix below is the real calibration and is
    # what the fitted extrinsics in coordinate_transformer.py are matched to.
    # Flip to True only once camera_info holds a genuine calibration.
    'use_camera_info':       False,

    # --- YOLO ---
    'yolo_model':            '/home/akshit/geolocalization/models/yoloe-26x-seg.pt',
    'confidence_threshold':  0.5,
    'roi_fraction':          0.8,

    # --- Live playback ---
    'realtime':              True,   # pace to the bag clock; False = as fast as it runs
    'playback_speed':        1.0,
    'detect_every_n':        1,      # detect on every Nth frame, display all
    'start_offset_s':        0.0,    # skip this many seconds in
    'display_width':         1280,   # preview downscale; None for native 1920
    'show_preview':          True,

    # --- Output (set to None to disable) ---
    'csv_output_path':       '/home/akshit/geolocalization/outputs/detections.csv',
    'clustered_csv_path':    '/home/akshit/geolocalization/outputs/clustered_detections.csv',
    'snapshot_dir':          '/home/akshit/geolocalization/outputs/snapshots',

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
# CSV output
# ---------------------------------------------------------------------------

def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def init_csv(path: str):
    if not path:
        return
    _ensure_parent_dir(path)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'timestamp_sec', 'frame', 'person_idx',
            'person_lat', 'person_lon',
            'drone_lat', 'drone_lon', 'drone_altitude_agl',
            'altitude_source', 'altitude_lidar', 'altitude_baro',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'confidence', 'num_detections_in_frame',
        ])


def init_clustered_csv(path: str):
    if not path:
        return
    _ensure_parent_dir(path)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'person_slot', 'cluster_id',
            'cluster_lat', 'cluster_lon',
            'num_points', 'mean_confidence',
            'first_detection_ts', 'last_detection_ts',
        ])


def log_detection(path, timestamp_sec, frame, person_idx, person_lat, person_lon,
                  telem, confidence, num_detections):
    if not path:
        return
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([
            f'{timestamp_sec:.9f}', frame, person_idx,
            f'{person_lat:.9f}',    f'{person_lon:.9f}',
            f'{telem.lat:.9f}',     f'{telem.lon:.9f}', f'{telem.alt_agl:.3f}',
            telem.alt_source,
            '' if telem.alt_lidar is None else f'{telem.alt_lidar:.3f}',
            '' if telem.alt_baro  is None else f'{telem.alt_baro:.3f}',
            f'{np.rad2deg(telem.roll):.4f}', f'{np.rad2deg(telem.pitch):.4f}',
            f'{np.rad2deg(telem.yaw):.4f}',
            f'{confidence:.4f}', num_detections,
        ])


def log_clusters(path: str, slot: int, clusters: list[dict]):
    if not (path and clusters):
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
# Live viewer
# ---------------------------------------------------------------------------

class Viewer:
    """Preview window, HUD and keyboard control for the live run."""

    WINDOW = 'Person Geolocalisation — live from bag'

    def __init__(self, cfg: dict):
        self.cfg           = cfg
        self.enabled       = cfg.get('show_preview', True)
        self.display_width = cfg.get('display_width')
        self.speed         = float(cfg.get('playback_speed', 1.0) or 1.0)
        self.detect        = True
        self.show_hud      = True
        self.paused        = False
        self.step_once     = False
        self.quit          = False
        self.snapshot_dir  = cfg.get('snapshot_dir')
        self._saved        = 0
        if self.enabled:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            if self.display_width:
                cv2.resizeWindow(self.WINDOW, self.display_width,
                                 int(self.display_width * 9 / 16))

    # -- HUD ----------------------------------------------------------------

    @staticmethod
    def _text(img, lines, x=10, y0=30, colour=(255, 255, 255)):
        for i, line in enumerate(lines):
            origin = (x, y0 + i * 25)
            cv2.putText(img, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2, cv2.LINE_AA)

    def draw_hud(self, img, frame_index, t_rel, telem, n_det, fps, lag):
        if not self.show_hud:
            return img
        if telem is None:
            self._text(img, [f'Frame: {frame_index}', 'No telemetry'],
                       colour=(0, 0, 255))
            return img

        self._text(img, [
            f'Frame: {frame_index}   t={t_rel:7.2f}s',
            f'Lat:   {telem.lat:.7f}',
            f'Lon:   {telem.lon:.7f}',
            f'Alt:   {telem.alt_agl:.1f} m AGL ({telem.alt_source})',
            f'Roll:  {np.rad2deg(telem.roll):7.1f} deg',
            f'Pitch: {np.rad2deg(telem.pitch):7.1f} deg',
            f'Yaw:   {np.rad2deg(telem.yaw):7.1f} deg',
            f'Hdg:   {telem.compass:7.1f} deg',
            f'Zoom:  {telem.zoom:.1f}x',
            f'Detections: {n_det}',
        ])

        status = [
            f'{fps:5.1f} FPS   x{self.speed:g}' + ('  [PAUSED]' if self.paused else ''),
            f'detector: {"on" if self.detect else "OFF"}',
        ]
        if lag > 0.5:
            status.append(f'behind bag clock by {lag:.1f}s')
        h, w = img.shape[:2]
        self._text(img, status, x=w - 430, y0=h - 60, colour=(0, 255, 255))
        return img

    # -- window -------------------------------------------------------------

    def show(self, img):
        if not self.enabled:
            return
        if self.display_width and img.shape[1] != self.display_width:
            scale = self.display_width / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        cv2.imshow(self.WINDOW, img)

    def handle_keys(self, raw_frame, frame_index) -> None:
        """Drain keyboard input; blocks while paused."""
        if not self.enabled:
            return
        while True:
            key = cv2.waitKey(0 if self.paused else 1) & 0xFF

            if key in (ord('q'), 27):
                self.quit = True
                return
            if key == ord(' '):
                self.paused = not self.paused
                if not self.paused:
                    return
                continue
            if key in (ord(','), ord('.')) and self.paused:
                return                                   # advance one frame
            if key == ord('['):
                self.speed = max(0.1, self.speed / 2)
            elif key == ord(']'):
                self.speed = min(16.0, self.speed * 2)
            elif key == ord('r'):
                self.speed = 1.0
            elif key == ord('d'):
                self.detect = not self.detect
            elif key == ord('h'):
                self.show_hud = not self.show_hud
            elif key == ord('s'):
                self._snapshot(raw_frame, frame_index)

            if not self.paused:
                return

    def _snapshot(self, frame, frame_index):
        if not self.snapshot_dir:
            return
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, f'frame_{frame_index:06d}.png')
        cv2.imwrite(path, frame)
        self._saved += 1
        log.info(f'Saved {path}')

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class PersonLocalisationPipeline:

    # Longest the player will wait for the next frame before treating the gap
    # as a recording stall rather than as being ahead of schedule.
    MAX_SLEEP_S = 1.0

    CALIBRATED_K = np.array([
        [1584.02798,    0,        953.27546],
        [   0,       1581.52473,  518.97355],
        [   0,          0,          1       ]
    ])

    def __init__(self, cfg: dict = CONFIG):
        self.cfg           = cfg
        self.bag_path      = cfg['bag_path']
        self.zoom_baseline = cfg.get('zoom_baseline', 1.0) or 1.0
        self.jump_thresh_m = cfg['cluster_jump_metres']
        self.csv_path      = cfg.get('csv_output_path')
        self.clustered_csv = cfg.get('clustered_csv_path')

        self.intrinsic_matrix = self._resolve_intrinsics()

        self.detector = YOLODetector(
            model_name=cfg['yolo_model'],
            confidence_threshold=cfg['confidence_threshold'],
            roi_fraction=cfg.get('roi_fraction', 0.8),
        )
        self.transformer = CoordinateTransformer(self.intrinsic_matrix)
        log.info('YOLO model loaded ✓')

        init_csv(self.csv_path)
        init_clustered_csv(self.clustered_csv)

        self._buffers: dict[int, DetectionBuffer] = {}
        self._dbscan_eps         = cfg['cluster_eps_metres']
        self._dbscan_min_samples = cfg['cluster_min_samples']

        self.zoom_ratio = resolve_zoom(
            self.bag_path,
            cfg.get('zoom_ratio', 'auto'),
            cfg.get('topics', {}).get('camera_state', '/fcb/state'),
            use_camera_state=cfg.get('use_camera_state_zoom', False),
        )

        self.source = BagSource(
            self.bag_path,
            topics=cfg.get('topics'),
            zoom=self.zoom_ratio,
            max_gap_s=cfg.get('max_telemetry_gap_s', DEFAULT_MAX_GAP_S),
            altitude_source=cfg.get('altitude_source', 'lidar'),
            lidar=cfg.get('lidar'),
        )

    def _resolve_intrinsics(self) -> np.ndarray:
        if not self.cfg.get('use_camera_info'):
            return self.CALIBRATED_K
        k = camera_info_matrix(
            self.bag_path,
            self.cfg.get('topics', {}).get('camera_info', '/fcb/camera_info'),
        )
        if k is None:
            log.warning('use_camera_info set but no CameraInfo on the bag — '
                        'falling back to the calibrated matrix')
            return self.CALIBRATED_K
        log.warning('Using /fcb/camera_info intrinsics; the fitted extrinsics '
                    'in coordinate_transformer.py were matched to the '
                    'calibrated matrix, so accuracy may drop')
        log.info(f'camera_info K: fx={k[0,0]:.2f} fy={k[1,1]:.2f} '
                 f'cx={k[0,2]:.2f} cy={k[1,2]:.2f}')
        return k

    # ------------------------------------------------------------------ #
    # Per-frame processing                                                 #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray, frame_index: int,
                      telem: TelemetrySample | None, run_detector: bool):
        """Returns (annotated_frame, n_detections)."""
        if telem is None or not telem.is_complete():
            return frame.copy(), 0

        if not run_detector:
            return frame.copy(), 0

        detections = self.detector.detect_persons(frame)
        if not detections:
            return self.detector.draw_detections(frame, [], []), 0

        zoom_factor = telem.zoom / self.zoom_baseline
        gps_coords  = []

        for idx, det in enumerate(detections):
            cx, cy     = det['center']
            confidence = det['confidence']

            result = self.transformer.pixel_to_gps(
                u=cx, v=cy,
                drone_lat=telem.lat, drone_lon=telem.lon,
                drone_altitude=telem.alt_agl,
                roll=telem.roll, pitch=telem.pitch, yaw=telem.yaw,
                ground_altitude=0.0,
                zoom=zoom_factor,
            )

            if result is None:
                gps_coords.append((None, None))
                continue

            target_lat, target_lon = result
            gps_coords.append((target_lat, target_lon))

            log_detection(
                self.csv_path, telem.ts, frame_index, idx + 1,
                target_lat, target_lon, telem,
                confidence, len(detections),
            )

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
                    log_clusters(self.clustered_csv, idx, buf.flush())

            buf.append(target_lat, target_lon, telem.ts, confidence)

        vis = self.detector.draw_detections(frame, detections, gps_coords)
        return vis, len(detections)

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        cfg      = self.cfg
        viewer   = Viewer(cfg)
        realtime = cfg.get('realtime', True)
        every_n  = max(1, int(cfg.get('detect_every_n', 1)))
        skip_to  = float(cfg.get('start_offset_s', 0.0) or 0.0)

        t_bag0     = None          # capture time of the first emitted frame
        t_wall0    = None          # wall clock that t_anchor maps onto
        t_anchor   = None          # bag-relative time t_wall0 corresponds to
        last_speed = None
        prev_rel   = None
        fps_window = deque(maxlen=30)
        n_frames = n_det_frames = n_detections = n_no_telem = 0
        lag = 0.0

        log.info(f'Streaming {self.bag_path} — press q in the window to stop')

        try:
            for frame_index, ts, frame, telem in self.source.stream(skip_to):
                if ts is None:
                    continue
                if t_bag0 is None:
                    t_bag0 = ts
                t_rel = ts - t_bag0

                # Frames before start_offset_s come back without an image.
                if frame is None:
                    continue

                # --- pace against the bag clock -------------------------
                # t_wall0 is the wall time that bag-relative t_anchor maps to.
                # Both are rebased whenever the mapping breaks: at the first
                # frame, after a pause, and on a speed change — otherwise the
                # new rate would be applied retroactively to the whole run and
                # the clock would jump.
                now = time.perf_counter()
                # Capture stamps are not guaranteed monotonic — this recorder
                # has 155 packets whose stamp precedes their predecessor's, by
                # up to 67 ms. Rebase rather than trying to sleep backwards.
                went_back = prev_rel is not None and t_rel < prev_rel
                prev_rel  = t_rel
                if t_wall0 is None or viewer.speed != last_speed or went_back:
                    t_wall0, t_anchor = now, t_rel
                    last_speed = viewer.speed
                    lag = 0.0
                elif realtime:
                    target = t_wall0 + (t_rel - t_anchor) / max(viewer.speed, 1e-6)
                    ahead  = target - now
                    if ahead > self.MAX_SLEEP_S:
                        # A recording stall, not a frame we are early for.
                        # 25m_1x_2ms has a 20.5s hole at t=374s; honouring it
                        # literally would freeze the window for 20 seconds.
                        log.info(f'Gap of {ahead:.1f}s in the bag at '
                                 f't={t_rel:.1f}s — skipping the wait')
                        t_wall0, t_anchor = now, t_rel
                        lag = 0.0
                    elif ahead > 0:
                        time.sleep(ahead)
                        lag = 0.0
                    else:
                        lag = -ahead

                t_frame = time.perf_counter()

                run_detector = viewer.detect and (frame_index % every_n == 0)
                vis, n_det = self.process_frame(frame, frame_index, telem,
                                                run_detector)
                if telem is None:
                    n_no_telem += 1
                if run_detector:
                    n_det_frames += 1
                    n_detections += n_det
                n_frames += 1

                dt = time.perf_counter() - t_frame
                fps_window.append(dt)
                fps = len(fps_window) / max(sum(fps_window), 1e-6)

                vis = viewer.draw_hud(vis, frame_index, t_rel, telem, n_det,
                                      fps, lag)
                viewer.show(vis)
                viewer.handle_keys(frame, frame_index)
                if viewer.quit:
                    log.info('User requested quit.')
                    break

                # A pause stops the bag clock, so rebase when it resumes.
                if viewer.paused:
                    t_wall0 = None

        except KeyboardInterrupt:
            log.info('KeyboardInterrupt — shutting down.')
        finally:
            self._shutdown(viewer, n_frames, n_det_frames, n_detections,
                           n_no_telem)

    def _shutdown(self, viewer, n_frames, n_det_frames, n_detections, n_no_telem):
        for slot, buf in self._buffers.items():
            if len(buf) > 0:
                log_clusters(self.clustered_csv, slot, buf.flush())
        viewer.close()
        log.info(f'Frames shown {n_frames} | detector ran on {n_det_frames} | '
                 f'{n_detections} detection(s) | {n_no_telem} without telemetry')
        if self.csv_path:
            log.info(f'Detections -> {self.csv_path}')
        log.info('Shutdown complete.')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='Person geolocalisation, live off a ROS 2 bag')
    ap.add_argument('bag', nargs='?', default=CONFIG['bag_path'],
                    help='rosbag2 directory (default: the CONFIG value)')
    ap.add_argument('--zoom', default=None,
                    help="optical zoom ratio, or 'auto' to read it from the "
                         "bag name (default: the CONFIG value)")
    ap.add_argument('--roi', type=float, default=None,
                    help='detection ROI as a fraction of the frame '
                         f"(default: {CONFIG['roi_fraction']})")
    ap.add_argument('--start', type=float, default=None,
                    help='skip this many seconds into the flight')
    ap.add_argument('--alt-source', choices=('lidar', 'baro'), default=None,
                    help='height above ground from the downward lidar on /scan '
                         f"(default: {CONFIG['altitude_source']}) or from the "
                         'barometric rel_alt alone')
    ap.add_argument('--no-preview', action='store_true',
                    help='run headless')
    args = ap.parse_args()

    cfg = dict(CONFIG)
    cfg['bag_path'] = os.path.expanduser(args.bag)
    if args.zoom is not None:
        cfg['zoom_ratio'] = (args.zoom if args.zoom == 'auto'
                             else float(args.zoom))
    if args.roi is not None:
        cfg['roi_fraction'] = args.roi
    if args.start is not None:
        cfg['start_offset_s'] = args.start
    if args.alt_source is not None:
        cfg['altitude_source'] = args.alt_source
    if args.no_preview:
        cfg['show_preview'] = False

    PersonLocalisationPipeline(cfg).run()
