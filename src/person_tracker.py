"""
Person Centering — standalone (no ROS)
=======================================
Reads frames from cv2.VideoCapture, detects the largest person with YOLOv8,
and publishes body-frame velocity set-points via pymavlink MAVLink.

Body-frame velocity mapping (downward-facing camera):
  pixel +X (person right of centre)  → body +Y  (strafe right)
  pixel +Y (person below  centre)    → body +X  (nose forward)
"""

from __future__ import annotations

import logging
import math
import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO
from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('person_tracker')

# ---------------------------------------------------------------------------
# Default configuration  (edit here or override at runtime)
# ---------------------------------------------------------------------------

CONFIG = {
    # Camera
    'camera_source':      0,          # int index, 'rtsp://…', or GStreamer string
    'camera_width':       640,
    'camera_height':      480,

    # MAVLink
    'mavlink_connection': 'udpin:0.0.0.0:14550',   # or 'serial:/dev/ttyUSB0,57600'
    'mavlink_dialect':    'ardupilotmega',

    # YOLO
    'model_path':         'best1.pt',
    'confidence_thresh':  0.45,

    # Control
    'dead_zone_px':       20,
    'max_linear_speed':   0.4,        # m/s
    'gain_xy':            0.0025,
    'publish_rate':       20.0,       # Hz — max rate for velocity set-points

    # Debug
    'show_preview':       True,
    'debug_window_title': 'Person Tracker',
}

# ---------------------------------------------------------------------------
# MAVLink velocity sender
# ---------------------------------------------------------------------------

class MAVLinkVelocitySender:
    """
    Wraps a pymavlink connection and exposes a simple method to send
    SET_POSITION_TARGET_LOCAL_NED in body-frame velocity mode.

    This replicates what MAVROS's /mavros/setpoint_velocity/cmd_vel
    (frame_id='base_link') does under the hood.
    """

    # Type mask: ignore everything except vx, vy, vz
    _TYPE_MASK = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )

    # MAV_FRAME_BODY_NED = 8  (body frame, forward/right/down)
    _FRAME = mavutil.mavlink.MAV_FRAME_BODY_NED

    def __init__(self, connection_string: str, dialect: str = 'ardupilotmega'):
        log.info(f'MAVLink connecting → {connection_string}')
        self._mav = mavutil.mavlink_connection(connection_string, dialect=dialect)
        self._mav.wait_heartbeat(timeout=15)
        log.info('MAVLink heartbeat received ✓')
        self._lock = threading.Lock()

    def send_body_velocity(self, vx: float, vy: float, vz: float = 0.0,
                           yaw_rate: float = 0.0):
        """
        Send a body-frame velocity command (m/s).
          vx  — forward  (+) / backward (-)
          vy  — right    (+) / left     (-)
          vz  — down     (+) / up       (-)  (keep 0 to maintain altitude)
        """
        boot_ms = int(time.monotonic() * 1000) & 0xFFFF_FFFF
        with self._lock:
            self._mav.mav.set_position_target_local_ned_send(
                boot_ms,
                self._mav.target_system,
                self._mav.target_component,
                self._FRAME,
                self._TYPE_MASK,
                0, 0, 0,          # x, y, z  (ignored)
                vx, vy, vz,       # vx, vy, vz
                0, 0, 0,          # ax, ay, az (ignored)
                0,                # yaw       (ignored)
                yaw_rate,         # yaw_rate  (ignored)
            )

    def hover(self):
        """Send zero velocity — drone holds position."""
        self.send_body_velocity(0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class PersonCenteringTracker:

    def __init__(self, cfg: dict = CONFIG):
        self.cfg          = cfg
        self.dead_zone    = cfg['dead_zone_px']
        self.max_speed    = cfg['max_linear_speed']
        self.gain         = cfg['gain_xy']
        self._min_dt      = 1.0 / cfg['publish_rate']
        self._last_pub_ts = 0.0

        log.info(f"Loading YOLOv8 model: {cfg['model_path']}")
        self.model = YOLO(cfg['model_path'])
        log.info('YOLOv8 loaded ✓')

        self.mav = MAVLinkVelocitySender(
            connection_string=cfg['mavlink_connection'],
            dialect=cfg['mavlink_dialect'],
        )

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #

    def _detect_largest_person(self, frame: np.ndarray):
        """Return bounding box (x1,y1,x2,y2) of the largest detected person, or None."""
        results = self.model(
            frame,
            classes=[0],
            conf=self.cfg['confidence_thresh'],
            verbose=False,
        )
        best_box, best_area = None, 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box  = (x1, y1, x2, y2)
        return best_box

    # ------------------------------------------------------------------ #
    # Control                                                              #
    # ------------------------------------------------------------------ #

    def _compute_velocity(self, err_x: float, err_y: float):
        """
        Map pixel error to body-frame velocity.
          err_x (pixel right)  → body +Y (strafe right)
          err_y (pixel down)   → body +X (forward)
        """
        dist = math.hypot(err_x, err_y)
        if dist <= self.dead_zone:
            return 0.0, 0.0

        vx =  self.gain * err_y   # body X = forward
        vy =  self.gain * err_x   # body Y = right

        speed = math.hypot(vx, vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx   *= scale
            vy   *= scale

        return vx, vy

    def _maybe_publish(self, vx: float, vy: float):
        now = time.monotonic()
        if now - self._last_pub_ts >= self._min_dt:
            self.mav.send_body_velocity(vx, vy)
            self._last_pub_ts = now

    # ------------------------------------------------------------------ #
    # Debug visualisation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _draw_debug(frame: np.ndarray, cx_img: int, cy_img: int,
                    best_box, err_x: int = 0, err_y: int = 0) -> np.ndarray:
        vis = frame.copy()
        cv2.drawMarker(vis, (cx_img, cy_img),
                       (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            bx = (x1 + x2) // 2
            by = (y1 + y2) // 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.drawMarker(vis, (bx, by), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            label = f"err({err_x:+d},{err_y:+d})"
            cv2.putText(vis, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        return vis

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        src = self.cfg['camera_source']
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error(f'Cannot open camera: {src}')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg['camera_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg['camera_height'])
        log.info(f'Camera opened: {src}')

        log.info(
            f'\n{"="*55}\n'
            f'  Person Centering — READY  (body-frame velocity)\n'
            f'  Dead-zone: ±{self.dead_zone} px  |  Gain: {self.gain}\n'
            f'{"="*55}'
        )

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.error('Camera read failed.')
                    break

                h, w    = frame.shape[:2]
                cx_img  = w // 2
                cy_img  = h // 2

                best_box = self._detect_largest_person(frame)

                err_x, err_y = 0, 0
                vx,    vy    = 0.0, 0.0

                if best_box is not None:
                    x1, y1, x2, y2 = best_box
                    bx = (x1 + x2) // 2
                    by = (y1 + y2) // 2
                    err_x = bx - cx_img
                    err_y = by - cy_img
                    vx, vy = self._compute_velocity(err_x, err_y)

                    if vx == 0.0 and vy == 0.0:
                        log.info('✓ Person centred — hovering',
                                 )
                    else:
                        log.info(
                            f'err=({err_x:+.0f},{err_y:+.0f})px  '
                            f'body_vel=({vx:+.3f},{vy:+.3f})m/s'
                        )
                else:
                    log.warning('No person detected — hovering')

                self._maybe_publish(vx, vy)

                if self.cfg.get('show_preview', True):
                    vis = self._draw_debug(frame, cx_img, cy_img,
                                           best_box, err_x, err_y)
                    cv2.imshow(self.cfg['debug_window_title'], vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        log.info('User quit.')
                        break

        except KeyboardInterrupt:
            log.info('KeyboardInterrupt — shutting down.')
        finally:
            log.info('Sending stop command.')
            self.mav.hover()
            cap.release()
            cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tracker = PersonCenteringTracker(CONFIG)
    tracker.run()
