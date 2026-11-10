"""
Height above ground from the downward-facing 2D lidar
=====================================================
The flight controller's `rel_alt` is a barometric height above the launch
point. It is smooth but it is not a measurement of the distance to the ground
under the drone: it drifts with pressure, it is referenced to wherever the
drone armed rather than to the terrain it is now over, and on these bags it
reads **0.9-1.8 m low**, bag by bag (see the table in the README). The
projection divides by that number, so a 1.5 m error at 25 m AGL scales every
ground offset by 6% — a target 15 m from the nadir point lands ~0.9 m wrong.

The drone also carries a downward-facing 2D lidar on `/scan`. It measures the
distance to the ground directly, at 5 cm noise, and that is what this module
turns into a per-scan height.

What the scans look like
------------------------
`sensor_msgs/LaserScan`, `frame_id: laser`, ~4.8 Hz, 100 points per message,
`range_max` 50 m. The fan lies within +-45 deg of nadir, but each message
carries a different slice of it: the window runs anywhere from 29 to 82 deg
wide, `angle_increment` varies from 0.30 to 0.83 deg between messages, and the
window does not always contain the nadir direction. Anything that assumes one
fixed beam layout across messages is wrong on this sensor; every scan is read
through its own `angle_min` / `angle_increment`.

Over flat ground the profile is a V: slant range is shortest along the nadir
ray and grows as `h / cos(off-nadir angle)` towards both edges. The height is
therefore the range at the bottom of that V, which is also what makes this
estimate independent of the drone's attitude — a roll or pitch just slides the
bottom of the V along the fan, it does not change its depth.

How the height is estimated
---------------------------
The bottom of the V, found by fitting rather than by picking a sample:

1. Take the valid returns (finite, within `range_min`..`range_max`; this
   sensor also writes -1.0 for a miss, which the same test rejects).
2. Take the beams within `half_window_deg` of the shortest return and fit a
   quadratic of range against angle.
3. Reject outliers and re-fit, twice. The rejection is **asymmetric**: a
   return more than 3 sigma *shorter* than the fit is something between the
   drone and the ground — a person, a vehicle, a tree — and must not be
   allowed to pull the height down, while a longer one is usually just noise.
4. The vertex of the fit is the height. Reject the scan if the parabola opens
   downward or its vertex falls outside the window, which is what happens when
   the message's slice of the fan does not contain nadir.

Measured over the six bags with a `/scan` topic: fit residual (sigma) 0.02 m
median / 0.10 m p95, and an estimate on 89-91% of the in-flight scans. The
scans that are refused are almost all the nadir-not-in-window case; they are
refused rather than guessed at, and `at()` interpolates across them as long as
the hole stays under `max_gap_s`.

Step 3 is why a person standing under the drone does not corrupt the height,
and step 4 is why a partial scan pointing off to one side does not silently
report a slant range as a height.

Interpolation, and what it refuses
----------------------------------
`at(ts)` linearly interpolates the per-scan heights onto the capture clock and
returns None rather than inventing a value when the two scans it would
interpolate between are more than `max_gap_s` apart, matching the way
`bag_source._Series` treats a telemetry dropout. Scans arrive every ~0.21 s and
a refused scan leaves a ~0.4 s hole, so the default 0.6 s bridges the normal
refusals while still catching a real lidar dropout.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from bag_source import _open_bag, _stamp_is_sane

log = logging.getLogger('lidar_altimeter')

DEFAULT_SCAN_TOPIC = '/scan'

# Scans arrive at ~4.8 Hz (0.21 s apart) and a refused scan leaves a ~0.4 s
# hole. Wide enough to bridge those, narrow enough that a genuine lidar
# dropout is refused rather than interpolated across.
DEFAULT_MAX_SCAN_GAP_S = 0.6

# Beams within this many degrees of the shortest return are fitted. Wider
# windows pull in more of the fan's model error (measured: +-10 deg gives a
# 0.19 m high-frequency spread against the barometer, +-15 deg 0.26 m,
# +-25 deg 0.42 m); narrower ones have too few beams to average the noise.
DEFAULT_HALF_WINDOW_DEG = 10.0

# Fewer beams than this in the window and the fit is not worth trusting.
DEFAULT_MIN_BEAMS = 8


class ScanHeight:
    """One scan's height estimate, with the numbers behind it."""

    __slots__ = ('ts', 'height', 'nadir_deg', 'sigma', 'n_beams')

    def __init__(self, ts, height, nadir_deg, sigma, n_beams):
        self.ts        = ts
        self.height    = height
        self.nadir_deg = nadir_deg      # where the bottom of the V sat in the fan
        self.sigma     = sigma          # robust fit residual, metres
        self.n_beams   = n_beams        # beams surviving rejection

    def __repr__(self):
        return (f'ScanHeight(h={self.height:.2f}m, nadir={self.nadir_deg:+.1f}deg, '
                f'sigma={self.sigma:.3f}m, n={self.n_beams})')


def height_from_scan(angles, ranges, range_min, range_max,
                     half_window_deg: float = DEFAULT_HALF_WINDOW_DEG,
                     min_beams: int = DEFAULT_MIN_BEAMS):
    """
    Vertical distance to the ground from one scan, or None if this scan cannot
    support one. See the module docstring for the method.

    `angles` are radians in the scan's own frame, `ranges` metres, both full
    length; invalid returns are filtered here.
    """
    angles = np.asarray(angles, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=np.float64)

    valid = (np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max))
    if valid.sum() < min_beams:
        return None
    a, r = angles[valid], ranges[valid]

    half = math.radians(half_window_deg)
    a0   = a[np.argmin(r)]
    win  = np.abs(a - a0) <= half
    if win.sum() < min_beams:
        return None

    x, y = a[win] - a0, r[win]
    keep = np.ones(len(x), dtype=bool)
    coef = sigma = None

    for _ in range(3):
        coef  = np.polyfit(x[keep], y[keep], 2)
        res   = y - np.polyval(coef, x)
        sigma = 1.4826 * np.median(np.abs(res[keep])) + 1e-6
        # Asymmetric: a short return is an object under the drone, not noise.
        keep  = (res > -3.0 * sigma) & (res < 4.0 * sigma)
        if keep.sum() < min_beams:
            return None
    # The loop's last fit predates its last rejection; fit once more so the
    # vertex below comes from exactly the beams that survived.
    coef  = np.polyfit(x[keep], y[keep], 2)
    sigma = 1.4826 * np.median(np.abs((y - np.polyval(coef, x))[keep])) + 1e-6

    if coef[0] <= 0:                      # opens downward: not a ground V
        return None
    x_vertex = -coef[1] / (2.0 * coef[0])
    if abs(x_vertex) > half:              # nadir is outside this scan's window
        return None
    height = float(np.polyval(coef, x_vertex))
    if not (range_min <= height <= range_max):
        return None

    return ScanHeight(ts=None, height=height,
                      nadir_deg=math.degrees(a0 + x_vertex),
                      sigma=float(sigma), n_beams=int(keep.sum()))


class LidarAltimeter:
    """
    Every `/scan` in one bag, reduced to a height above ground on the capture
    clock, with `at(ts)` answering for any time in range.

    Heights are the lidar's own distance to the ground. `mount_offset_m` is
    added to every one of them, for the case where the camera sits at a
    different height on the airframe than the lidar; leave it at 0 unless you
    have measured that offset. (On these bags the lidar reads 0.3-0.5 m with
    the drone on the ground, which is its height above the ground when landed,
    not the lidar-to-camera offset.)
    """

    def __init__(self, bag_path: str, topic: str = DEFAULT_SCAN_TOPIC,
                 max_gap_s: float = DEFAULT_MAX_SCAN_GAP_S,
                 half_window_deg: float = DEFAULT_HALF_WINDOW_DEG,
                 min_beams: int = DEFAULT_MIN_BEAMS,
                 mount_offset_m: float = 0.0):
        self.topic     = topic
        self.max_gap_s = max_gap_s
        self.n_scans   = 0
        self.n_refused = 0
        self._t = np.zeros(0)
        self._h = np.zeros(0)
        self._sigma = np.zeros(0)
        self._load(bag_path, half_window_deg, min_beams, mount_offset_m)

    @property
    def available(self) -> bool:
        return len(self._t) >= 2

    def _load(self, bag_path, half_window_deg, min_beams, mount_offset_m):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        reader = _open_bag(bag_path)
        types  = {t.name: t.type for t in reader.get_all_topics_and_types()}
        if self.topic not in types:
            log.info(f'No lidar topic {self.topic} on this bag — '
                     f'altitude will come from the barometer')
            return

        sf = rosbag2_py.StorageFilter()
        sf.topics = [self.topic]
        reader.set_filter(sf)

        times, heights, sigmas = [], [], []
        n_fallback = 0

        while reader.has_next():
            _topic, data, bag_ns = reader.read_next()
            bag_ts = bag_ns * 1e-9
            msg    = deserialize_message(data, get_message(types[self.topic]))
            self.n_scans += 1

            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if not _stamp_is_sane(stamp, bag_ts):
                stamp = bag_ts
                n_fallback += 1

            n   = len(msg.ranges)
            ang = msg.angle_min + np.arange(n) * msg.angle_increment
            got = height_from_scan(ang, msg.ranges, msg.range_min, msg.range_max,
                                   half_window_deg, min_beams)
            if got is None:
                self.n_refused += 1
                continue

            times.append(stamp)
            heights.append(got.height + mount_offset_m)
            sigmas.append(got.sigma)

        if not times:
            log.warning(f'{self.topic}: {self.n_scans} scan(s), none usable — '
                        f'altitude will come from the barometer')
            return

        order = np.argsort(np.asarray(times), kind='stable')
        self._t     = np.asarray(times)[order]
        self._h     = np.asarray(heights)[order]
        self._sigma = np.asarray(sigmas)[order]
        self._stale = np.diff(self._t) > self.max_gap_s

        if n_fallback:
            log.warning(f'{self.topic}: {n_fallback}/{self.n_scans} header stamp(s) '
                        f'unusable — those fell back to bag receive-time')
        log.info(f'Lidar altimeter: {len(self._t)}/{self.n_scans} scan(s) gave a '
                 f'height ({self.n_refused} refused), median fit residual '
                 f'{np.median(self._sigma):.3f} m, spanning '
                 f'{self._t[-1] - self._t[0]:.1f}s')

    def at(self, ts: float) -> float | None:
        """
        Height above ground at capture time `ts`, or None when the lidar cannot
        support it — outside the scan span, or across a hole wider than
        `max_gap_s`.
        """
        if not self.available or not (self._t[0] <= ts <= self._t[-1]):
            return None
        i = int(np.searchsorted(self._t, ts, side='right')) - 1
        if i >= len(self._stale):
            return float(self._h[-1])
        if self._t[i] == ts:
            return float(self._h[i])
        if self._stale[i]:
            return None
        return float(np.interp(ts, self._t, self._h))

    @property
    def median_sigma(self) -> float:
        """Median robust fit residual over the scans that produced a height."""
        return float(np.median(self._sigma)) if len(self._sigma) else float('nan')

    def samples(self):
        """(times, heights) for every scan that produced an estimate."""
        return self._t, self._h

    def coverage(self, times) -> float:
        """Fraction of `times` this altimeter can answer for."""
        times = np.asarray(times, dtype=np.float64)
        if len(times) == 0 or not self.available:
            return 0.0
        return float(np.mean([self.at(t) is not None for t in times]))


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
#
#   python3 lidar_altimeter.py
#
# Synthetic scans with a known height, to show what the estimator does with a
# person underneath the drone and with a scan whose window misses nadir. Run it
# after changing `height_from_scan`.

if __name__ == '__main__':
    import math as _math

    def _synth(h=25.0, phi_deg=3.0, a0=-45, a1=45, n=100, person=None,
               noise=0.02, seed=0):
        """A flat-ground scan: range = h / cos(angle - mount tilt), plus noise."""
        rng = np.random.default_rng(seed)
        a   = np.radians(np.linspace(a0, a1, n))
        r   = h / np.cos(a - _math.radians(phi_deg)) + rng.normal(0, noise, n)
        if person:                      # height, angle: an object under the drone
            r[np.abs(np.degrees(a) - person[1]) < 1.0] -= person[0]
        return a, r

    CASES = [
        ('flat ground',                {}),
        ('person 1.7 m at nadir',      {'person': (1.7, 3.0)}),
        ('person 1.7 m at +8 deg',     {'person': (1.7, 8.0)}),
        ('window misses nadir',        {'a0': -45, 'a1': -15}),
        ('narrow window, on nadir',    {'a0': -12, 'a1': 18}),
        ('5x the measured noise',      {'noise': 0.10}),
        ('every return a miss',        {'noise': 0.0, 'h': 25.0, 'n': 100}),
    ]

    print(f'{"case":<26} {"estimate":>10}   (truth 25.000 m)')
    for name, kw in CASES:
        if name == 'every return a miss':
            a, r = np.radians(np.linspace(-45, 45, 100)), np.full(100, -1.0)
        else:
            a, r = _synth(**kw)
        est = height_from_scan(a, r, 0.2, 50.0)
        print(f'{name:<26} '
              + ('   refused' if est is None
                 else f'{est.height:10.3f}   nadir {est.nadir_deg:+5.2f} deg, '
                      f'sigma {est.sigma:.3f} m, {est.n_beams} beams'))
