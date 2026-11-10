"""
ROS 2 bag data source
=====================
Everything the pipeline consumes comes out of a single rosbag2 directory:

    ~/rosbags/25m_1x_2ms/
        metadata.yaml
        fcb-*_0.db3          <- imagery AND telemetry both live in here

Video
-----
The imagery is on `/fcb/image/ffmpeg` as `ffmpeg_image_transport_msgs/FFMPEGPacket`
— H.265 packets, one per frame. That message package is not installed on this
machine, so `parse_ffmpeg_packet` below unpacks the CDR by hand and PyAV does
the decode. Decoding from the bag rather than the sidecar .mp4 is deliberate:
the .mp4 the recorder writes holds 31490 frames against the bag's 31507
packets (the encoder's start-up latency is not flushed to the file), so mp4
frame N is not bag packet N and every projection would inherit that skew.

Time base — every topic is keyed on its header stamp
----------------------------------------------------
Bag receive-time is when the recorder *wrote* a message, not when the data was
captured, and the two topics do not lag by the same amount. Measured over
25m_1x_2ms (header minus bag receive):

    /fcb/image/ffmpeg                 -51.9 ms  (sd 163.2, min -20537)
    /fcb/camera_info                  -51.9 ms
    /mavros/global_position/global     -0.2 ms  (sd 0.2)
    /mavros/imu/data                   -0.2 ms  (sd 0.2)

The camera is ~52 ms behind the flight controller, so keying both off bag time
matches every frame to telemetry ~52 ms in its own future — a fixed attitude
and position lead, not noise. That min of -20537 ms is the recorder's 20.5 s
stall at t=374 s, where bag time is wrong by the whole length of the stall.
Header stamps are the capture clock and are what everything here uses; all the
stamps sit in the same UTC epoch domain, so they are directly comparable.

`/mavros/global_position/rel_alt` is the exception: it is a bare `Float64` and
carries no header at all. It is published back-to-back with
`global_position/global` from the same MAVLink `GLOBAL_POSITION_INT`, so it
inherits that message's header stamp. Verified on this bag: all 27391 rel_alt
messages are immediately preceded by a GPS message (mean bag gap 0.045 ms, max
3.2 ms), and rel_alt is bit-identical to the headered
`global_position/local` z (max difference 0.000 m over the flight).

Interpolating across a dropout would silently invent telemetry, so `at()`
refuses to interpolate across a gap wider than `max_gap_s`. For reference the
observed gaps are: GPS median 19.7 ms / max 241 ms, IMU median 8.2 ms / max
96 ms.

Height above ground
-------------------
`rel_alt` is a barometric height above the launch point, not a measurement of
the distance to the ground. The drone carries a downward-facing 2D lidar on
`/scan`, and `lidar_altimeter.LidarAltimeter` turns that into a direct height,
which is what `alt_agl` carries by default. On these bags the lidar reads
0.9-1.8 m above the barometer, bag by bag — an error the projection would
otherwise divide straight into every ground offset. Pass
`altitude_source='baro'` to go back to the barometer alone.

Telemetry
---------
MAVROS publishes orientation in ENU (ROS REP-103). The projection in
`coordinate_transformer.py` was calibrated against the old recorder CSV, whose
`yaw_deg` matched its `heading_deg` — i.e. NED, clockwise from north. The two
are not interchangeable, so `_enu_to_ned_rpy` converts. Verified against GPS
course-over-ground on the 25m_1x_2ms bag: course 179.6 deg vs converted yaw
178.6 deg, and course 0.4 deg vs 358.7 deg.
"""

from __future__ import annotations

import logging
import math
import os
import re
import struct
from dataclasses import dataclass

import numpy as np

log = logging.getLogger('bag_source')

# Refuse to interpolate telemetry across a hole wider than this. Generous
# against the gaps this recorder actually produces (GPS max 241 ms) while still
# catching a real dropout, which lasts seconds rather than milliseconds.
DEFAULT_MAX_GAP_S = 0.25

# A header stamp is trusted only if it is a plausible UTC epoch time and lands
# near the bag's own write time. Zero/unset stamps and any clock in a different
# domain are rejected, and the topic falls back to bag receive-time.
_MIN_EPOCH_S     = 1_000_000_000.0      # 2001-09-09; anything below is unset
_MAX_STAMP_SKEW_S = 60.0

# Lidar-vs-barometer offset is measured only above this height. Below it the
# drone is on or near the ground, where the lidar is reading its own mounting
# height off the tarmac and the two quantities are not comparable.
_BIAS_MIN_ALT_M = 5.0

# Topic names, overridable through the pipeline CONFIG.
DEFAULT_TOPICS = {
    'image':       '/fcb/image/ffmpeg',
    'camera_info': '/fcb/camera_info',
    'gps':         '/mavros/global_position/global',
    'rel_alt':     '/mavros/global_position/rel_alt',
    'imu':         '/mavros/imu/data',
    'camera_state':'/fcb/state',
    'scan':        '/scan',
}


def _header_stamp(msg) -> float | None:
    """Header stamp in epoch seconds, or None if the message has no header."""
    header = getattr(msg, 'header', None)
    if header is None:
        return None
    stamp = header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def _stamp_is_sane(stamp: float | None, bag_ts: float) -> bool:
    """True if `stamp` looks like a capture time for a message written at `bag_ts`."""
    return (stamp is not None
            and stamp > _MIN_EPOCH_S
            and abs(stamp - bag_ts) <= _MAX_STAMP_SKEW_S)


class _Series:
    """
    One telemetry channel on the capture clock, with dropouts made explicit.

    `value_at` linearly interpolates, but returns None rather than inventing a
    value when the two samples it would interpolate between are further apart
    than `max_gap`.
    """

    def __init__(self, times, values, max_gap: float, name: str):
        t = np.asarray(times, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)

        order = np.argsort(t, kind='stable')     # stamps need not arrive sorted
        self.t, self.v = t[order], v[order]
        self.max_gap = max_gap
        self.name    = name
        # Interval i spans self.t[i]..self.t[i+1]; True where it is too wide to
        # interpolate across.
        self.stale = (np.diff(self.t) > max_gap) if len(self.t) > 1 else np.zeros(0, bool)

    def __len__(self):
        return len(self.t)

    @property
    def t_start(self):
        return float(self.t[0])

    @property
    def t_end(self):
        return float(self.t[-1])

    def value_at(self, ts: float) -> float | None:
        if not (self.t[0] <= ts <= self.t[-1]):
            return None
        i = int(np.searchsorted(self.t, ts, side='right')) - 1
        if i >= len(self.stale):                 # exactly on the last sample
            return float(self.v[-1])
        if self.t[i] == ts:                      # exact hit: real data, no gap
            return float(self.v[i])
        if self.stale[i]:
            return None
        return float(np.interp(ts, self.t, self.v))

    def gap_at(self, ts: float) -> float:
        """Width of the sample interval containing `ts`, for diagnostics."""
        if len(self.t) < 2 or not (self.t[0] <= ts <= self.t[-1]):
            return float('inf')
        i = min(int(np.searchsorted(self.t, ts, side='right')) - 1,
                len(self.stale) - 1)
        return float(self.t[i + 1] - self.t[i])


@dataclass
class TelemetrySample:
    """One interpolated telemetry state, in the projection's NED convention."""
    ts:      float          # bag time, UTC epoch seconds
    lat:     float
    lon:     float
    alt_agl: float          # what the projection uses; see alt_source
    roll:    float          # radians, NED
    pitch:   float          # radians, NED
    yaw:     float          # radians, NED (clockwise from north)
    compass: float          # degrees, 0-360
    zoom:    float
    frame:   int = -1
    # Where `alt_agl` came from, and what the other source said at the same
    # instant. Both are carried so a run can be re-scored against either.
    alt_source: str = 'baro'        # 'lidar' | 'lidar-bias' | 'baro'
    alt_lidar:  float | None = None  # measured distance to ground, or None
    alt_baro:   float | None = None  # rel_alt, height above the launch point

    def is_complete(self) -> bool:
        return None not in (self.lat, self.lon, self.alt_agl,
                            self.roll, self.pitch, self.yaw)


# ---------------------------------------------------------------------------
# FFMPEGPacket CDR
# ---------------------------------------------------------------------------

def parse_ffmpeg_packet(buf: bytes):
    """
    Unpack ffmpeg_image_transport_msgs/FFMPEGPacket straight from CDR.

    Field order on the wire is header, width, height, encoding, pts, flags,
    is_bigendian, data. Alignment is measured from the end of the 4-byte
    encapsulation header, which is why `_align` subtracts 4.

    Returns (header_stamp_sec, width, height, pts, is_keyframe, payload).
    """
    o = 4

    def _align(n):
        nonlocal o
        rem = (o - 4) % n
        if rem:
            o += n - rem

    sec  = struct.unpack_from('<i', buf, o)[0]; o += 4
    nsec = struct.unpack_from('<I', buf, o)[0]; o += 4
    ln   = struct.unpack_from('<I', buf, o)[0]; o += 4 + ln      # frame_id
    _align(4)
    width  = struct.unpack_from('<I', buf, o)[0]; o += 4
    height = struct.unpack_from('<I', buf, o)[0]; o += 4
    ln   = struct.unpack_from('<I', buf, o)[0]; o += 4 + ln      # encoding
    _align(8)
    pts   = struct.unpack_from('<Q', buf, o)[0]; o += 8
    flags = buf[o]; o += 1
    o += 1                                                       # is_bigendian
    _align(4)
    n_data = struct.unpack_from('<I', buf, o)[0]; o += 4

    return (sec + nsec * 1e-9, width, height, pts,
            bool(flags & 1), bytes(buf[o:o + n_data]))


# ---------------------------------------------------------------------------
# Optical zoom
# ---------------------------------------------------------------------------
#
# The camera publishes its state on /fcb/state as a fcb_interfaces/msg/CameraState.
# That package is not installed, so the message cannot be deserialised properly
# and the zoom ratio has to come from somewhere else. Two sources, in order:
#
#   1. The bag directory name. Recordings here are named like `25m_1x_2ms`
#      — altitude, zoom, speed — so the zoom is right there and it is a value
#      you control. This is the default and it is what `auto` uses.
#
#   2. `/fcb/state`, decoded by locating the field rather than trusting a fixed
#      byte offset. EXPERIMENTAL and off by default — see `zoom_from_camera_state`.
#
# A number given explicitly in CONFIG or on the command line always wins.

# `25m_1x_2ms` -> 1.0, `30m_2.5x_3ms` -> 2.5. The guards stop it matching the
# `2ms` in the same name, or a hex literal like `0x0711`.
# Lookarounds, not consuming guards, so that two zoom-ish tokens in one name
# are both seen and the name is correctly rejected as ambiguous.
_ZOOM_IN_NAME = re.compile(
    r'(?<![0-9A-Za-z])(\d+(?:\.\d+)?)x(?![0-9A-Za-z])', re.IGNORECASE)


def zoom_from_name(bag_path: str) -> float | None:
    """Optical zoom parsed out of the bag directory name, or None."""
    name = os.path.basename(os.path.normpath(bag_path))
    found = _ZOOM_IN_NAME.findall(name)
    if len(found) != 1:                  # nothing, or ambiguous — say nothing
        return None
    value = float(found[0])
    return value if 0.1 <= value <= 60.0 else None


def zoom_from_camera_state(bag_path: str,
                           topic: str = DEFAULT_TOPICS['camera_state']):
    """
    EXPERIMENTAL. Recover the zoom ratio from /fcb/state without the message
    definition, by finding the one float32 field that behaves like a zoom.

    Every 4-byte-aligned offset is read as a float32 across the whole topic. A
    candidate must be in [1, 60] in every message and take few distinct values
    (zoom is a discrete setting, not a continuously drifting sensor reading).
    Exactly one candidate must survive, or this gives up rather than guess.

    Unverified: both bags available here were shot at 1x, so a correct reading
    of 1.0 proves nothing about how this behaves at 2x. Validate with
    `probe_zoom.py` against a bag you know is not 1x before relying on it —
    that is why `use_camera_state_zoom` defaults to False.

    Returns (zoom, n_distinct, offset) or None.
    """
    import rosbag2_py

    reader = _open_bag(bag_path)
    types  = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        return None

    sf = rosbag2_py.StorageFilter()
    sf.topics = [topic]
    reader.set_filter(sf)

    raw = []
    while reader.has_next():
        _t, data, _ts = reader.read_next()
        raw.append(bytes(data))
    if not raw:
        return None

    size = len(raw[0])
    raw  = [m for m in raw if len(m) == size]      # a changed layout is unusable

    candidates = []
    for off in range(0, size - 3, 4):
        vals = np.array([struct.unpack_from('<f', m, off)[0] for m in raw])
        if not np.all(np.isfinite(vals)):
            continue
        if vals.min() < 1.0 or vals.max() > 60.0:
            continue
        distinct = np.unique(np.round(vals, 3))
        if len(distinct) > 8:                      # a drifting sensor, not a setting
            continue
        candidates.append((off, vals, len(distinct)))

    if len(candidates) != 1:
        log.warning(f'{topic}: {len(candidates)} zoom-shaped field(s); '
                    f'cannot identify the zoom without fcb_interfaces')
        return None

    off, vals, n_distinct = candidates[0]
    return float(np.median(vals)), n_distinct, off


def resolve_zoom(bag_path: str, setting, topic: str = DEFAULT_TOPICS['camera_state'],
                 use_camera_state: bool = False) -> float:
    """
    Turn the CONFIG `zoom_ratio` into a number, and say where it came from.

    `setting` is either a number (used as-is) or 'auto'.
    """
    if setting is not None and not isinstance(setting, str):
        log.info(f'Optical zoom: {float(setting):g}x (set explicitly)')
        return float(setting)

    from_state = None
    if use_camera_state:
        got = zoom_from_camera_state(bag_path, topic)
        if got is not None:
            from_state = got[0]
            log.info(f'Optical zoom: {from_state:g}x from {topic} '
                     f'(offset {got[2]}, {got[1]} distinct value(s)) — EXPERIMENTAL')

    from_name = zoom_from_name(bag_path)

    if from_state is not None and from_name is not None and \
            abs(from_state - from_name) > 1e-6:
        log.warning(f'Zoom disagreement: {topic} says {from_state:g}x but the bag '
                    f'name says {from_name:g}x — using the name. Set zoom_ratio '
                    f'explicitly to settle it.')
        return from_name
    if from_name is not None:
        log.info(f'Optical zoom: {from_name:g}x from the bag name')
        return from_name
    if from_state is not None:
        return from_state

    log.warning(f'Optical zoom unknown for {os.path.basename(bag_path)} — '
                f'assuming 1.0x. Name the bag like `25m_2x_2ms`, or set '
                f'zoom_ratio in CONFIG. A wrong zoom scales every coordinate.')
    return 1.0


# ---------------------------------------------------------------------------
# Angle conventions
# ---------------------------------------------------------------------------

def _enu_to_ned_rpy(quat_xyzw):
    """
    MAVROS ENU/FLU quaternion -> the NED/FRD roll, pitch, yaw the projection
    was calibrated in. Yaw becomes clockwise-from-north; pitch flips sign.
    """
    from scipy.spatial.transform import Rotation
    roll, pitch, yaw = Rotation.from_quat(quat_xyzw).as_euler('xyz')
    return roll, -pitch, math.pi / 2 - yaw


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class BagTelemetry:
    """
    Loads the MAVROS topics into memory once, then answers `at(t)` for any time
    in range: linear interpolation for position and altitude, SLERP for
    orientation.

    Every channel is keyed on its message's **header stamp** — the capture
    clock — not on bag receive-time, so the frame and the telemetry it is
    matched against refer to the same instant. See the module docstring for the
    measured lag between the two clocks and for how the headerless
    `rel_alt` topic is stamped. A lookup that would have to interpolate across
    a dropout returns None instead of a fabricated value.
    """

    def __init__(self, bag_path: str, topics: dict | None = None,
                 zoom: float = 1.0, max_gap_s: float = DEFAULT_MAX_GAP_S,
                 altitude_source: str = 'lidar', lidar: dict | None = None):
        self.topics    = dict(DEFAULT_TOPICS, **(topics or {}))
        self.zoom      = zoom
        self.max_gap_s = max_gap_s
        self.altitude_source = altitude_source
        self._load(bag_path)
        self.altimeter = None
        self.alt_bias  = 0.0          # median(lidar - baro) over this flight
        self._bias_t   = np.zeros(0)  # per-scan lidar - baro, for the fallback
        self._bias_v   = np.zeros(0)
        if altitude_source == 'lidar':
            self._load_altimeter(bag_path, lidar or {})
        elif altitude_source != 'baro':
            raise ValueError(f"altitude_source must be 'lidar' or 'baro', "
                             f"got {altitude_source!r}")

    def _load_altimeter(self, bag_path, lidar_cfg: dict):
        """
        Load the downward-facing lidar and measure how it sits against the
        barometer over this flight.

        The two are not the same quantity: the barometer reports height above
        the launch point, the lidar the distance to the ground right now. They
        differ by the terrain between launch and here, plus whatever the
        barometer has drifted — on these bags a median 0.9 to 1.8 m, which the
        projection would otherwise divide straight into every ground offset.

        That difference is also what makes the fallback continuous. Where the
        lidar refuses a moment (a scan whose slice of the fan misses nadir, a
        dropout), `at()` uses the barometer shifted by the difference measured
        at the surrounding scans, so the height does not jump when the source
        changes. `alt_bias` — the flight median, over scans above
        `_BIAS_MIN_ALT_M`, where the drone is airborne and the two are
        comparable — is only the last resort, for a query the per-scan series
        cannot reach.
        """
        from lidar_altimeter import LidarAltimeter

        alt = LidarAltimeter(bag_path, topic=self.topics.get('scan', '/scan'),
                             **lidar_cfg)
        if not alt.available:
            if alt.n_scans == 0:
                log.info(f'No lidar on {self.topics.get("scan", "/scan")} — '
                         f'the barometer is the only height source on this bag')
            else:
                log.warning(f'{alt.n_scans} lidar scan(s) but no usable height — '
                            f'falling back to the barometer for every frame')
            self.altitude_source = 'baro'
            return

        self.altimeter = alt

        # Per-scan lidar-minus-barometer, the series the fallback rides on.
        bias_t, bias_v, airborne = [], [], []
        for t, h in zip(*alt.samples()):
            baro = self.alt.value_at(t)
            if baro is None:
                continue
            bias_t.append(t)
            bias_v.append(h - baro)
            if h > _BIAS_MIN_ALT_M:
                airborne.append(h - baro)

        self._bias_t = np.asarray(bias_t)
        self._bias_v = np.asarray(bias_v)

        if airborne:
            self.alt_bias = float(np.median(airborne))
            spread = (np.percentile(airborne, 84) - np.percentile(airborne, 16))
            log.info(f'Lidar reads {self.alt_bias:+.2f} m against the barometer '
                     f'(median over {len(airborne)} scan(s) above '
                     f'{_BIAS_MIN_ALT_M:g} m, p16-p84 spread {spread:.2f} m) — '
                     f'the barometer is shifted onto the lidar where the lidar '
                     f'is silent')
        else:
            log.warning('Lidar and barometer never overlap above '
                        f'{_BIAS_MIN_ALT_M:g} m; the fallback will be unshifted')

    def _load(self, bag_path):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from scipy.spatial.transform import Rotation, Slerp

        reader = _open_bag(bag_path)
        types  = {t.name: t.type for t in reader.get_all_topics_and_types()}

        gps_topic = self.topics['gps']
        alt_topic = self.topics['rel_alt']
        imu_topic = self.topics['imu']

        wanted  = [gps_topic, alt_topic, imu_topic]
        missing = [t for t in wanted if t not in types]
        if missing:
            raise ValueError(f'Bag {bag_path} is missing telemetry topic(s): {missing}')

        sf = rosbag2_py.StorageFilter()
        sf.topics = wanted
        reader.set_filter(sf)

        gps_t, gps_lat, gps_lon = [], [], []
        alt_t, alt_v            = [], []
        imu_t, imu_q            = [], []

        # Per-topic tally of header stamps that were unusable, so a bag with a
        # broken clock is reported rather than silently mis-synced.
        bad = {t: 0 for t in wanted}
        n   = {t: 0 for t in wanted}

        # rel_alt has no header of its own; it inherits the stamp of the GPS
        # message it is published alongside. None until the first GPS arrives.
        last_gps_stamp = None
        orphan_alt     = 0

        for topic, data, bag_ns in _iter(reader):
            bag_ts = bag_ns * 1e-9
            msg    = deserialize_message(data, get_message(types[topic]))
            n[topic] += 1

            if topic == alt_topic:
                if last_gps_stamp is None:
                    orphan_alt += 1          # precedes any GPS fix; unstampable
                    continue
                alt_t.append(last_gps_stamp)
                alt_v.append(msg.data)
                continue

            stamp = _header_stamp(msg)
            if not _stamp_is_sane(stamp, bag_ts):
                bad[topic] += 1
                stamp = bag_ts               # fall back to the write time

            if topic == gps_topic:
                last_gps_stamp = stamp
                gps_t.append(stamp)
                gps_lat.append(msg.latitude)
                gps_lon.append(msg.longitude)
            else:
                q = msg.orientation
                imu_t.append(stamp)
                imu_q.append((q.x, q.y, q.z, q.w))

        if not (gps_t and alt_t and imu_t):
            raise ValueError(f'Bag {bag_path} has empty telemetry')

        for topic in (gps_topic, imu_topic):
            if bad[topic]:
                log.warning(
                    f'{topic}: {bad[topic]}/{n[topic]} header stamp(s) unusable '
                    f'(unset, or more than {_MAX_STAMP_SKEW_S:g}s from bag time) '
                    f'— those fell back to bag receive-time')
        if orphan_alt:
            log.info(f'{alt_topic}: {orphan_alt} message(s) before the first GPS '
                     f'fix had no stamp to inherit and were dropped')

        # Latitude and longitude are two channels sharing one clock.
        gps_t = np.asarray(gps_t)
        order = np.argsort(gps_t, kind='stable')
        self.lat = _Series(gps_t[order], np.asarray(gps_lat)[order],
                           self.max_gap_s, 'lat')
        self.lon = _Series(gps_t[order], np.asarray(gps_lon)[order],
                           self.max_gap_s, 'lon')
        self.alt = _Series(alt_t, alt_v, self.max_gap_s, 'alt')

        # Slerp needs strictly increasing times.
        imu_t = np.asarray(imu_t)
        imu_q = np.asarray(imu_q)
        order = np.argsort(imu_t, kind='stable')
        imu_t, imu_q = imu_t[order], imu_q[order]
        keep  = np.concatenate(([True], np.diff(imu_t) > 0))
        dropped_imu = int((~keep).sum())
        self._imu_t = imu_t[keep]
        self._imu_stale = np.diff(self._imu_t) > self.max_gap_s
        self._slerp = Slerp(self._imu_t, Rotation.from_quat(imu_q[keep]))
        if dropped_imu:
            log.info(f'{imu_topic}: dropped {dropped_imu} message(s) sharing a '
                     f'stamp with their predecessor')

        self.t_start = max(self.lat.t_start, self.alt.t_start, self._imu_t[0])
        self.t_end   = min(self.lat.t_end,   self.alt.t_end,   self._imu_t[-1])

        log.info(f'Telemetry loaded on the capture clock: {len(self.lat)} GPS, '
                 f'{len(self.alt)} alt, {len(self._imu_t)} IMU over '
                 f'{self.t_end - self.t_start:.1f}s')

    def _attitude_at(self, ts: float):
        """SLERPed attitude, or None across an IMU dropout."""
        if not (self._imu_t[0] <= ts <= self._imu_t[-1]):
            return None
        i = int(np.searchsorted(self._imu_t, ts, side='right')) - 1
        if (0 <= i < len(self._imu_stale) and self._imu_stale[i]
                and self._imu_t[i] != ts):       # exact hit is real data
            return None
        return _enu_to_ned_rpy(self._slerp(ts).as_quat())

    def at(self, ts: float) -> TelemetrySample | None:
        """
        Interpolated state at capture time `ts`, or None when the bag cannot
        support it — outside the telemetry span, or inside a dropout wider than
        `max_gap_s`. Returning None is deliberate: the caller skips the frame
        rather than projecting against invented telemetry.
        """
        if not (self.t_start <= ts <= self.t_end):
            return None

        attitude = self._attitude_at(ts)
        if attitude is None:
            return None
        roll, pitch, yaw = attitude

        lat = self.lat.value_at(ts)
        lon = self.lon.value_at(ts)
        baro = self.alt.value_at(ts)
        if None in (lat, lon, baro):
            return None

        alt, source, lidar = self._altitude_at(ts, baro)

        return TelemetrySample(
            ts=ts,
            lat=lat, lon=lon, alt_agl=alt,
            roll=roll, pitch=pitch, yaw=yaw,
            compass=math.degrees(yaw) % 360.0,
            zoom=self.zoom,
            alt_source=source, alt_lidar=lidar, alt_baro=baro,
        )

    def _altitude_at(self, ts: float, baro: float):
        """
        (height used, where it came from, what the lidar said).

        The lidar measures the distance to the ground and is preferred. Where
        it is silent the barometer stands in, shifted by the flight's measured
        lidar-barometer offset so the height does not step when the source
        changes.
        """
        if self.altimeter is None:
            return baro, 'baro', None
        lidar = self.altimeter.at(ts)
        if lidar is not None:
            return lidar, 'lidar', lidar
        return baro + self._bias_at(ts), 'lidar-bias', None

    def _bias_at(self, ts: float) -> float:
        """
        Lidar-minus-barometer around `ts`, held flat outside the scan span.

        Taking it from the surrounding scans rather than from the flight median
        is what keeps the height continuous across a lidar gap: at the edge of
        the gap the shifted barometer equals the lidar reading exactly, and it
        drifts from there at whatever rate the barometer does.
        """
        if len(self._bias_t) == 0:
            return self.alt_bias
        return float(np.interp(ts, self._bias_t, self._bias_v))

    def sync_report(self, ts: float) -> dict:
        """Per-channel sample spacing around `ts`, for diagnosing a mismatch."""
        return {
            'gps_gap_s': self.lat.gap_at(ts),
            'alt_gap_s': self.alt.gap_at(ts),
            'max_gap_s': self.max_gap_s,
        }


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

class BagVideo:
    """
    Decodes `/fcb/image/ffmpeg` on the fly, yielding (packet_index, ts, bgr),
    where `ts` is the packet's **header stamp** — the capture time, the same
    clock `BagTelemetry` is keyed on. Using bag receive-time here instead would
    hand every frame to telemetry ~52 ms in its own future, and would be wrong
    by the full 20.5 s across this recorder's stall.

    Each packet is stamped with its own index before being handed to the
    decoder, and the decoded frame is matched back by that stamp. Pairing them
    in FIFO order instead would be wrong: the decoder does not emit a frame for
    every packet it is given. On the 25m_1x_2ms bag the first 17 packets
    produce no output at all (they precede the first parameter sets), so a
    queue would marry frame 0 to packet 0's timestamp and skew the whole
    flight by ~0.29 s — about 0.6 m of ground error at 2 m/s. Matching on the
    stamp also survives any frame reordering.

    `index` is therefore the packet's position in the bag, not a count of
    frames emitted, and it is what `play_video.py` screenshots are named for.
    """

    def __init__(self, bag_path: str, topic: str = DEFAULT_TOPICS['image'],
                 codec: str | None = None):
        self.bag_path = bag_path
        self.topic    = topic
        self.codec    = codec
        self.n_packets = 0

    def frames(self, skip_before_rel: float = 0.0):
        """
        Yields (index, bag_ts, bgr). Frames earlier than `skip_before_rel`
        seconds into the video are still decoded — an inter-coded stream gives
        no choice — but are yielded with `bgr=None` rather than paying the
        YUV->BGR conversion, which is most of the cost of a skipped frame.
        """
        import av
        import rosbag2_py

        reader = _open_bag(self.bag_path)
        types  = {t.name: t.type for t in reader.get_all_topics_and_types()}
        if self.topic not in types:
            raise ValueError(f'Bag has no image topic {self.topic}. '
                             f'Available: {sorted(types)}')

        sf = rosbag2_py.StorageFilter()
        sf.topics = [self.topic]
        reader.set_filter(sf)

        ctx     = None
        stamps  = {}          # packet index -> bag time, pending a decode
        index   = 0           # packet index, used as the frame's pts
        t_first = None

        def _emit(frame):
            """Pair one decoded frame back to its packet's bag time."""
            nonlocal t_first
            ts = stamps.pop(frame.pts, None)
            if ts is None:                            # no stamp: unusable time
                return None
            if t_first is None:
                t_first = ts
            skip = skip_before_rel > 0.0 and ts - t_first < skip_before_rel
            return (int(frame.pts), ts,
                    None if skip else frame.to_ndarray(format='bgr24'))

        n_fallback = 0

        while reader.has_next():
            _topic, data, bag_ns = reader.read_next()
            hdr_ts, w, h, _pts, _key, payload = parse_ffmpeg_packet(data)
            if not payload:
                index += 1
                continue
            self.n_packets += 1

            bag_ts = bag_ns * 1e-9
            if not _stamp_is_sane(hdr_ts, bag_ts):
                hdr_ts = bag_ts                       # unset or foreign clock
                n_fallback += 1

            if ctx is None:
                # The encoding string on the packet is like 'hevc;nv12;bgr8;bgr8'.
                name = self.codec or _codec_of(data) or 'hevc'
                ctx = av.CodecContext.create(name, 'r')
                log.info(f'Video: {w}x{h}, codec {name}, topic {self.topic}')

            packet = av.Packet(payload)
            packet.pts = packet.dts = index
            stamps[index] = hdr_ts
            index += 1

            try:
                decoded = ctx.decode(packet)
            except Exception as exc:                  # corrupt packet mid-bag
                log.warning(f'Decode error at packet {index - 1}: {exc}')
                stamps.pop(index - 1, None)
                continue

            for frame in decoded:
                out = _emit(frame)
                if out is not None:
                    yield out

        if ctx is not None:                           # flush the decoder tail
            try:
                for frame in ctx.decode(None):
                    out = _emit(frame)
                    if out is not None:
                        yield out
            except Exception as exc:                  # noqa: BLE001
                log.warning(f'Decoder flush failed: {exc}')

        if n_fallback:
            log.warning(f'{self.topic}: {n_fallback}/{self.n_packets} header '
                        f'stamp(s) unusable — those fell back to bag receive-time')
        if stamps:
            log.info(f'{len(stamps)} packet(s) produced no frame '
                     f'(decoder start-up before the first keyframe)')


class BagFrameClock:
    """
    Frame number -> bag time, without decoding a single frame.

    A "frame number" here is the image packet's position in the bag, which is
    what `BagVideo` yields as its index and what `play_video.py` names its
    screenshots after. Times are the packets' header stamps, the same capture
    clock `BagVideo` and `BagTelemetry` use, so a frame number resolves to the
    identical instant whether it came from a live run or a saved screenshot.
    Only the headers are parsed — no frame is decoded — so this is cheap even
    on a long bag.
    """

    def __init__(self, bag_path: str, topic: str = DEFAULT_TOPICS['image']):
        import rosbag2_py

        reader = _open_bag(bag_path)
        types  = {t.name: t.type for t in reader.get_all_topics_and_types()}
        if topic not in types:
            raise ValueError(f'Bag has no image topic {topic}')

        sf = rosbag2_py.StorageFilter()
        sf.topics = [topic]
        reader.set_filter(sf)

        times = []
        n_fallback = 0
        while reader.has_next():
            _topic, data, bag_ns = reader.read_next()
            bag_ts = bag_ns * 1e-9
            try:
                hdr_ts = parse_ffmpeg_packet(data)[0]
            except (struct.error, IndexError):
                hdr_ts = None
            if not _stamp_is_sane(hdr_ts, bag_ts):
                hdr_ts = bag_ts
                n_fallback += 1
            times.append(hdr_ts)

        if not times:
            raise ValueError(f'No image packets on {topic}')

        self._times = np.asarray(times)
        if n_fallback:
            log.warning(f'Frame clock: {n_fallback}/{len(times)} header stamp(s) '
                        f'unusable — those fell back to bag receive-time')
        log.info(f'Frame clock: {len(self._times)} image packet(s) '
                 f'spanning {self._times[-1] - self._times[0]:.1f}s')

    def __len__(self):
        return len(self._times)

    def time_of(self, frame: int) -> float | None:
        """Bag time of that packet, or None if the bag has no such frame."""
        if not 0 <= frame < len(self._times):
            return None
        return float(self._times[frame])


def _codec_of(buf: bytes) -> str | None:
    """Pull the codec name out of the packet's 'hevc;nv12;bgr8;bgr8' string."""
    try:
        o = 4
        o += 8                                            # stamp
        ln = struct.unpack_from('<I', buf, o)[0]; o += 4 + ln
        rem = (o - 4) % 4
        if rem:
            o += 4 - rem
        o += 8                                            # width, height
        ln = struct.unpack_from('<I', buf, o)[0]; o += 4
        enc = buf[o:o + ln].split(b'\x00')[0].decode('ascii', 'ignore')
        return enc.split(';')[0] or None
    except Exception:
        return None


def _iter(reader):
    """Yield (topic, data, bag_ns) from an opened reader."""
    while reader.has_next():
        yield reader.read_next()


def _open_bag(bag_path: str):
    import rosbag2_py
    if not os.path.isdir(bag_path):
        raise FileNotFoundError(f'Bag directory not found: {bag_path}')
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'),
    )
    return reader


def camera_info_matrix(bag_path: str,
                       topic: str = DEFAULT_TOPICS['camera_info']):
    """First CameraInfo K on the bag as a 3x3, or None if the topic is absent."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = _open_bag(bag_path)
    types  = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        return None

    sf = rosbag2_py.StorageFilter()
    sf.topics = [topic]
    reader.set_filter(sf)
    if not reader.has_next():
        return None

    _t, data, _ts = reader.read_next()
    msg = deserialize_message(data, get_message(types[topic]))
    return np.asarray(msg.k, dtype=np.float64).reshape(3, 3)


# ---------------------------------------------------------------------------
# Combined source
# ---------------------------------------------------------------------------

class BagSource:
    """Iterates (index, ts, frame, telemetry) over one bag."""

    def __init__(self, bag_path: str, topics: dict | None = None,
                 zoom: float = 1.0, max_gap_s: float = DEFAULT_MAX_GAP_S,
                 altitude_source: str = 'lidar', lidar: dict | None = None):
        self.bag_path = bag_path
        self.topics   = dict(DEFAULT_TOPICS, **(topics or {}))
        self.telemetry = BagTelemetry(bag_path, self.topics, zoom=zoom,
                                      max_gap_s=max_gap_s,
                                      altitude_source=altitude_source,
                                      lidar=lidar)
        self.video     = BagVideo(bag_path, self.topics['image'])

    def __iter__(self):
        yield from self.stream()

    def stream(self, skip_before_rel: float = 0.0):
        for index, ts, frame in self.video.frames(skip_before_rel):
            telem = self.telemetry.at(ts) if ts is not None else None
            if telem is not None:
                telem.frame = index
            yield index, ts, frame, telem
