#!/usr/bin/env python3
"""
What the downward-facing lidar says, bag by bag
===============================================
Prints, per bag: how many `/scan` messages produced a height, how those
heights compare with the barometric `rel_alt` the projection used to rely on,
and how much of the flight the lidar can actually answer for.

    cd src
    python3 probe_lidar.py ~/rosbags/*
    python3 probe_lidar.py ~/rosbags/25m_1x_2ms --detail

Read the `lidar-baro` column as the error the barometer was contributing: it
is the height the projection divides by, so a +1.5 m offset at 25 m AGL scales
every ground offset by 6%. It is not the same number on every flight — that is
the point; it is not a constant you can calibrate out once.

`--detail` additionally walks the flight in 10 s steps, which is the quickest
way to see whether a bag has a stretch where the lidar drops out or saturates
(`range_max` on this sensor is 50 m).
"""

from __future__ import annotations

import logging
import math
import os
import sys

import numpy as np

from bag_source import BagTelemetry

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

IN_FLIGHT_M = 5.0          # below this the lidar is reading its own mount height


def probe(bag_path: str, detail: bool = False):
    name = os.path.basename(os.path.normpath(bag_path))
    try:
        telem = BagTelemetry(bag_path, altitude_source='lidar')
    except Exception as exc:                              # noqa: BLE001
        print(f'{name:<24} unreadable: {exc}')
        return

    alt = telem.altimeter
    if alt is None:
        print(f'{name:<24} no usable lidar — barometer only')
        return

    t, h = alt.samples()
    baro = np.array([telem.alt.value_at(x) if telem.alt.value_at(x) is not None
                     else np.nan for x in t])
    flying = (h > IN_FLIGHT_M) & np.isfinite(baro)
    diff   = h[flying] - baro[flying]

    # How much of the flight the lidar can answer for, sampled at 10 Hz over
    # the stretch the barometer says the drone was airborne.
    grid = np.arange(telem.t_start, telem.t_end, 0.1)
    air  = [x for x in grid
            if (telem.alt.value_at(x) or -1) > IN_FLIGHT_M]
    cover = alt.coverage(air) if air else 0.0

    print(f'{name:<24} {len(t):5d}/{alt.n_scans:<5d} scans  '
          f'in flight {flying.sum():5d}  '
          f'lidar-baro {np.median(diff):+5.2f} m (p16-p84 '
          f'{np.percentile(diff, 84) - np.percentile(diff, 16):4.2f})  '
          f'coverage {100 * cover:4.1f}%  '
          f'fit sigma {alt.median_sigma:.3f} m')

    if not detail:
        return
    print(f'{"t":>8} {"lidar":>8} {"baro":>8} {"diff":>7}')
    for x in np.arange(telem.t_start, telem.t_end, 10.0):
        lid = alt.at(x)
        bar = telem.alt.value_at(x)
        if bar is None or bar < IN_FLIGHT_M:
            continue
        print(f'{x - telem.t_start:8.1f} '
              f'{"--" if lid is None else f"{lid:8.2f}"} {bar:8.2f} '
              f'{"" if lid is None else f"{lid - bar:+7.2f}"}')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', help='rosbag2 directories')
    ap.add_argument('--detail', action='store_true',
                    help='also walk the flight in 10 s steps')
    args = ap.parse_args(argv)

    for bag in args.bags:
        bag = os.path.expanduser(bag)
        if os.path.isdir(bag):
            probe(bag, args.detail)
    return 0


if __name__ == '__main__':
    sys.exit(main())
