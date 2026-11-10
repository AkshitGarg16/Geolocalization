"""
Check what zoom a bag reports, and from where
=============================================
The camera's zoom lives on `/fcb/state` as a `fcb_interfaces/msg/CameraState`,
and that package is not installed, so the pipeline normally takes the zoom from
the bag directory name instead (`25m_1x_2ms` -> 1x). This tool shows both
sources side by side so you can see whether they agree.

    python3 probe_zoom.py ~/rosbags/*

**Run this on a bag you know is NOT 1x.** Both bags available when this was
written were shot at 1x, so the `/fcb/state` decoder reading 1.0 proved
nothing — a field that is always 1.0 looks identical to a field that is
coincidentally 1.0. If a 2x bag makes it print 2.0, the decoder is finding the
real field and you can switch `use_camera_state_zoom` on in CONFIG. If it
prints 1.0 or nothing there, leave it off and keep naming bags with their zoom.

A wrong zoom is not a noisy error. It scales the focal length, so every
coordinate is displaced by a consistent, plausible-looking amount.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from bag_source import (DEFAULT_TOPICS, zoom_from_camera_state, zoom_from_name)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', help='one or more rosbag2 directories')
    ap.add_argument('--topic', default=DEFAULT_TOPICS['camera_state'],
                    help=f'camera state topic (default: {DEFAULT_TOPICS["camera_state"]})')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    print(f'{"bag":32s} {"from name":>10s} {"from /fcb/state":>16s}   detail')
    print('-' * 84)

    agree = disagree = 0
    for bag in args.bags:
        bag = os.path.expanduser(bag.rstrip('/'))
        if not os.path.isdir(bag):
            continue
        name = os.path.basename(bag)

        by_name = zoom_from_name(bag)
        try:
            got = zoom_from_camera_state(bag, args.topic)
        except Exception as exc:                          # noqa: BLE001
            got = None
            detail = f'error: {exc}'
        else:
            detail = (f'offset {got[2]}, {got[1]} distinct value(s)'
                      if got else 'no single zoom-shaped field found')

        by_state = got[0] if got else None
        if by_name is not None and by_state is not None:
            if abs(by_name - by_state) < 1e-6:
                agree += 1
                detail += '  [agree]'
            else:
                disagree += 1
                detail += '  [DISAGREE]'

        print(f'{name:32s} '
              f'{("-" if by_name is None else f"{by_name:g}x"):>10s} '
              f'{("-" if by_state is None else f"{by_state:g}x"):>16s}   {detail}')

    print()
    if disagree:
        print(f'{disagree} bag(s) disagree. Trust the name, set `zoom_ratio` '
              f'explicitly, and leave use_camera_state_zoom off.')
    elif agree:
        names = 'bag' if agree == 1 else 'bags'
        print(f'{agree} {names} agree. If any of them is NOT 1x, the /fcb/state '
              f'decoder is finding the real field and you can set '
              f'use_camera_state_zoom: True in CONFIG.')
    else:
        print('Nothing to cross-check. Name bags like `25m_2x_2ms` so the zoom '
              'travels with the recording.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
