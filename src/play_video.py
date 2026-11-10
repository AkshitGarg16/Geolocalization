"""
Bag video player with frame numbers + screenshot capture
========================================================
Plays the imagery out of a rosbag2 directory with the frame number, bag time
and telemetry burned on screen, and saves a screenshot of whatever frame is
showing when you hit SPACE.

Screenshots are saved CLEAN — without the overlay — so the pixel coordinates
you read off them are the real full-frame coordinates the pipeline expects.
The frame number is in the filename and in the on-screen overlay.

    python3 play_video.py                        # bag from the pipeline CONFIG
    python3 play_video.py ~/rosbags/25m_1x_2ms

Then open a saved frame in any image viewer, note the u,v of your target, and
feed `frame,u,v` to geolocate_pixels.py to get its GPS position.

Seeking
-------
A bag is a sequential H.265 stream, so there is no random access. The last
`--cache` frames are held in memory, which is what backward stepping moves
through; stepping back beyond the cache reopens the bag and decodes forward to
the target, which takes a moment. Forward seeks always decode forward.

Controls
--------
  SPACE       save a screenshot of the current frame
  p           pause / resume
  . / ,       step forward / back one frame (pauses)
  > / <       step forward / back 10 frames
  g           jump to a frame number (typed in the terminal)
  + / -       faster / slower playback
  q / Esc     quit
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone

import cv2

from bag_source import BagTelemetry, BagVideo, DEFAULT_TOPICS, resolve_zoom

# ASCII only: the OpenCV Qt backend cannot look a window up by a name
# containing non-ASCII characters, which breaks createTrackbar/imshow.
WINDOW = 'bag video - SPACE: screenshot, p: pause, q: quit'

HELP_LINES = [
    'SPACE screenshot   p pause   . , step 1   > < step 10',
    'g goto frame   + - speed   q quit',
]


def _text(img, s, org, colour=(255, 255, 255), scale=0.6):
    """White text with a dark outline, readable over any background."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)


class BagFrames:
    """
    Sequential frame access over a bag, with a bounded cache so that short
    backward steps do not force a re-decode from the top of the file.
    """

    def __init__(self, bag_path, topic=DEFAULT_TOPICS['image'], cache=60):
        self.bag_path = bag_path
        self.topic    = topic
        self.cache_n  = max(2, cache)
        self._cache   = OrderedDict()      # index -> (ts, frame)
        self._restart()

    def _restart(self):
        self._iter    = BagVideo(self.bag_path, self.topic).frames()
        self._last_id = -1

    def _advance(self):
        """Pull one frame off the decoder into the cache. False at end of bag."""
        for index, ts, frame in self._iter:
            if frame is None:
                continue
            self._cache[index] = (ts, frame)
            while len(self._cache) > self.cache_n:
                self._cache.popitem(last=False)
            self._last_id = index
            return True
        return False

    def get(self, index):
        """
        (index, ts, frame) at or after `index`, or None past the end of the bag.

        The index that comes back is not always the one asked for: not every
        image packet decodes to a picture. The first 17 packets on these bags
        precede the stream's parameter sets and produce nothing at all, and a
        corrupt packet mid-bag is skipped too, so asking for frame 0 — which is
        what the player does on startup — has to fall forward to frame 17 or
        the player has nothing to show and quits on the spot.

        The returned index is the packet's real position in the bag, which is
        what the screenshot is named after and what geolocate_pixels.py turns
        back into a capture time, so the caller must use it rather than the
        number it asked for.
        """
        if index < 0:
            index = 0
        if index in self._cache:
            return (index, *self._cache[index])

        # Behind the cache: the stream cannot rewind, so start it over.
        if index <= self._last_id:
            self._cache.clear()
            self._restart()

        while self._last_id < index:
            if not self._advance():
                return None
        if index in self._cache:
            return (index, *self._cache[index])
        later = [i for i in self._cache if i >= index]
        if not later:
            return None
        got = min(later)
        return (got, *self._cache[got])


class Player:

    def __init__(self, bag_path, out_dir, start_frame, cache, zoom):
        self.bag_path = bag_path
        self.out_dir  = out_dir
        self.frames   = BagFrames(bag_path, cache=cache)
        self.fps      = 60.0                   # display pacing only

        os.makedirs(out_dir, exist_ok=True)

        # Telemetry is optional — the player still works without it.
        self.telemetry = None
        try:
            self.telemetry = BagTelemetry(bag_path, zoom=zoom)
        except Exception as exc:                          # noqa: BLE001
            print(f'Telemetry not loaded ({exc}); timestamps will be bag-time only')

        self.frame_idx = -1
        self.frame     = None
        self.ts        = None
        self.t0        = None
        self.playing   = True
        self.speed     = 1.0
        self.n_saved   = 0

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        self._seek(max(0, start_frame))

    # ------------------------------------------------------------ playback --

    def _seek(self, idx):
        got = self.frames.get(max(0, idx))
        if got is None:
            self.playing = False
            return
        index, ts, frame = got
        if self.t0 is None:
            self.t0 = ts
        self.frame_idx, self.ts, self.frame = index, ts, frame

    # ------------------------------------------------------------- capture --

    def screenshot(self):
        """Save the current frame with no overlay drawn on it."""
        path = os.path.join(self.out_dir, f'frame_{self.frame_idx:06d}.png')
        cv2.imwrite(path, self.frame)
        self.n_saved += 1
        print(f'  saved frame {self.frame_idx} -> {path}')
        return path

    # ---------------------------------------------------------------- draw --

    def _telemetry_lines(self):
        lines = []
        if self.ts is not None and self.t0 is not None:
            lines.append(f'Bag time: {self.ts - self.t0:8.3f} s')
            try:
                utc = datetime.fromtimestamp(self.ts, tz=timezone.utc)
                lines.append(f'UTC: {utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}')
            except (OverflowError, OSError, ValueError):
                pass

        if self.telemetry is not None and self.ts is not None:
            telem = self.telemetry.at(self.ts)
            if telem is not None:
                lines += [
                    f'Lat: {telem.lat:.7f}   Lon: {telem.lon:.7f}',
                    f'Alt: {telem.alt_agl:.1f} m AGL ({telem.alt_source})'
                    f'   Hdg: {telem.compass:.1f} deg',
                ]
            else:
                lines.append('(no telemetry at this bag time)')
        return lines

    def render(self):
        vis = self.frame.copy()
        header = [f'Frame: {self.frame_idx}']
        header += self._telemetry_lines()
        header.append(f'{"PLAYING" if self.playing else "PAUSED"}   {self.speed:g}x'
                      f'   saved: {self.n_saved}')

        for i, line in enumerate(header):
            _text(vis, line, (10, 30 + i * 26))

        for i, line in enumerate(HELP_LINES):
            _text(vis, line,
                  (10, vis.shape[0] - 14 - (len(HELP_LINES) - 1 - i) * 22),
                  (200, 200, 200), 0.5)
        return vis

    # ---------------------------------------------------------------- loop --

    def run(self):
        print(__doc__.split('Controls')[1])
        print(f'Bag: {self.bag_path}')
        print(f'Screenshots -> {self.out_dir}\n')

        while self.frame is not None:
            cv2.imshow(WINDOW, self.render())

            delay = max(1, int(1000.0 / (self.fps * self.speed))) if self.playing else 20
            key = cv2.waitKey(delay) & 0xFF

            if self.playing:
                self._seek(self.frame_idx + 1)

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                self.screenshot()
            elif key == ord('p'):
                self.playing = not self.playing
            elif key in (ord('.'), 83):              # 83 = Right arrow
                self.playing = False; self._seek(self.frame_idx + 1)
            elif key in (ord(','), 81):              # 81 = Left arrow
                self.playing = False; self._seek(self.frame_idx - 1)
            elif key == ord('>'):
                self.playing = False; self._seek(self.frame_idx + 10)
            elif key == ord('<'):
                self.playing = False; self._seek(self.frame_idx - 10)
            elif key in (ord('+'), ord('=')):
                self.speed = min(self.speed * 2, 8.0)
            elif key in (ord('-'), ord('_')):
                self.speed = max(self.speed / 2, 0.125)
            elif key == ord('g'):
                self.playing = False
                try:
                    t = input('  go to frame: ').strip()
                except (EOFError, KeyboardInterrupt):
                    t = ''
                if t.lstrip('-').isdigit():
                    print('  seeking …')
                    self._seek(int(t))

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        print(f'\n{self.n_saved} screenshot(s) in {self.out_dir}')


def main(argv=None):
    from geolocalization import CONFIG

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', nargs='?', default=CONFIG['bag_path'],
                    help='rosbag2 directory (default: the pipeline CONFIG value)')
    ap.add_argument('-o', '--out-dir',
                    default='/home/akshit/geolocalization/outputs/frames',
                    help='where screenshots are written (default: outputs/frames)')
    ap.add_argument('-f', '--start-frame', type=int, default=0,
                    help='frame to open at (default: 0)')
    ap.add_argument('--cache', type=int, default=60,
                    help='frames held in memory for backward stepping '
                         '(default: 60, ~370 MB at 1080p)')
    ap.add_argument('--zoom', default=CONFIG.get('zoom_ratio', 'auto'),
                    help="optical zoom ratio, or 'auto' to read it from the "
                         'bag name (default: the pipeline CONFIG value)')
    args = ap.parse_args(argv)

    # CONFIG's zoom_ratio is 'auto' by default. Resolve it the way the pipeline
    # does; a float-typed argument cannot even hold it, because argparse runs
    # `type` over a string default as well as over what you type.
    zoom = resolve_zoom(
        os.path.expanduser(args.bag),
        args.zoom,
        CONFIG.get('topics', {}).get('camera_state', '/fcb/state'),
        use_camera_state=CONFIG.get('use_camera_state_zoom', False),
    )

    Player(os.path.expanduser(args.bag), args.out_dir,
           args.start_frame, args.cache, zoom).run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
