"""
Click the target on saved frames
================================

Opens each saved frame, you click the target, and the pixel is written straight
to a `frame,u,v,label` CSV that `geolocate_pixels.py` reads. No image editor
needed and nothing to install — this uses the OpenCV you already have.

The frames are never modified or moved. Each person's directory gets its own
CSV, written beside their frames:

    outputs/frames/surya/frame_002867.png
    outputs/frames/surya/surya_points.csv     <- written here

The label likewise comes from the directory name, so `surya/` is labelled
`surya` automatically.

    python3 click_points.py                    # every frame under outputs/frames
    python3 click_points.py --dir outputs/frames/surya
    python3 click_points.py --csv-name points.csv

Controls
--------
  left click    mark the target (one mark per image; clicking again moves it)
  right click   remove the mark on this image
  n / space     next image      p   previous image
  m             toggle the magnifier
  q / Esc       save and quit

Progress is saved on every click, so quitting and restarting resumes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2

WINDOW = 'click the target - n: next, p: prev, q: save and quit'
MAG_SIZE = 180
MAG_ZOOM = 8


def find_frame_number(filename: str) -> int | None:
    digits = ''.join(ch if ch.isdigit() else ' ' for ch in os.path.basename(filename))
    parts = digits.split()
    return int(parts[-1]) if parts else None


def _text(img, s, org, colour=(255, 255, 255), scale=0.6):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)


class Clicker:

    def __init__(self, images, csv_name, label_override):
        self.images   = images
        self.csv_name = csv_name
        self.label_override = label_override
        self.idx      = 0
        self.marks    = {}          # image path -> (u, v)
        self.cursor   = None
        self.magnify  = True
        self.frame    = None
        self.scale    = 1.0         # display scale, so clicks map back to full res

        self._load_existing()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        self._load_image()
        cv2.imshow(WINDOW, self.frame)
        cv2.waitKey(1)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    # ------------------------------------------------------------------ io --

    def label_for(self, path):
        return self.label_override or os.path.basename(os.path.dirname(path))

    def csv_for(self, path):
        """Each person's CSV lives in that person's own frame directory."""
        directory = os.path.dirname(path)
        name = self.csv_name or f'{os.path.basename(directory)}_points.csv'
        return os.path.join(directory, name)

    def directories(self):
        """Every frame directory in play, in first-seen order."""
        seen = []
        for path in self.images:
            d = os.path.dirname(path)
            if d not in seen:
                seen.append(d)
        return seen

    def _load_existing(self):
        """Reload marks from each directory's own CSV, so a session can resume."""
        by_csv = {}
        for path in self.images:
            csv_path = self.csv_for(path)
            if csv_path not in by_csv:
                rows = {}
                if os.path.exists(csv_path):
                    with open(csv_path, newline='') as f:
                        for row in csv.DictReader(f):
                            try:
                                rows[int(float(row['frame']))] = (float(row['u']),
                                                                  float(row['v']))
                            except (TypeError, ValueError, KeyError):
                                continue
                by_csv[csv_path] = rows
            mark = by_csv[csv_path].get(find_frame_number(path))
            if mark:
                self.marks[path] = mark
        if self.marks:
            print(f'Resuming: {len(self.marks)} point(s) already marked '
                  f'across {len(by_csv)} CSV file(s)')

    def save(self):
        """Write one CSV per directory, beside that directory's frames."""
        by_csv: dict[str, list] = {}
        for path, (u, v) in self.marks.items():
            frame = find_frame_number(path)
            if frame is None:
                continue
            by_csv.setdefault(self.csv_for(path), []).append(
                (frame, u, v, self.label_for(path)))

        # A directory whose marks were all removed still gets an empty CSV, so a
        # stale file never lingers with points the user has deleted.
        for directory in self.directories():
            probe = os.path.join(directory, 'x.png')
            by_csv.setdefault(self.csv_for(probe), [])

        total = 0
        for csv_path, rows in by_csv.items():
            rows.sort(key=lambda r: r[0])
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frame', 'u', 'v', 'label'])
                for frame, u, v, label in rows:
                    w.writerow([frame, f'{u:.1f}', f'{v:.1f}', label])
            total += len(rows)
        return total

    # --------------------------------------------------------------- image --

    def _load_image(self):
        path = self.images[self.idx]
        img = cv2.imread(path)
        if img is None:
            print(f'  cannot read {path}')
            self.frame = None
            return
        self.frame = img

    def _step(self, delta):
        self.idx = (self.idx + delta) % len(self.images)
        self._load_image()

    # --------------------------------------------------------------- mouse --

    def _on_mouse(self, event, x, y, flags, _param):
        self.cursor = (x, y)
        path = self.images[self.idx]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.marks[path] = (float(x), float(y))
            n = self.save()
            print(f'  {os.path.basename(path)}  u={x} v={y}  '
                  f'label={self.label_for(path)}   ({n} saved)')
        elif event == cv2.EVENT_RBUTTONDOWN and path in self.marks:
            del self.marks[path]
            self.save()
            print(f'  {os.path.basename(path)}  mark removed')

    # ---------------------------------------------------------------- draw --

    def render(self):
        vis = self.frame.copy()
        path = self.images[self.idx]
        h, w = vis.shape[:2]

        mark = self.marks.get(path)
        if mark:
            u, v = int(mark[0]), int(mark[1])
            cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(vis, (u, v), 14, (0, 0, 255), 2)

        header = [
            f'[{self.idx + 1}/{len(self.images)}]  {self.label_for(path)}  '
            f'frame {find_frame_number(path)}',
            f'This image: {"u=%d v=%d" % (int(mark[0]), int(mark[1])) if mark else "not marked"}',
            f'Marked {len(self.marks)} of {len(self.images)}  ->  '
            f'{os.path.basename(self.csv_for(path))}',
        ]
        for i, line in enumerate(header):
            _text(vis, line, (10, 30 + i * 26))
        _text(vis, 'left click mark   right click remove   n next   p prev   q save+quit',
              (10, h - 16), (200, 200, 200), 0.5)

        if self.magnify and self.cursor:
            self._draw_magnifier(vis)
        return vis

    def _draw_magnifier(self, vis):
        h, w = vis.shape[:2]
        cxp, cyp = self.cursor
        if not (0 <= cxp < w and 0 <= cyp < h):
            return
        half = MAG_SIZE // (2 * MAG_ZOOM)
        x0, y0 = max(0, cxp - half), max(0, cyp - half)
        x1, y1 = min(w, cxp + half), min(h, cyp + half)
        patch = self.frame[y0:y1, x0:x1]
        if patch.size == 0:
            return
        mag = cv2.resize(patch, (MAG_SIZE, MAG_SIZE), interpolation=cv2.INTER_NEAREST)
        px = int((cxp - x0) / max(x1 - x0, 1) * MAG_SIZE)
        py = int((cyp - y0) / max(y1 - y0, 1) * MAG_SIZE)
        cv2.line(mag, (px, 0), (px, MAG_SIZE), (0, 255, 255), 1)
        cv2.line(mag, (0, py), (MAG_SIZE, py), (0, 255, 255), 1)
        cv2.rectangle(mag, (0, 0), (MAG_SIZE - 1, MAG_SIZE - 1), (255, 255, 255), 1)

        # Keep the inset away from the cursor, the header and the help line.
        mx = w - MAG_SIZE - 10 if cxp < w // 2 else 10
        my = 120 if cyp > h // 2 else h - MAG_SIZE - 60
        vis[my:my + MAG_SIZE, mx:mx + MAG_SIZE] = mag
        _text(vis, f'{cxp},{cyp}', (mx, my - 8), (0, 255, 255), 0.5)

    # ---------------------------------------------------------------- loop --

    def run(self):
        print(__doc__.split('Controls')[1])
        print(f'{len(self.images)} image(s) across {len(self.directories())} '
              f'directory(ies); one CSV written per directory\n')

        while True:
            if self.frame is None:
                self._step(1)
                continue
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key in (ord('n'), ord(' '), 83):
                self._step(1)
            elif key in (ord('p'), 81):
                self._step(-1)
            elif key == ord('m'):
                self.magnify = not self.magnify

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

        n = self.save()
        cv2.destroyAllWindows()
        print(f'\nSaved {n} point(s):')
        for directory in self.directories():
            csv_path = self.csv_for(os.path.join(directory, 'x.png'))
            count = sum(1 for p in self.marks if os.path.dirname(p) == directory)
            print(f'  {count:3d} -> {csv_path}')
        unmarked = [p for p in self.images if p not in self.marks]
        if unmarked:
            print(f'{len(unmarked)} image(s) left unmarked:')
            for p in unmarked:
                print(f'  {os.path.relpath(p)}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='/home/akshit/geolocalization/outputs/frames',
                    help='directory of saved frames (searched recursively)')
    ap.add_argument('--csv-name',
                    help='CSV filename written inside each frame directory '
                         '(default: <directory name>_points.csv)')
    ap.add_argument('--label', help='override the label (default: sub-directory name)')
    args = ap.parse_args(argv)

    images = []
    for dirpath, _dirnames, filenames in os.walk(args.dir):
        for fn in sorted(filenames):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append(os.path.join(dirpath, fn))
    images.sort()
    if not images:
        sys.exit(f'No images found under {args.dir}')

    Clicker(images, args.csv_name, args.label).run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
