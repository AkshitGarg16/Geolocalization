"""
Recover annotated pixels from marked-up frame images
====================================================

Workflow: open a saved frame in any image editor, draw a mark on the target
(a dot, an X, a circle — anything, any colour), and save it as a copy under a
separate directory. This script diffs each marked copy against the pristine
original, finds what you drew, and writes its centre as a `frame,u,v,label`
row for `geolocate_pixels.py`.

    outputs/frames/<name>/frame_003360.png       <- pristine, from play_video.py
    outputs/annotated/<name>/frame_003360.png    <- your marked copy

The label comes from the sub-directory name, so each person's folder is
labelled automatically. Save marked copies as PNG: JPEG re-compression adds
noise everywhere, which makes the diff far less precise.

    python3 extract_annotations.py
    python3 extract_annotations.py --annotated DIR --originals DIR -o points.csv
    python3 extract_annotations.py --preview          # write check images

Then:

    python3 geolocate_pixels.py points.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

DEFAULT_ORIGINALS = '/home/akshit/geolocalization/outputs/frames'
DEFAULT_ANNOTATED = '/home/akshit/geolocalization/outputs/annotated'

# A pixel counts as "drawn on" when any channel moves by more than this. Well
# above PNG-identical noise, low enough to catch a thin or anti-aliased stroke.
DIFF_THRESHOLD = 40

# Blobs smaller than this are treated as compression speckle, not a mark.
MIN_BLOB_AREA = 4


def find_frame_number(filename: str) -> int | None:
    """Pull the frame index out of a name like 'frame_003360.png'."""
    digits = ''.join(ch if ch.isdigit() else ' ' for ch in os.path.basename(filename))
    parts = digits.split()
    return int(parts[-1]) if parts else None


def locate_original(marked_path: str, annotated_root: str, originals_root: str):
    """Find the pristine frame matching a marked copy."""
    rel = os.path.relpath(marked_path, annotated_root)
    candidate = os.path.join(originals_root, rel)
    if os.path.exists(candidate):
        return candidate

    # Fall back to matching on basename anywhere under the originals root.
    base = os.path.basename(marked_path)
    for dirpath, _dirnames, filenames in os.walk(originals_root):
        if base in filenames:
            return os.path.join(dirpath, base)

    # Last resort: same frame number, any extension.
    frame = find_frame_number(marked_path)
    if frame is not None:
        for dirpath, _dirnames, filenames in os.walk(originals_root):
            for fn in filenames:
                if find_frame_number(fn) == frame:
                    return os.path.join(dirpath, fn)
    return None


def marks_in(original, marked):
    """Centroids of everything drawn onto `marked`, largest blob first."""
    if original.shape != marked.shape:
        return None, 'size differs from the original (was it resized or cropped?)'

    diff = cv2.absdiff(original, marked).max(axis=2)
    mask = (diff > DIFF_THRESHOLD).astype(np.uint8)
    if mask.sum() == 0:
        return [], 'no mark found — is this the unedited frame?'

    # Join up strokes that are close together (an X, a ring, a dashed circle).
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, n):                          # 0 is the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA:
            continue
        cx, cy = centroids[i]
        found.append({'u': float(cx), 'v': float(cy), 'area': int(area)})
    found.sort(key=lambda m: -m['area'])
    return found, None


def write_preview(path, marked, marks):
    vis = marked.copy()
    for m in marks:
        u, v = int(round(m['u'])), int(round(m['v']))
        cv2.drawMarker(vis, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(vis, (u, v), 24, (0, 255, 255), 2)
        cv2.putText(vis, f'{u},{v}', (u + 28, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(vis, f'{u},{v}', (u + 28, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, vis)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotated', default=DEFAULT_ANNOTATED,
                    help=f'root of your marked copies (default: {DEFAULT_ANNOTATED})')
    ap.add_argument('--originals', default=DEFAULT_ORIGINALS,
                    help=f'root of the pristine frames (default: {DEFAULT_ORIGINALS})')
    ap.add_argument('-o', '--output',
                    default='/home/akshit/geolocalization/outputs/points.csv',
                    help='annotation CSV to write')
    ap.add_argument('--label', help='override the label (default: sub-directory name)')
    ap.add_argument('--preview', action='store_true',
                    help='write check images showing what was detected')
    ap.add_argument('--all-marks', action='store_true',
                    help='keep every mark per image, not just the largest')
    args = ap.parse_args(argv)

    if not os.path.isdir(args.annotated):
        sys.exit(f'No annotated directory: {args.annotated}\n'
                 f'Create it, copy frames in from {args.originals}, and mark them.')

    images = []
    for dirpath, _dirnames, filenames in os.walk(args.annotated):
        for fn in sorted(filenames):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                images.append(os.path.join(dirpath, fn))
    if not images:
        sys.exit(f'No images under {args.annotated}')

    rows, problems = [], []
    for path in sorted(images):
        rel = os.path.relpath(path, args.annotated)
        frame = find_frame_number(path)
        if frame is None:
            problems.append(f'{rel}: no frame number in the filename')
            continue

        original_path = locate_original(path, args.annotated, args.originals)
        if original_path is None:
            problems.append(f'{rel}: no matching original frame found')
            continue

        original, marked = cv2.imread(original_path), cv2.imread(path)
        if original is None or marked is None:
            problems.append(f'{rel}: could not read image')
            continue

        marks, err = marks_in(original, marked)
        if err:
            problems.append(f'{rel}: {err}')
            continue

        label = args.label or os.path.basename(os.path.dirname(path))
        keep = marks if args.all_marks else marks[:1]
        for m in keep:
            rows.append({'frame': frame, 'u': m['u'], 'v': m['v'], 'label': label})
            print(f'  {rel:38s} frame {frame:6d}  u={m["u"]:7.1f} v={m["v"]:7.1f}  '
                  f'label={label}  ({m["area"]} px mark)')
        if len(marks) > 1 and not args.all_marks:
            print(f'      note: {len(marks)} marks in this image, kept the largest '
                  f'(--all-marks keeps all)')

        if args.preview:
            write_preview(os.path.join(os.path.dirname(args.output), 'preview', rel),
                          marked, keep)

    if rows:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'u', 'v', 'label'])
            for r in sorted(rows, key=lambda r: (r['label'], r['frame'])):
                w.writerow([r['frame'], f'{r["u"]:.1f}', f'{r["v"]:.1f}', r['label']])
        print(f'\nWrote {len(rows)} point(s) -> {args.output}')
        if args.preview:
            print(f'Preview images -> {os.path.join(os.path.dirname(args.output), "preview")}')

    if problems:
        print(f'\n{len(problems)} image(s) skipped:')
        for p in problems:
            print(f'  {p}')

    return 0 if rows else 1


if __name__ == '__main__':
    sys.exit(main())
