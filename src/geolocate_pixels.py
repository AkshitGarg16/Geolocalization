"""
Geolocalise manually annotated pixels.
======================================

Runs the pixel -> GPS projection on hand-annotated points instead of YOLO
detections, reusing the telemetry, intrinsics and extrinsics the main pipeline
uses. Useful for checking the projection against surveyed ground truth and for
re-calibrating the extrinsics without the detector in the loop.

Input CSV — one row per annotated point, header required:

    frame,u,v
    3630,1077,570
    3650,1120,540,left_tower      # optional 4th column: a label

`u`,`v` are pixel coordinates in the FULL frame (top-left origin), not in the
central-60% detection ROI — annotate on the raw video frame.

Usage
-----
    python3 geolocate_pixels.py points.csv
    python3 geolocate_pixels.py points.csv -o out.csv
    python3 geolocate_pixels.py points.csv --ground-truth 28.7537558,77.1155393
    python3 geolocate_pixels.py points.csv --extrinsics nominal   # A/B the old matrix
    python3 geolocate_pixels.py points.csv --alt-source baro      # A/B the height source
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

from bag_source import BagFrameClock, BagTelemetry, resolve_zoom
from geolocalization import CONFIG, haversine_metres
from coordinate_transformer import CoordinateTransformer

INTRINSIC_MATRIX = np.array([
    [1584.02798,    0,        953.27546],
    [   0,       1581.52473,  518.97355],
    [   0,          0,          1       ]
])

OUT_COLUMNS = [
    'frame', 'label', 'u', 'v',
    'target_lat', 'target_lon',
    'drone_lat', 'drone_lon', 'drone_altitude_agl',
    'altitude_source', 'altitude_lidar', 'altitude_baro',
    'roll_deg', 'pitch_deg', 'yaw_deg', 'zoom',
    'error_m',
]


def read_points(path: str) -> list[dict]:
    """Parse the annotation CSV into {frame, u, v, label} rows."""
    if not os.path.exists(path):
        sys.exit(f'Annotation CSV not found: {path}')

    points = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        missing = {'frame', 'u', 'v'} - set(reader.fieldnames or [])
        if missing:
            sys.exit(f'{path}: missing required column(s): {", ".join(sorted(missing))}')

        for lineno, row in enumerate(reader, start=2):
            if not (row.get('frame') or '').strip():
                continue
            try:
                points.append({
                    'frame': int(float(row['frame'])),
                    'u':     float(row['u']),
                    'v':     float(row['v']),
                    'label': (row.get('label') or '').strip(),
                })
            except (TypeError, ValueError):
                print(f'  {path}:{lineno}: unparseable row, skipped', file=sys.stderr)
    return points


def parse_ground_truth(text: str | None):
    if not text:
        return None
    try:
        lat, lon = (float(p) for p in text.split(','))
    except ValueError:
        sys.exit('--ground-truth expects LAT,LON (e.g. 28.7537558,77.1155393)')
    return lat, lon


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('points_csv', nargs='+',
                    help='one or more CSVs of annotated pixels: frame,u,v[,label]')
    ap.add_argument('-o', '--output',
                    help='combine every input into this one CSV '
                         '(default: one <input>_geolocated.csv beside each input)')
    ap.add_argument('--bag', default=CONFIG['bag_path'],
                    help='rosbag2 directory the frame numbers refer to '
                         '(default: the pipeline CONFIG value)')
    ap.add_argument('--ground-truth', metavar='LAT,LON',
                    help='surveyed position; adds an error column and a summary')
    ap.add_argument('--extrinsics', choices=('calibrated', 'nominal'), default='calibrated',
                    help='which camera->body matrix to use (default: calibrated)')
    ap.add_argument('--alt-source', choices=('lidar', 'baro'),
                    default=CONFIG.get('altitude_source', 'lidar'),
                    help='height above ground from the downward lidar on /scan '
                         '(default) or from the barometric rel_alt alone')
    args = ap.parse_args(argv)

    ground_truth = parse_ground_truth(args.ground_truth)

    # Each input keeps its own output unless -o asks for one combined file.
    jobs = []
    for path in args.points_csv:
        points = read_points(path)
        if not points:
            print(f'No usable rows in {path}, skipped', file=sys.stderr)
            continue
        jobs.append((path, points))
    if not jobs:
        sys.exit('No usable rows in any input CSV')

    # Annotated rows are keyed by frame number, which on a bag means the image
    # packet's position in the file — the same number play_video.py puts in the
    # screenshot filename. The clock turns that back into a bag time without
    # decoding any video.
    clock     = BagFrameClock(os.path.expanduser(args.bag))
    # CONFIG's zoom_ratio is 'auto' by default; resolve it the same way the
    # pipeline does, or the projection would try to scale fx by a string.
    zoom = resolve_zoom(
        os.path.expanduser(args.bag),
        CONFIG.get('zoom_ratio', 'auto'),
        CONFIG.get('topics', {}).get('camera_state', '/fcb/state'),
        use_camera_state=CONFIG.get('use_camera_state_zoom', False),
    )
    telemetry = BagTelemetry(os.path.expanduser(args.bag),
                             topics=CONFIG.get('topics'),
                             zoom=zoom,
                             altitude_source=args.alt_source,
                             lidar=CONFIG.get('lidar'))
    print(f'Height above ground: {args.alt_source}')
    transformer = CoordinateTransformer(INTRINSIC_MATRIX)
    if args.extrinsics == 'nominal':
        transformer.R_BODY_CAM = CoordinateTransformer.R_BODY_CAM_NOMINAL
    print(f'Extrinsics: {args.extrinsics}')

    zoom_baseline = CONFIG.get('zoom_baseline', 1.0) or 1.0
    combined, all_errors = [], []

    for src_path, points in jobs:
        rows, errors, skipped = [], [], 0
        for pt in points:
            ts    = clock.time_of(pt['frame'])
            telem = telemetry.at(ts) if ts is not None else None
            if telem is None or not telem.is_complete():
                print(f'  {os.path.basename(src_path)} frame {pt["frame"]}: '
                      f'no usable telemetry, skipped', file=sys.stderr)
                skipped += 1
                continue

            result = transformer.pixel_to_gps(
                u=pt['u'], v=pt['v'],
                drone_lat=telem.lat, drone_lon=telem.lon,
                drone_altitude=telem.alt_agl,
                roll=telem.roll, pitch=telem.pitch, yaw=telem.yaw,
                ground_altitude=0.0,
                zoom=telem.zoom / zoom_baseline,
            )
            if result is None:
                print(f'  {os.path.basename(src_path)} frame {pt["frame"]}: '
                      f'ray does not hit the ground, skipped', file=sys.stderr)
                skipped += 1
                continue

            lat, lon = result
            err = ''
            if ground_truth:
                err = haversine_metres(lat, lon, ground_truth[0], ground_truth[1])
                errors.append(err)
                all_errors.append(err)
                err = f'{err:.3f}'

            rows.append([
                pt['frame'], pt['label'], f'{pt["u"]:.1f}', f'{pt["v"]:.1f}',
                f'{lat:.9f}', f'{lon:.9f}',
                f'{telem.lat:.9f}', f'{telem.lon:.9f}', f'{telem.alt_agl:.3f}',
                telem.alt_source,
                '' if telem.alt_lidar is None else f'{telem.alt_lidar:.3f}',
                '' if telem.alt_baro  is None else f'{telem.alt_baro:.3f}',
                f'{np.rad2deg(telem.roll):.4f}', f'{np.rad2deg(telem.pitch):.4f}',
                f'{np.rad2deg(telem.yaw):.4f}', f'{telem.zoom:.2f}',
                err,
            ])

        if args.output:
            combined.extend(rows)
        else:
            out_path = f'{os.path.splitext(src_path)[0]}_geolocated.csv'
            _write(out_path, rows)
            print(f'{os.path.basename(src_path)}: geolocalised {len(rows)} point(s)'
                  + (f', skipped {skipped}' if skipped else '')
                  + f' -> {out_path}')
            _summarise(errors)

    if args.output:
        _write(args.output, combined)
        print(f'Geolocalised {len(combined)} point(s) -> {args.output}')

    if all_errors and (args.output or len(jobs) > 1):
        print('Overall', end=' ')
        _summarise(all_errors)

    return 0


def _write(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(OUT_COLUMNS)
        writer.writerows(rows)


def _summarise(errors):
    if not errors:
        return
    e = np.array(errors)
    print(f'  error vs ground truth: mean {e.mean():.2f} m  median {np.median(e):.2f} m  '
          f'min {e.min():.2f} m  max {e.max():.2f} m  rmse {np.sqrt((e ** 2).mean()):.2f} m')


if __name__ == '__main__':
    sys.exit(main())
