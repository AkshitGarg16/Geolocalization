"""
Detection Clusterer + Bokeh Map  (two-stage)
=============================================
Reads raw detections.csv, collapses all detections of the same person into a
single GPS point, and renders the result as a standalone Bokeh map on satellite
imagery.  Same two-stage clustering as dbscan.py; the map is the bokeh_plot.py
job done with bokeh instead of folium.

Install:
    pip install scikit-learn numpy bokeh

Usage:
    python plot_clusters.py \
        --input       ../outputs/detections.csv \
        --output      ../outputs/cluster_map.html \
        --csv_output  ../outputs/clustered.csv \
        --eps         1.0 \
        --min_samples 3 \
        --merge_dist  5.0

Why two stages?
---------------
A single DBSCAN with large eps (e.g. 3 m) merges people who stand only 3-5 m
apart.  A single DBSCAN with small eps over-splits one person's noisy
detection cloud into 2-3 fragments.  Two stages handle both:

  Stage 1 - tight DBSCAN (eps=1.0 m)
      Finds dense cores.  May over-split one person into small fragments.

  Stage 2 - absorb small fragments (merge_dist=5.0 m)
      Any cluster smaller than the median cluster size whose centroid lies
      within merge_dist of a larger cluster is absorbed into that larger one.
      Repeats until stable.

Parameters:
    --eps               Stage-1 DBSCAN radius (m).              [default: 1.0]
    --min_samples       Min detections to form a core.          [default: 3]
    --merge_dist        Stage-2 absorption radius (m).          [default: 5.0]
                        Set to 0 to skip Stage 2 entirely.
    --min_alt           Drop detections below this drone AGL.   [default: 5.0]
                        Near the ground the projection ray runs almost parallel
                        to the ground plane, so the intersection collapses onto
                        the drone's own position -- those fixes are meaningless.
    --min_cluster_size  Drop clusters with fewer than N points. [default: 20]
    --min_conf          Drop clusters whose best conf < this.   [default: 0.0]
    --tiles             esri (satellite) or osm (street).       [default: esri]

Note on basemaps: CARTO now watermarks its keyless tiles with "API KEY
REQUIRED", so this uses Esri World Imagery (satellite, no key) by default.
"""

import argparse
import csv
import math
import os

import numpy as np
from sklearn.cluster import DBSCAN

from bokeh.layouts import column
from bokeh.models import (ColumnDataSource, DataTable, Div, HoverTool, LabelSet,
                          NumberFormatter, TableColumn, WMTSTileSource)
from bokeh.plotting import figure, output_file, save

# -- palette (validated all-pairs vs imagery surface; the contrast WARN is
#    relieved by the direct labels + table view, per the relief rule) ---------
BLUE   = '#3987e5'   # person clusters + their member detections
RED    = '#d03b3b'   # rejected detections
AQUA   = '#1baf7a'   # drone path
HALO   = '#ffffff'   # ring that lifts marks off the imagery
SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_2     = '#52514e'
INK_MUTED = '#898781'

MERCATOR_R = 6_378_137.0

TILES = {
    'esri': ('https://server.arcgisonline.com/ArcGIS/rest/services/'
             'World_Imagery/MapServer/tile/{Z}/{Y}/{X}',
             'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics'),
    'osm':  ('https://tile.openstreetmap.org/{Z}/{X}/{Y}.png',
             '&copy; OpenStreetMap contributors'),
}


# -- helpers ------------------------------------------------------------------

def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV not found: {path}')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No data in {path}')
    return rows


def to_metres(lats, lons, cos_lat):
    return np.column_stack([lats * 111111, lons * 111111 * cos_lat])


def centroid(lats, lons, mask):
    return np.median(lats[mask]), np.median(lons[mask])


def dist_m(lat1, lon1, lat2, lon2, cos_lat):
    return math.sqrt(((lat1 - lat2) * 111111) ** 2 +
                     ((lon1 - lon2) * 111111 * cos_lat) ** 2)


def to_mercator(lats, lons):
    """WGS84 -> EPSG:3857, the projection the tile sources use."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    x = MERCATOR_R * np.radians(lons)
    y = MERCATOR_R * np.log(np.tan(np.pi / 4 + np.radians(lats) / 2))
    return x, y


# -- Stage 1: DBSCAN ----------------------------------------------------------

def stage1(lats, lons, cos_lat, eps, min_samples):
    labels = DBSCAN(
        eps=eps, min_samples=min_samples
    ).fit_predict(to_metres(lats, lons, cos_lat))
    return labels


# -- Stage 2: absorb small clusters into nearby large ones ---------------------

def stage2(labels, lats, lons, cos_lat, merge_dist):
    if merge_dist <= 0:
        return labels

    labels = labels.copy()
    changed = True

    while changed:
        changed = False
        ids = [l for l in set(labels) if l != -1]
        if not ids:
            break

        sizes = {l: int((labels == l).sum()) for l in ids}
        med_size = float(np.median(list(sizes.values())))

        smalls = [l for l, s in sizes.items() if s <= med_size]
        larges = [l for l, s in sizes.items() if s >  med_size]

        if not smalls or not larges:
            break

        for s in smalls:
            mask_s = labels == s
            clat_s, clon_s = centroid(lats, lons, mask_s)

            # nearest large cluster
            best_d, best_l = float('inf'), None
            for lg in larges:
                mask_l = labels == lg
                clat_l, clon_l = centroid(lats, lons, mask_l)
                d = dist_m(clat_s, clon_s, clat_l, clon_l, cos_lat)
                if d < best_d:
                    best_d, best_l = d, lg

            if best_d <= merge_dist:
                labels[labels == s] = best_l
                changed = True
                break   # restart -- centroids have shifted

    # renumber contiguously
    remap, counter = {}, 0
    for l in labels:
        if l != -1 and l not in remap:
            remap[l] = counter
            counter += 1
    return np.array([remap[l] if l != -1 else -1 for l in labels])


# -- Summarise one cluster -> one record --------------------------------------

def summarise(rows, mask, lats, lons, cos_lat, person_idx):
    c_rows = [r for r, m in zip(rows, mask) if m]
    c_lats = lats[mask]
    c_lons = lons[mask]
    confs  = np.array([float(r['confidence'])         for r in c_rows])
    alts   = np.array([float(r['drone_altitude_agl']) for r in c_rows])
    times  = np.array([float(r['timestamp_sec'])      for r in c_rows])

    rep_lat = float(np.median(c_lats))
    rep_lon = float(np.median(c_lons))
    rms_m   = math.sqrt((c_lats.std() * 111111)          ** 2 +
                        (c_lons.std() * 111111 * cos_lat) ** 2)

    best = c_rows[int(np.argmax(confs))]

    return {
        'timestamp_sec':           best['timestamp_sec'],
        'person_idx':              person_idx,
        'person_lat':              f'{rep_lat:.9f}',
        'person_lon':              f'{rep_lon:.9f}',
        'drone_lat':               best['drone_lat'],
        'drone_lon':               best['drone_lon'],
        'drone_altitude_agl':      best['drone_altitude_agl'],
        'roll_deg':                best['roll_deg'],
        'pitch_deg':               best['pitch_deg'],
        'yaw_deg':                 best['yaw_deg'],
        'confidence':              f'{confs.max():.4f}',
        'num_detections_in_frame': best['num_detections_in_frame'],
        'cluster_size':            len(c_rows),
        'cluster_rms_m':           f'{rms_m:.3f}',
        'cluster_alt_mean_m':      f'{alts.mean():.3f}',
        'duration_s':              f'{times.max() - times.min():.1f}',
        'mean_conf':               f'{confs.mean():.4f}',
    }


# -- Write output CSV ---------------------------------------------------------

FIELDS = [
    'timestamp_sec', 'person_idx',
    'person_lat', 'person_lon',
    'drone_lat', 'drone_lon', 'drone_altitude_agl',
    'roll_deg', 'pitch_deg', 'yaw_deg',
    'confidence', 'num_detections_in_frame',
    'cluster_size', 'cluster_rms_m', 'cluster_alt_mean_m',
]

def write(path, summaries):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s[k] for k in FIELDS})


# -- Bokeh map ----------------------------------------------------------------

def build_map(kept, member, reject, drone, args, stats):
    px, py = to_mercator([float(s['person_lat']) for s in kept],
                         [float(s['person_lon']) for s in kept])
    mx, my = to_mercator(member['lat'], member['lon'])
    rx, ry = to_mercator(reject['lat'], reject['lon'])
    dx, dy = to_mercator(drone['lat'],  drone['lon'])

    sizes  = np.array([s['cluster_size'] for s in kept], dtype=float)
    radius = 16 + 24 * np.sqrt(sizes / sizes.max())

    src_person = ColumnDataSource(dict(
        x=px, y=py, size=radius,
        label=[f"P{s['person_idx']}" for s in kept],
        lat=[float(s['person_lat']) for s in kept],
        lon=[float(s['person_lon']) for s in kept],
        n=[s['cluster_size'] for s in kept],
        rms=[float(s['cluster_rms_m']) for s in kept],
        conf=[float(s['confidence']) for s in kept],
        mean_conf=[float(s['mean_conf']) for s in kept],
        dur=[float(s['duration_s']) for s in kept],
        alt=[float(s['cluster_alt_mean_m']) for s in kept],
    ))
    src_member = ColumnDataSource(dict(x=mx, y=my, lat=member['lat'],
                                       lon=member['lon'], conf=member['conf'],
                                       label=member['label']))
    src_reject = ColumnDataSource(dict(x=rx, y=ry, lat=reject['lat'],
                                       lon=reject['lon'], conf=reject['conf'],
                                       why=reject['why']))

    span = max(mx.max() - mx.min(), my.max() - my.min()) if len(mx) else 100.0
    pad  = span * 0.22
    p = figure(
        x_axis_type='mercator', y_axis_type='mercator',
        x_range=(mx.min() - pad, mx.max() + pad),
        y_range=(my.min() - pad, my.max() + pad),
        match_aspect=True,
        height=640, sizing_mode='stretch_width',
        tools='pan,wheel_zoom,box_zoom,reset,save',
        active_scroll='wheel_zoom',
        background_fill_color=SURFACE, border_fill_color=SURFACE,
        outline_line_color=None,
    )

    url, attribution = TILES[args.tiles]
    p.add_tile(WMTSTileSource(url=url, attribution=attribution), retina=True)
    p.grid.visible = False
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.axis.major_label_text_color = INK_MUTED
    p.axis.major_label_text_font_size = '10px'

    p.line(dx, dy, line_color=AQUA, line_width=2, line_alpha=0.85,
           legend_label='Drone path')

    r_member = p.scatter('x', 'y', source=src_member, size=4,
                         fill_color=BLUE, fill_alpha=0.28, line_color=None,
                         legend_label='Detections (kept)')

    r_reject = p.scatter('x', 'y', source=src_reject, size=11, marker='x',
                         line_color=RED, line_width=2.5, line_alpha=0.95,
                         legend_label='Rejected')

    # white halo lifts the marker off busy imagery (relief rule)
    r_person = p.scatter('x', 'y', source=src_person, size='size',
                         fill_color=BLUE, fill_alpha=0.92,
                         line_color=HALO, line_width=2.5,
                         legend_label='Person (cluster)')

    p.add_layout(LabelSet(x='x', y='y', text='label', source=src_person,
                          x_offset=0, y_offset=-6, text_align='center',
                          text_font_size='12px', text_font_style='bold',
                          text_color='#ffffff'))

    tip = 'font:12px system-ui,-apple-system,"Segoe UI",sans-serif;padding:2px 4px'
    p.add_tools(HoverTool(renderers=[r_person], attachment='above', tooltips=f"""
        <div style="{tip}">
          <div style="font-weight:600;font-size:13px;color:{INK}">Person @label</div>
          <div style="color:{INK_2};margin-top:3px">
            <div>@lat{{0.0000000}}, @lon{{0.0000000}}</div>
            <div>@n detections &middot; rms @rms{{0.00}} m</div>
            <div>best conf @conf{{0.000}} &middot; mean @mean_conf{{0.000}}</div>
            <div>seen for @dur{{0.0}} s &middot; drone alt @alt{{0.0}} m</div>
          </div>
        </div>"""))
    p.add_tools(HoverTool(renderers=[r_member], attachment='above', tooltips=f"""
        <div style="{tip}">
          <div style="font-weight:600;color:{INK}">Detection &rarr; @label</div>
          <div style="color:{INK_2};margin-top:3px">
            <div>@lat{{0.0000000}}, @lon{{0.0000000}}</div>
            <div>conf @conf{{0.000}}</div>
          </div>
        </div>"""))
    p.add_tools(HoverTool(renderers=[r_reject], attachment='above', tooltips=f"""
        <div style="{tip}">
          <div style="font-weight:600;color:{INK}">Rejected</div>
          <div style="color:{INK_2};margin-top:3px">
            <div>@why</div>
            <div>@lat{{0.0000000}}, @lon{{0.0000000}}</div>
            <div>conf @conf{{0.000}}</div>
          </div>
        </div>"""))

    p.legend.location = 'top_right'
    p.legend.click_policy = 'hide'
    p.legend.background_fill_color = SURFACE
    p.legend.background_fill_alpha = 0.92
    p.legend.border_line_color = '#e1e0d9'
    p.legend.label_text_color = INK_2
    p.legend.label_text_font_size = '11px'
    p.legend.spacing = 2

    num = lambda dp: NumberFormatter(format='0.' + '0' * dp if dp else '0')
    table = DataTable(
        source=src_person, height=min(60 + 28 * len(kept), 340),
        sizing_mode='stretch_width', index_position=None,
        columns=[
            TableColumn(field='label', title='Person', width=70),
            TableColumn(field='lat',   title='Latitude',  formatter=num(7)),
            TableColumn(field='lon',   title='Longitude', formatter=num(7)),
            TableColumn(field='n',     title='Detections'),
            TableColumn(field='rms',   title='RMS (m)',   formatter=num(2)),
            TableColumn(field='conf',  title='Best conf', formatter=num(3)),
            TableColumn(field='dur',   title='Duration (s)', formatter=num(1)),
        ],
    )

    header = Div(sizing_mode='stretch_width', text=f"""
      <div style="font:14px system-ui,-apple-system,'Segoe UI',sans-serif;color:{INK}">
        <h1 style="margin:0 0 2px;font-size:20px;font-weight:600;letter-spacing:-0.01em">
          Person geolocation &mdash; {len(kept)} people</h1>
        <p style="margin:0 0 14px;color:{INK_2};font-size:13px;line-height:1.5">
          {stats['kept_pts']:,} of {stats['total']:,} detections clustered &middot;
          stage&nbsp;1 DBSCAN eps&nbsp;{args.eps:g}&nbsp;m / min&nbsp;{args.min_samples},
          stage&nbsp;2 merge {args.merge_dist:g}&nbsp;m.
          Dropped <b>{stats['n_alt']:,}</b> below {args.min_alt:g}&nbsp;m AGL,
          <b>{stats['n_noise']:,}</b> DBSCAN noise, and
          <b>{stats['n_small']}</b> cluster(s) under {args.min_cluster_size} points.
        </p>
      </div>""")

    footer = Div(sizing_mode='stretch_width', text=f"""
      <div style="font:12px system-ui,-apple-system,'Segoe UI',sans-serif;
                  color:{INK_MUTED};margin-top:10px;line-height:1.5;
                  border-top:1px solid #e1e0d9;padding-top:10px">
        Marker area is proportional to detection count; position is the median of
        the cluster's member detections. Click a legend entry to hide that layer.
        Basemap: {'Esri World Imagery' if args.tiles == 'esri' else 'OpenStreetMap'}.
      </div>""")

    return column(header, p, table, footer,
                  sizing_mode='stretch_width',
                  styles={'max-width': '1200px', 'margin': '0 auto',
                          'padding': '22px', 'background': SURFACE})


# -- Main ---------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outs = os.path.join(os.path.dirname(here), 'outputs')

    ap = argparse.ArgumentParser()
    ap.add_argument('--input',      default=os.path.join(outs, 'detections.csv'))
    ap.add_argument('--output',     default=os.path.join(outs, 'cluster_map.html'))
    ap.add_argument('--csv_output', default=os.path.join(outs, 'clustered.csv'))
    ap.add_argument('--eps',              type=float, default=1.0,
                    help='Stage-1 DBSCAN radius in metres            [default: 1.0]')
    ap.add_argument('--min_samples',      type=int,   default=3,
                    help='Min detections per Stage-1 cluster          [default: 3]')
    ap.add_argument('--merge_dist',       type=float, default=5.0,
                    help='Stage-2 merge radius in metres              [default: 5.0]')
    ap.add_argument('--min_alt',          type=float, default=5.0,
                    help='Drop detections below this drone AGL (m)    [default: 5.0]')
    ap.add_argument('--min_cluster_size', type=int,   default=20,
                    help='Drop clusters with fewer than N detections  [default: 20]')
    ap.add_argument('--min_conf',         type=float, default=0.0,
                    help='Drop clusters whose best confidence < this  [default: 0.0]')
    ap.add_argument('--tiles', choices=sorted(TILES), default='esri',
                    help='Basemap: esri (satellite) or osm            [default: esri]')
    args = ap.parse_args()

    all_rows = load(args.input)
    n_total  = len(all_rows)
    print(f'Input         : {args.input}  ({n_total} detections)')

    # Altitude gate
    rows     = [r for r in all_rows if float(r['drone_altitude_agl']) >= args.min_alt]
    low_rows = [r for r in all_rows if float(r['drone_altitude_agl']) <  args.min_alt]
    print(f'Altitude gate : >= {args.min_alt}m AGL  ->  {len(low_rows)} degenerate '
          f'fix(es) dropped')

    lats = np.array([float(r['person_lat']) for r in rows])
    lons = np.array([float(r['person_lon']) for r in rows])
    cos_lat = np.cos(np.deg2rad(lats.mean()))

    # Stage 1
    labels = stage1(lats, lons, cos_lat, args.eps, args.min_samples)
    n1 = sum(1 for l in set(labels) if l != -1)
    nn = int((labels == -1).sum())
    print(f'Stage 1 DBSCAN: eps={args.eps}m  min_samples={args.min_samples}'
          f'  ->  {n1} clusters,  {nn} noise points dropped')

    # Stage 2
    labels = stage2(labels, lats, lons, cos_lat, args.merge_dist)
    n2 = sum(1 for l in set(labels) if l != -1)
    print(f'Stage 2 merge : merge_dist={args.merge_dist}m  ->  {n2} raw cluster(s)')

    # Summarise + filter
    all_summaries = [summarise(rows, labels == l, lats, lons, cos_lat, l + 1)
                     for l in range(n2)]

    kept, dropped = [], []
    for s in all_summaries:
        if (s['cluster_size'] >= args.min_cluster_size and
                float(s['confidence']) >= args.min_conf):
            kept.append(s)
        else:
            dropped.append(s)

    kept.sort(key=lambda s: s['cluster_size'], reverse=True)
    old_to_new = {s['person_idx'] - 1: i + 1 for i, s in enumerate(kept)}
    for i, s in enumerate(kept):
        s['person_idx'] = i + 1

    print(f'Filters       : min_cluster_size={args.min_cluster_size}  '
          f'min_conf={args.min_conf}')
    if dropped:
        print(f'  Dropped {len(dropped)} cluster(s):')
        for s in dropped:
            print(f'    n={s["cluster_size"]}  conf={s["confidence"]}'
                  f'  ({float(s["person_lat"]):.6f}, {float(s["person_lon"]):.6f})')
    print(f'  ->  {len(kept)} final person(s)\n')

    for s in kept:
        print(f"  Person {s['person_idx']:2d}:  "
              f"({float(s['person_lat']):.7f}, {float(s['person_lon']):.7f})  "
              f"n={s['cluster_size']:4d}  rms={s['cluster_rms_m']}m  "
              f"alt={s['cluster_alt_mean_m']}m  conf={s['confidence']}")

    # Split the raw detections into map layers
    member = {'lat': [], 'lon': [], 'conf': [], 'label': []}
    reject = {'lat': [], 'lon': [], 'conf': [], 'why': []}

    for r, l in zip(rows, labels):
        lat, lon = float(r['person_lat']), float(r['person_lon'])
        conf = float(r['confidence'])
        if l in old_to_new:
            member['lat'].append(lat); member['lon'].append(lon)
            member['conf'].append(conf)
            member['label'].append(f'P{old_to_new[l]}')
        else:
            reject['lat'].append(lat); reject['lon'].append(lon)
            reject['conf'].append(conf)
            reject['why'].append('DBSCAN noise' if l == -1
                                 else f'cluster < {args.min_cluster_size} points')
    for r in low_rows:
        reject['lat'].append(float(r['person_lat']))
        reject['lon'].append(float(r['person_lon']))
        reject['conf'].append(float(r['confidence']))
        reject['why'].append(f'drone below {args.min_alt:g} m AGL')

    drone = {'lat': [float(r['drone_lat']) for r in rows],
             'lon': [float(r['drone_lon']) for r in rows]}

    stats = {'total': n_total, 'kept_pts': len(member['lat']),
             'n_alt': len(low_rows), 'n_noise': nn, 'n_small': len(dropped)}

    write(args.csv_output, kept)
    print(f'\nSaved {len(kept)} person(s) -> {args.csv_output}')

    output_file(args.output, title='Person geolocation clusters', mode='inline')
    save(build_map(kept, member, reject, drone, args, stats))
    print(f'Map saved -> {args.output}')


if __name__ == '__main__':
    main()
