#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLEAN MONTHLY BUILD + strict month filtering + multi-tile mosaicking

- Inputs: GeoTIFF mosaics/tiles per band (B04, B08, B11, SCL) per month using glob templates.
- AOI in lon/lat (WGS84), --size WIDTH HEIGHT in pixels → unified grid for all months.
- SCL mask: drop shadow(3), water(6), cloud(8/9), cirrus(10), snow(11), and 0 (nodata).
- LFMC = 35 + 80*NDVI_n + 50*SM_n + 20*(ETf - 0.5)
- Monthly panel (2 columns × ceil(n/2) rows), with mean ± std overlay text.
- Province outlines overlay (optional) + per-province means via rasterization (optional).
- Strict month selection: picks files whose filename dates overlap the month window [d1..d2].
- If a month has multiple tiles (e.g., *_x0_y0.tif, *_x1_y0.tif, ...), they are mosaicked
  per band by reprojecting each tile to the target grid and averaging on overlaps.

Added:
- Save monthly LFMC GeoTIFF per month at out_dir/geotiff/lfmc_YYYY-MM.tif
python produce_lfmc_local.py \
  --input_type geotiff \
  --aoi 97 17 101.5 21 \
  --size 720 720 \
  --start 2024-01 --end 2024-04 \
  --b04_tpl "data/S2/B04_*.tif" \
  --b08_tpl "data/S2/B08_*.tif" \
  --b11_tpl "data/S2/B11_*.tif" \
  --scl_tpl "data/S2/SCL_*.tif" \
  --prov_shp ../run_lfmc_nrt/Province8P/Province8P.shp \
  --palette danger \
  --overall_summary_csv --per_province_csv \
  --out_dir outputs_v71

"""

import os
import re
import glob
import math
import argparse
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv

# ---------- formatters ----------
def lon_formatter(x, _): return f"{int(round(x))}°E" if x>=0 else f"{int(round(-x))}°W"
def lat_formatter(y, _): return f"{int(round(y))}°N" if y>=0 else f"{int(round(-x))}°S"

# ---------- classed "danger" palette ----------
DANGER_BOUNDS = [0, 80, 100, 120, 140, 1000]
DANGER_COLORS = ["#800000", "#FF4500", "#FFD700", "#9ACD32", "#006400"]
DANGER_LABELS = ["Extreme Dry (<80)", "Dry (80-100)", "Moderate Dry (100-120)", "Moderate Moist (120-140)", "Moist (>140)"]

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="LFMC monthly panel from local GeoTIFF mosaics/tiles")
    p.add_argument("--input_type", default="geotiff", choices=["geotiff"], help="Only geotiff supported")
    p.add_argument("--aoi", nargs=4, type=float, required=True, metavar=("xmin","ymin","xmax","ymax"))
    p.add_argument("--size", nargs=2, type=int, required=True, metavar=("WIDTH","HEIGHT"))
    p.add_argument("--start", required=True, help="YYYY-MM")
    p.add_argument("--end",   required=True, help="YYYY-MM")
    p.add_argument("--b04_tpl", required=True)
    p.add_argument("--b08_tpl", required=True)
    p.add_argument("--b11_tpl", required=True)
    p.add_argument("--scl_tpl", required=True)
    p.add_argument("--prov_shp", help="Province shapefile (WGS84 preferred)")
    # compat flags (no behavioral effect, kept for pipeline consistency)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gdal_threads", default="ALL_CPUS")
    p.add_argument("--day_cache_dir", default="/tmp/lfmc_cache")
    # styling / outputs
    p.add_argument("--palette", default="danger", choices=["danger","BrBG"])
    p.add_argument("--overall_summary_csv", action="store_true")
    p.add_argument("--per_province_csv", action="store_true")
    # reserved
    p.add_argument("--mask_shp")
    p.add_argument("--mask_filter")
    p.add_argument("--mask_mode")
    p.add_argument("--mask_raster")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()

# ---------- months ----------
def month_iter(start_ym, end_ym):
    s = datetime.strptime(start_ym, "%Y-%m")
    e = datetime.strptime(end_ym,   "%Y-%m")
    months = []
    cur = s
    while cur <= e:
        y, m = cur.year, cur.month
        mname = cur.strftime("%Y-%m")  # use YYYY-MM for filenames
        d1 = datetime(y, m, 1)
        d2 = (d1 + relativedelta(months=1)) - relativedelta(days=1)
        months.append((mname, d1, d2))
        cur = d1 + relativedelta(months=1)
    return months

# ---------- IO ----------
def open_as_float32(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    return arr.astype("float32", copy=False)

def open_uint(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    return arr

def build_target_grid(aoi, size_wh):
    xmin, ymin, xmax, ymax = aoi
    width, height = size_wh
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    return transform

def reproject_to(arr, src_path, dst_shape, dst_transform, resampling=Resampling.nearest):
    with rasterio.open(src_path) as src:
        src_transform = src.transform
        src_crs = src.crs
    dst = np.zeros(dst_shape, dtype=arr.dtype)
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs="EPSG:4326",
        resampling=resampling,
        num_threads=0
    )
    return dst

# ---------- math ----------
def normalize_01(x, lo, hi, invert=False):
    y = (x - lo) / (hi - lo + 1e-9)
    y = np.clip(y, 0, 1)
    return 1 - y if invert else y

def compute_lfmc(B04, B08, B11, scl_mask_valid):
    # reflectance scaling (if data are in 0..10000)
    if np.nanmax(B04) > 1.5 or np.nanmax(B08) > 1.5 or np.nanmax(B11) > 1.5:
        B04 = B04 / 10000.0
        B08 = B08 / 10000.0
        B11 = B11 / 10000.0
    M = scl_mask_valid.astype(bool)
    NDVI = np.where(M, (B08 - B04) / (B08 + B04 + 1e-6), np.nan)
    NDII = np.where(M, (B08 - B11) / (B08 + B11 + 1e-6), np.nan)
    MSI  = np.where(M, B11 / (B08 + 1e-6), np.nan)
    NDVI_n = normalize_01(NDVI, 0.2, 0.8)
    NDII_n = normalize_01(NDII, 0.0, 0.6)
    MSI_n  = normalize_01(MSI, 0.6, 1.4, invert=True)
    SM_n   = 0.5*NDII_n + 0.5*MSI_n
    ETf    = np.clip(1.25*NDVI_n, 0, 1)
    LFMC = 35 + 80*NDVI_n + 50*SM_n + 20*(ETf - 0.5)
    return np.clip(LFMC, 0, 200)

def scl_valid_mask(SCL):
    bad = (SCL==3)|(SCL==6)|(SCL==8)|(SCL==9)|(SCL==10)|(SCL==11)|(SCL==0)
    return ~bad

# ---------- plotting ----------
def figure_setup(nmonths, aoi):
    ncols = 2
    nrows = math.ceil(nmonths / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.0 + 2.2*(nrows-1)), constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)
    extent = [aoi[0], aoi[2], aoi[1], aoi[3]]
    return fig, axes, extent

def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

# ---------- strict month-aware file picking ----------
_DATE_PATTERNS = [
    re.compile(r'_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})'),  # range
    re.compile(r'_(\d{8})(?!\d)'),                               # daily YYYYMMDD
    re.compile(r'_(\d{4}-\d{2})(?!-\d)'),                        # monthly YYYY-MM
]

def parse_dates_from_name(path):
    """Return (start_dt, end_dt) if recognizable from filename, else (None, None)."""
    fname = os.path.basename(path)
    m = _DATE_PATTERNS[0].search(fname)
    if m:
        s = datetime.strptime(m.group(1), "%Y-%m-%d")
        e = datetime.strptime(m.group(2), "%Y-%m-%d")
        return s, e
    m = _DATE_PATTERNS[1].search(fname)
    if m:
        d = datetime.strptime(m.group(1), "%Y%m%d")
        return d, d
    m = _DATE_PATTERNS[2].search(fname)
    if m:
        s = datetime.strptime(m.group(1) + "-01", "%Y-%m-%d")
        e = (s + relativedelta(months=1)) - relativedelta(days=1)
        return s, e
    return None, None

def suffix_token(path):
    # remove the leading band prefix (e.g., "B04_") and return the rest
    base = os.path.basename(path)
    parts = base.split("_", 1)
    return parts[1] if len(parts) > 1 else base

def overlapping_files(pattern, d1, d2):
    files = sorted(glob.glob(pattern))
    out = []
    for p in files:
        s, e = parse_dates_from_name(p)
        if s is None:
            continue
        if e < d1 or s > d2:
            continue
        out.append(p)
    return out

def index_by_suffix(paths):
    return {suffix_token(p): p for p in paths}

def paired_tiles_for_month(b04_tpl, b08_tpl, b11_tpl, scl_tpl, d1, d2):
    """
    Return a list of (b04,b08,b11,scl) paths for tiles/files that overlap [d1,d2],
    pairing by filename suffix (everything after the first underscore).
    """
    b04s = overlapping_files(b04_tpl, d1, d2)
    b08_idx = index_by_suffix(overlapping_files(b08_tpl, d1, d2))
    b11_idx = index_by_suffix(overlapping_files(b11_tpl, d1, d2))
    scl_idx = index_by_suffix(overlapping_files(scl_tpl, d1, d2))

    pairs = []
    for b4 in b04s:
        suf = suffix_token(b4)
        b8  = b08_idx.get(suf)
        b11 = b11_idx.get(suf)
        scl = scl_idx.get(suf)
        if b8 and b11 and scl:
            pairs.append((b4, b8, b11, scl))
    return pairs

# ---------- GeoTIFF writer ----------
def save_monthly_geotiff(lfmc_arr, transform, out_dir, ym, nodata=np.nan):
    """Write monthly LFMC GeoTIFF as Float32, EPSG:4326."""
    tif_dir = os.path.join(out_dir, "geotiff")
    os.makedirs(tif_dir, exist_ok=True)
    out_path = os.path.join(tif_dir, f"lfmc_{ym}.tif")
    h, w = lfmc_arr.shape
    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=min(256, w),
        blockysize=min(256, h),
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(lfmc_arr.astype("float32"), 1)
    return out_path

# ---------- main ----------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    AOI = tuple(map(float, args.aoi))
    WIDTH, HEIGHT = map(int, args.size)
    transform = build_target_grid(AOI, (WIDTH, HEIGHT))
    target_shape = (HEIGHT, WIDTH)

    # Province outlines (optional)
    prov = None
    if args.prov_shp:
        prov_all = gpd.read_file(args.prov_shp)
        if prov_all.crs is None or prov_all.crs.to_epsg() != 4326:
            prov_all = prov_all.to_crs(4326)
        prov = prov_all.clip(box(*AOI))

    months = month_iter(args.start, args.end)
    fig, axes, extent = figure_setup(len(months), AOI)
    vmin, vmax = 50, 160
    last_im = None

    overall_rows = []  # month, mean, std, n_valid
    perprov_rows = []  # month, province, id, mean, std, n_valid

    for ax, (ym, d1, d2) in zip(axes.ravel(), months):
        # gather ALL overlapping tiles/files, paired by suffix
        pairs = paired_tiles_for_month(args.b04_tpl, args.b08_tpl, args.b11_tpl, args.scl_tpl, d1, d2)

        if not pairs:
            ax.set_title(f"{ym}\n(no data)")
            ax.axis("off")
            continue

        # mosaic LFMC by averaging overlapping tile contributions
        sum_arr = np.zeros(target_shape, dtype=np.float64)
        cnt_arr = np.zeros(target_shape, dtype=np.int32)

        for f_b04, f_b08, f_b11, f_scl in pairs:
            B04 = reproject_to(open_as_float32(f_b04), f_b04, target_shape, transform, resampling=Resampling.bilinear)
            B08 = reproject_to(open_as_float32(f_b08), f_b08, target_shape, transform, resampling=Resampling.bilinear)
            B11 = reproject_to(open_as_float32(f_b11), f_b11, target_shape, transform, resampling=Resampling.bilinear)
            SCL = reproject_to(open_uint(f_scl),       f_scl, target_shape, transform, resampling=Resampling.nearest)

            valid = scl_valid_mask(SCL)
            LFMC_tile = compute_lfmc(B04, B08, B11, valid)
            finite = np.isfinite(LFMC_tile) & valid
            if np.any(finite):
                sum_arr[finite] += LFMC_tile[finite]
                cnt_arr[finite] += 1

        with np.errstate(invalid='ignore', divide='ignore'):
            LFMC = sum_arr / np.where(cnt_arr == 0, np.nan, cnt_arr)

        # ---- NEW: save monthly GeoTIFF ----
        tif_path = save_monthly_geotiff(LFMC, transform, args.out_dir, ym)
        print(f"✓ Saved monthly GeoTIFF: {tif_path}")

        # draw
        if args.palette == "danger":
            import matplotlib.colors as mcolors
            cmap = mcolors.ListedColormap(DANGER_COLORS)
            norm = mcolors.BoundaryNorm(DANGER_BOUNDS, cmap.N)
            last_im = ax.imshow(LFMC, cmap=cmap, norm=norm,
                                extent=extent, origin="upper", interpolation="nearest")
        else:
            last_im = ax.imshow(LFMC, cmap="BrBG", vmin=vmin, vmax=vmax,
                                extent=extent, origin="upper", interpolation="nearest")

        if prov is not None and not prov.empty:
            prov.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.6)

        ax.set_title(ym)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")

        # stats (overall)
        finite = np.isfinite(LFMC)
        if np.any(finite):
            mean_v = float(np.nanmean(LFMC[finite]))
            std_v  = float(np.nanstd(LFMC[finite]))
            overall_rows.append([ym, f"{mean_v:.3f}", f"{std_v:.3f}", int(np.sum(finite))])
            ax.text(
                0.02, 0.98, f"{mean_v:0.1f} ± {std_v:0.1f} %",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.25")
            )
        else:
            overall_rows.append([ym, "nan", "nan", 0])
            ax.text(0.02, 0.98, "No valid data", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10, color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.25"))

        # per-province stats (optional)
        if args.prov_shp and prov is not None and not prov.empty:
            shapes = [(geom, idx+1) for idx, geom in enumerate(prov.geometry)]
            prov_id = rasterize(
                shapes=shapes,
                out_shape=target_shape,
                transform=transform,
                fill=0,
                dtype="int32",
                all_touched=False
            )
            for idx, row in prov.reset_index().iterrows():
                pid = idx+1
                mask = (prov_id == pid) & finite
                if np.any(mask):
                    m = float(np.nanmean(LFMC[mask]))
                    s = float(np.nanstd(LFMC[mask]))
                    c = int(np.sum(mask))
                else:
                    m, s, c = float("nan"), float("nan"), 0
                name = row.get("NAME_1") or row.get("name") or f"prov_{pid}"
                perprov_rows.append([ym, str(name), pid, f"{m:.3f}" if c>0 else "nan",
                                     f"{s:.3f}" if c>0 else "nan", c])

    # shared colorbar
    if args.palette == "danger":
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(DANGER_COLORS)
        norm = mcolors.BoundaryNorm(DANGER_BOUNDS, cmap.N)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_ticks([(DANGER_BOUNDS[i]+DANGER_BOUNDS[i+1])/2 for i in range(len(DANGER_COLORS))])
        cbar.set_ticklabels(DANGER_LABELS)
        cbar.set_label("LFMC class")
    else:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("LFMC (%)")

    fig.suptitle(f"LFMC monthly patterns ({months[0][0]} to {months[-1][0]})", fontsize=14)

    # save panel
    os.makedirs(args.out_dir, exist_ok=True)
    png_path = os.path.join(args.out_dir, "lfmc_monthly_panel_stats.png")
    pdf_path = os.path.join(args.out_dir, "lfmc_monthly_panel_stats.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"✓ Saved: {png_path}, {pdf_path}")

    # CSVs
    if args.overall_summary_csv:
        write_csv(os.path.join(args.out_dir, "overall_summary.csv"),
                  overall_rows, header=["month","mean","std","n_valid"])
    if args.per_province_csv and args.prov_shp:
        write_csv(os.path.join(args.out_dir, "per_province.csv"),
                  perprov_rows, header=["month","province","prov_id","mean","std","n_valid"])

if __name__ == "__main__":
    main()
