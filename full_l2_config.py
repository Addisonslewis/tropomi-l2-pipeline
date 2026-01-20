#!/usr/bin/env python3
"""
pipeline_download_compile_cleanup.py

Download Sentinel-5P L2 CH4 OFFL -> compile pixel-level CSV -> (optionally) delete .nc files.

Key improvements vs. the draft:
- NO hardcoded credentials (reads from env vars CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY)
- CLI arguments for date range, bbox, workdir, and delete behavior
- Repo-friendly relative defaults (workdir defaults to ./tmp_run)
- Safer logging and clearer output paths

Requires:
  pip install requests boto3 numpy pandas xarray netcdf4 h5netcdf

Auth (Copernicus Data Space Ecosystem S3):
  export CDSE_S3_ACCESS_KEY="..."
  export CDSE_S3_SECRET_KEY="..."

Examples:
  python3 full_l2_config.py \
    --start 2020-01-01 --end 2020-01-07 \
    --lat-min 24 --lat-max 50 --lon-min -125 --lon-max -66 \
    --workdir tmp_run \
    --delete-nc

Notes:
- This script downloads *.nc into {workdir}/nc, compiles a CSV into {workdir}/, then (optionally) deletes the nc files.
- Do NOT commit secrets. Do NOT commit large .nc / .csv outputs to GitHub (use .gitignore).
"""

import os
import sys
import glob
import time
import argparse
import datetime as dt
from typing import Dict, Generator, List, Optional, Tuple

import requests
import boto3
import numpy as np
import pandas as pd
import xarray as xr

# -------------------- CONSTANTS --------------------
COLLECTION = "sentinel-5p-l2-ch4-offl"
STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
ASSET_KEYWORD = "_CH4_"  # hard lock to CH4 assets

PAGE_LIMIT_DEFAULT = 100
MAX_ITEMS_DEFAULT = 100_000
MAX_RETRIES_DEFAULT = 3

# L2 variable candidates
CH4_CANDIDATES = [
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
    "CH4_column_volume_mixing_ratio_dry_air",
]

SESSION = requests.Session()
# ---------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Sentinel-5P L2 CH4 OFFL, compile pixel-level CSV, optionally delete .nc files."
    )

    # Required-ish: date range
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date inclusive (YYYY-MM-DD)")

    # BBox
    p.add_argument("--lon-min", type=float, default=-125.0)
    p.add_argument("--lat-min", type=float, default=24.0)
    p.add_argument("--lon-max", type=float, default=-67.0)
    p.add_argument("--lat-max", type=float, default=49.0)

    # Output/work paths
    p.add_argument(
        "--workdir",
        default="tmp_run",
        help="Working directory for downloads and outputs (default: ./tmp_run)",
    )
    p.add_argument(
        "--csv-out",
        default=None,
        help="Optional explicit CSV output path. Default: {workdir}/l2_{start}_{end}_ch4.csv",
    )

    # Download controls
    p.add_argument("--page-limit", type=int, default=PAGE_LIMIT_DEFAULT)
    p.add_argument("--max-items", type=int, default=MAX_ITEMS_DEFAULT)
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)

    # Behavior flags
    p.add_argument(
        "--delete-nc",
        action="store_true",
        help="Delete downloaded .nc files after compiling the CSV",
    )
    p.add_argument(
        "--keep-nc",
        action="store_true",
        help="Keep .nc files even if --delete-nc is set elsewhere (takes precedence).",
    )

    return p.parse_args()


def iso_range(start: str, end: str) -> str:
    d0 = dt.datetime.strptime(start, "%Y-%m-%d").date()
    d1 = dt.datetime.strptime(end, "%Y-%m-%d").date()
    if d1 < d0:
        raise ValueError("end date must be >= start date")
    start_iso = dt.datetime.combine(d0, dt.time(0, 0)).isoformat() + "Z"
    end_iso = dt.datetime.combine(d1 + dt.timedelta(days=1), dt.time(0, 0)).isoformat() + "Z"
    return f"{start_iso}/{end_iso}"


def get_s3_client():
    """
    CDSE S3 credentials should be supplied via environment variables:
      CDSE_S3_ACCESS_KEY, CDSE_S3_SECRET_KEY
    """
    access_key = (os.environ.get("CDSE_S3_ACCESS_KEY") or "").strip()
    secret_key = (os.environ.get("CDSE_S3_SECRET_KEY") or "").strip()

    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing CDSE S3 credentials. Set env vars CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY."
        )

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
    )


def stac_query(datetime_interval: str, bbox: List[float], page_limit: int) -> Generator[Dict, None, None]:
    headers = {"Accept": "application/geo+json, application/json"}
    payload = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": datetime_interval,
        "limit": min(max(page_limit, 1), 1000),
    }

    resp = SESSION.post(STAC_SEARCH, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    while True:
        for feat in data.get("features", []):
            yield feat

        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next" and "href" in link:
                next_url = link["href"]
                break
        if not next_url:
            return

        try:
            resp = SESSION.get(next_url, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            print(f"[WARN] STAC pagination failed ({e}); stopping further pages.", file=sys.stderr)
            return


def choose_asset_href(assets: Dict, substring: str) -> Optional[str]:
    for _, asset in assets.items():
        href = asset.get("href")
        if href and substring.lower() in href.lower():
            return href
    return None


def download_s3(url: str, out_path: str, s3, max_retries: int = 3) -> None:
    if not url.startswith("s3://eodata/"):
        raise ValueError(f"Unsupported URL (expected s3://eodata/...): {url}")

    bucket = "eodata"
    key = url[len("s3://eodata/") :]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".part"

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[S3] Downloading s3://{bucket}/{key} -> {out_path}")
            s3.download_file(bucket, key, tmp_path)
            os.replace(tmp_path, out_path)
            return
        except Exception as e:
            print(f"[S3] Attempt {attempt} failed: {e}", file=sys.stderr)
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)


def open_l2(path: str):
    # Prefer PRODUCT group if present
    try:
        ds = xr.open_dataset(path, group="PRODUCT")
        if len(ds.data_vars) > 0:
            return ds
        ds.close()
    except Exception:
        pass
    return xr.open_dataset(path)


def find_ch4_var(ds) -> Optional[str]:
    for n in CH4_CANDIDATES:
        if n in ds.data_vars:
            return n
    return None


def first_existing(ds, names: List[str]) -> Optional[str]:
    for n in names:
        if n in ds.data_vars or n in ds.coords:
            return n
    return None


def step_download(
    start_date: str,
    end_date: str,
    bbox: List[float],
    nc_dir: str,
    page_limit: int,
    max_items: int,
    max_retries: int,
) -> None:
    os.makedirs(nc_dir, exist_ok=True)
    dt_interval = iso_range(start_date, end_date)
    print(f"[STAC] Searching {dt_interval} bbox={bbox}")
    print(f"[INFO] Download dir: {nc_dir}")

    s3 = get_s3_client()
    total = 0
    downloaded = 0

    for feat in stac_query(dt_interval, bbox, page_limit):
        total += 1
        if total > max_items:
            print(f"[INFO] Reached max-items={max_items}; stopping.")
            break

        item_id = feat.get("id", "")
        # tight filter to CH4 OFFL L2
        if not (item_id.startswith("S5P_") and "_L2__CH4___" in item_id):
            continue

        assets = feat.get("assets", {})
        href = choose_asset_href(assets, ASSET_KEYWORD)
        if not href:
            continue

        leaf = os.path.basename(href.split("?")[0]) or f"{item_id}.nc"
        out_path = os.path.join(nc_dir, leaf)
        if os.path.exists(out_path):
            continue

        try:
            download_s3(href, out_path, s3, max_retries=max_retries)
            downloaded += 1
        except Exception as e:
            print(f"[ERROR] Download failed for {leaf}: {e}", file=sys.stderr)

    print(f"[DONE] Download step complete. Downloaded {downloaded} files.")


def step_compile_csv(
    nc_dir: str,
    csv_out: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> None:
    files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    if not files:
        raise RuntimeError(f"No .nc files found in {nc_dir}")

    print(f"[INFO] Found {len(files)} .nc files")
    print(f"[INFO] Pixel bbox clip: lon[{min_lon},{max_lon}] lat[{min_lat},{max_lat}]")

    rows = []

    for i, path in enumerate(files, start=1):
        print(f"[INFO] ({i}/{len(files)}) {os.path.basename(path)}")

        try:
            ds = open_l2(path)
        except Exception as e:
            print(f"[WARN] Could not open {path}: {e}", file=sys.stderr)
            continue

        ch4_name = find_ch4_var(ds)
        if ch4_name is None or "qa_value" not in ds:
            ds.close()
            continue

        lat_name = first_existing(ds, ["latitude", "lat"])
        lon_name = first_existing(ds, ["longitude", "lon"])
        if lat_name is None or lon_name is None:
            ds.close()
            continue

        has_time = ("time_utc" in ds.data_vars) or ("time_utc" in ds.coords)
        var_list = [ch4_name, "qa_value"]
        if has_time:
            var_list.append("time_utc")

        try:
            sub = ds[var_list]
            df_file = sub.to_dataframe().reset_index()
        except Exception as e:
            print(f"[WARN] DataFrame conversion failed: {e}", file=sys.stderr)
            ds.close()
            continue

        if lat_name not in df_file.columns or lon_name not in df_file.columns:
            ds.close()
            continue

        # Keep only finite CH4/qa/lat/lon
        mask = (
            np.isfinite(df_file[ch4_name])
            & np.isfinite(df_file["qa_value"])
            & np.isfinite(df_file[lat_name])
            & np.isfinite(df_file[lon_name])
        )
        df_file = df_file.loc[mask].copy()
        if df_file.empty:
            ds.close()
            continue

        df_file.rename(
            columns={
                ch4_name: "ch4",
                lat_name: "latitude",
                lon_name: "longitude",
            },
            inplace=True,
        )

        if has_time and "time_utc" in df_file.columns:
            df_file.rename(columns={"time_utc": "utc_time"}, inplace=True)
        else:
            df_file["utc_time"] = np.nan

        before = len(df_file)
        df_file = df_file[
            (df_file["latitude"] >= min_lat)
            & (df_file["latitude"] <= max_lat)
            & (df_file["longitude"] >= min_lon)
            & (df_file["longitude"] <= max_lon)
        ].copy()
        after = len(df_file)
        if after == 0:
            ds.close()
            continue

        print(f"       bbox kept {after:,}/{before:,}")

        df_file["qa_label"] = np.where(df_file["qa_value"] >= 0.5, "good", "low")
        df_file["file"] = os.path.basename(path)

        cols = ["file", "latitude", "longitude", "ch4", "qa_value", "qa_label", "utc_time"]
        # keep a few optional indices if present
        for extra in ("scanline", "ground_pixel", "time"):
            if extra in df_file.columns and extra not in cols:
                cols.append(extra)
        df_file = df_file[cols]

        rows.append(df_file)
        ds.close()

    if not rows:
        raise RuntimeError("No data extracted (after filters).")

    df = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[DONE] Wrote {len(df):,} rows to {csv_out}")


def step_cleanup_nc(nc_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    print(f"[INFO] Deleting {len(files)} .nc files from {nc_dir}")
    deleted = 0
    for f in files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            print(f"[WARN] Failed to delete {f}: {e}", file=sys.stderr)
    print(f"[DONE] Deleted {deleted}/{len(files)} .nc files")


def main():
    args = parse_args()

    # Workdir and derived paths
    workdir = os.path.abspath(args.workdir)
    nc_dir = os.path.join(workdir, "nc")

    csv_out = args.csv_out
    if not csv_out:
        csv_out = os.path.join(workdir, f"l2_{args.start}_{args.end}_ch4.csv")
    else:
        csv_out = os.path.abspath(csv_out)

    # bbox
    min_lon, min_lat, max_lon, max_lat = args.lon_min, args.lat_min, args.lon_max, args.lat_max
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # safety: keep-nc overrides delete-nc
    delete_nc_after = bool(args.delete_nc) and not bool(args.keep_nc)

    os.makedirs(workdir, exist_ok=True)

    print("[INFO] --------------------")
    print(f"[INFO] start={args.start} end={args.end}")
    print(f"[INFO] bbox={bbox}")
    print(f"[INFO] workdir={workdir}")
    print(f"[INFO] nc_dir={nc_dir}")
    print(f"[INFO] csv_out={csv_out}")
    print(f"[INFO] delete_nc_after={delete_nc_after}")
    print("[INFO] --------------------\n")

    step_download(
        start_date=args.start,
        end_date=args.end,
        bbox=bbox,
        nc_dir=nc_dir,
        page_limit=args.page_limit,
        max_items=args.max_items,
        max_retries=args.max_retries,
    )

    step_compile_csv(
        nc_dir=nc_dir,
        csv_out=csv_out,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )

    if delete_nc_after:
        step_cleanup_nc(nc_dir)

    print("\n[ALL DONE]")
    print(f"CSV: {csv_out}")
    print(f"(Temp nc folder: {nc_dir})")


if __name__ == "__main__":
    main()