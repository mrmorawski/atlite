# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Explore the NCAR THREDDS/OPeNDAP ERA5 mirror and resolve open questions for
the era5-ncar atlite module.

Phase B  (default):  metadata exploration, coordinate verification, catalog listing.
Phase C  (--phase-c): feature-by-feature processing and comparison with cached CDS cutout.

Usage:
    python scripts/dataset_comparison.py           # Phase B only
    python scripts/dataset_comparison.py --phase-c # Phase C only
"""

import calendar
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

OPENDAP_BASE = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/"
CATALOG_BASE = "https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/"

# Test bounding box (BOUNDS = west, south, east, north)
X0, Y0, X1, Y1 = -4, 56, 1.5, 62  # west, south, east, north

# Known filenames from S3 inspection
AN_SFC_U10_JAN2013 = (
    "e5.oper.an.sfc/201301/e5.oper.an.sfc.128_165_10u.ll025sc.2013010100_2013013123.nc"
)
# Forecast accumulated files are split into ~15-day half-month chunks.
FC_SSRD_JAN2013_H1 = (
    "e5.oper.fc.sfc.accumu/201301/"
    "e5.oper.fc.sfc.accumu.128_169_ssrd.ll025sc.2013010106_2013011606.nc"
)

SECTION_WIDTH = 72


def section(title):
    print(f"\n{'=' * SECTION_WIDTH}")
    print(f"  {title}")
    print(f"{'=' * SECTION_WIDTH}")


def open_opendap(path):
    """Open a dataset from NCAR THREDDS via OPeNDAP."""
    url = OPENDAP_BASE + path
    print(f"  URL: {url}")
    # netCDF4 in the venv is not compiled with OPeNDAP support; use pydap instead.
    # Suppress the DAP2 legacy warning — DAP2 is fine for this server.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xr.open_dataset(url, engine="pydap")


def show_coords(ds):
    print(f"\n  Dimensions : {dict(ds.sizes)}")
    print(f"  Coordinates: {list(ds.coords)}")
    print(f"  Data vars  : {list(ds.data_vars)}")
    for name, coord in ds.coords.items():
        if coord.ndim == 0:
            print(f"    {name}: scalar = {coord.values}")
        else:
            step = float(coord[1] - coord[0]) if len(coord) > 1 else float("nan")
            print(
                f"    {name}: {float(coord.min()):.4f} .. {float(coord.max()):.4f}"
                f"  step={step:.4f}  n={len(coord)}"
            )


# ---------------------------------------------------------------------------
# Shared spatial subsetting helper (used by Phase B and Phase C)
# ---------------------------------------------------------------------------
def sel_bbox(ds, lat_coord, lon_coord, x0, y0, x1, y1):
    """
    Subset ds to a bounding box, handling 0..360 longitude and the case where
    the bbox crosses the 0-meridian (e.g. x0=-4, x1=1.5).

    IMPORTANT: never call ds.roll() on a lazy pydap dataset — it forces a full
    global download and will hit the THREDDS 500 MB limit (HTTP 403).
    Instead, convert the bbox to 0..360 and do two .sel() calls if needed.
    """
    lat = ds.coords[lat_coord].values
    lat_desc = lat[0] > lat[-1]

    # Latitude slice (must match N→S vs S→N storage order)
    lat_slice = slice(y1, y0) if lat_desc else slice(y0, y1)

    # Convert bbox lons to 0..360
    x0_360 = x0 % 360
    x1_360 = x1 % 360

    if x0_360 <= x1_360:
        # Bbox does not wrap — single .sel() call
        lon_slice = slice(x0_360, x1_360)
        return ds.sel(**{lat_coord: lat_slice, lon_coord: lon_slice})
    else:
        # Bbox wraps across 0° meridian (e.g. 356..360 ∪ 0..1.5).
        # x0_360 is the high-valued segment (e.g. 356..360) — subtract 360 to get
        # negative coords (-4..0). x1_360 is the low-valued segment (0..1.5) — keep as-is.
        high_seg = ds.sel(**{lat_coord: lat_slice, lon_coord: slice(x0_360, 360)})
        low_seg = ds.sel(**{lat_coord: lat_slice, lon_coord: slice(0, x1_360)})
        new_lon = high_seg.coords[lon_coord].values.copy() - 360
        high_seg = high_seg.assign_coords({lon_coord: new_lon})
        # Only concat variables that have the longitude dimension.
        # Variables like utc_date (time-only) don't have it and can't be concatenated.
        vars_with_lon = [v for v in ds.data_vars if lon_coord in ds[v].dims]
        return (
            xr.concat(
                [high_seg[vars_with_lon], low_seg[vars_with_lon]],
                dim=lon_coord,
                data_vars="all",
            )
            .sortby(lon_coord)
        )


# ===========================================================================
# PHASE B: Metadata exploration and open questions
# ===========================================================================

# ---------------------------------------------------------------------------
# Section 1: Analysis surface file — coordinate names, variable names, lat order
# ---------------------------------------------------------------------------
def inspect_analysis_surface():
    section("1. Analysis surface file: u10 (Jan 2013)")
    ds = open_opendap(AN_SFC_U10_JAN2013)
    show_coords(ds)

    # Use known coordinate names directly — NCAR files use 'latitude'/'longitude'
    lat_coord = "latitude" if "latitude" in ds.coords else None
    lon_coord = "longitude" if "longitude" in ds.coords else None

    if lat_coord:
        lat = ds.coords[lat_coord].values
        direction = "N→S (descending)" if lat[0] > lat[-1] else "S→N (ascending)"
        print(f"\n  Latitude coordinate '{lat_coord}': {direction}")
        print(f"    first={lat[0]:.2f}, last={lat[-1]:.2f}, step={lat[1] - lat[0]:.4f}")

    if lon_coord:
        lon = ds.coords[lon_coord].values
        print(f"\n  Longitude coordinate '{lon_coord}':")
        print(f"    first={lon[0]:.2f}, last={lon[-1]:.2f}, step={lon[1] - lon[0]:.4f}")
        print(f"    Convention: {'0..360' if lon.max() > 180 else '-180..180'}")
        print(f"    NOTE: bbox uses negative lons ({X0}..{X1}), needs coordinate roll!")

    print("\n  Variable details:")
    for vname in ds.data_vars:
        v = ds[vname]
        print(
            f"    {vname!r}: dims={v.dims}, dtype={v.dtype}, units={v.attrs.get('units', '?')}"
        )

    ds.close()
    return lat_coord, lon_coord


# ---------------------------------------------------------------------------
# Section 2: Spatial subset — verify .sel() with bbox in 0..360 longitude space
# ---------------------------------------------------------------------------
def inspect_spatial_subset(lat_coord, lon_coord):
    section("2. Spatial subset for test bbox (x=-4..1.5, y=56..62)")

    ds = open_opendap(AN_SFC_U10_JAN2013)

    lat = ds.coords[lat_coord].values
    lat_desc = lat[0] > lat[-1]
    lon = ds.coords[lon_coord].values
    lon_0_360 = lon.max() > 180

    print(
        f"\n  Latitude direction : {'N→S (descending)' if lat_desc else 'S→N (ascending)'}"
    )
    print(f"  Longitude convention: {'0..360' if lon_0_360 else '-180..180'}")

    x0_360, x1_360 = X0 % 360, X1 % 360
    wraps = x0_360 > x1_360
    print(
        f"  Bbox in 0..360 space: {x0_360}..360 ∪ 0..{x1_360}"
        if wraps
        else f"  Bbox in 0..360 space: {x0_360}..{x1_360}"
    )
    print(
        f"  Crosses 0-meridian: {'YES — using two .sel() + concat' if wraps else 'NO — single .sel()'}"
    )

    subset = sel_bbox(ds, lat_coord, lon_coord, X0, Y0, X1, Y1)
    show_coords(subset)

    # Size estimate — use main gridded variable (skip 1-D utc_date etc.)
    main_vars = [v for v in subset.data_vars if subset[v].ndim >= 3]
    if main_vars:
        arr = subset[main_vars[0]]
        size_mb = arr.size * 4 / (1024 * 1024)
        print(f"\n  Main variable '{main_vars[0]}' shape : {arr.shape}")
        print(f"  Estimated size                    : {size_mb:.1f} MB  (float32)")
        print(
            f"  Under 500 MB THREDDS limit        : {'YES' if size_mb < 400 else 'WARNING: close!'}"
        )

    ds.close()
    return lat_desc, lon_0_360


# ---------------------------------------------------------------------------
# Section 3: Forecast-accumulated file — structure, time dims, accumulation
# ---------------------------------------------------------------------------
def inspect_forecast_accumulated():
    section("3. Forecast-accumulated file: ssrd (Jan 2013, first half)")
    ds = open_opendap(FC_SSRD_JAN2013_H1)
    show_coords(ds)

    print("\n  Variable details:")
    for vname in ds.data_vars:
        v = ds[vname]
        print(
            f"    {vname!r}: dims={v.dims}, dtype={v.dtype}, units={v.attrs.get('units', '?')}"
        )
        for attr in ("long_name", "standard_name", "cell_methods"):
            if attr in v.attrs:
                print(f"      {attr}: {v.attrs[attr]}")

    # Examine both time-related coordinates
    for cname in ds.coords:
        if "time" in cname.lower() or "hour" in cname.lower():
            c = ds.coords[cname]
            print(f"\n  Coord '{cname}' (n={len(c)}):")
            print(f"    first 4: {c.values[:4]}")
            print(f"    last  4: {c.values[-4:]}")

    # Explain the (forecast_initial_time, forecast_hour) structure
    if "forecast_initial_time" in ds.coords and "forecast_hour" in ds.coords:
        n_init = ds.sizes["forecast_initial_time"]
        n_hour = ds.sizes["forecast_hour"]
        print(
            f"\n  Structure: {n_init} init times × {n_hour} forecast hours = {n_init * n_hour} total hourly values"
        )
        print(
            f"  To get real timestamps: actual_time = forecast_initial_time + forecast_hour * 1h"
        )
        print(
            f"  Accumulation is CUMULATIVE from init time — must diff along forecast_hour axis"
        )
        print(f"  Then divide by 3600 to convert J/m² → W/m²")

    ds.close()


# ---------------------------------------------------------------------------
# Section 4: THREDDS catalog listing — discover all parameter codes
# ---------------------------------------------------------------------------
def list_catalog(subdir, yyyymm=None):
    """Fetch THREDDS catalog XML and extract dataset filenames."""
    if yyyymm:
        url = f"{CATALOG_BASE}{subdir}/{yyyymm}/catalog.xml"
    else:
        url = f"{CATALOG_BASE}{subdir}/catalog.xml"
    print(f"  Catalog URL: {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching catalog: {e}")
        return [], ""

    # Parse dataset name attributes from catalog XML
    names = re.findall(r'name="([^"]+\.nc)"', resp.text)
    return names, resp.text


def inspect_param_codes():
    section("4. THREDDS catalog — discover parameter codes")

    print("\n--- e5.oper.an.sfc / 201301 ---")
    an_files, _ = list_catalog("e5.oper.an.sfc", "201301")
    for f in sorted(an_files):
        print(f"  {f}")

    print("\n--- e5.oper.fc.sfc.accumu / 201301 ---")
    fc_files, _ = list_catalog("e5.oper.fc.sfc.accumu", "201301")
    for f in sorted(fc_files):
        print(f"  {f}")

    # Invariant — catalog may list subdirectories rather than files
    print("\n--- e5.oper.invariant (top-level catalog) ---")
    inv_files, inv_xml = list_catalog("e5.oper.invariant")
    if inv_files:
        for f in sorted(inv_files):
            print(f"  {f}")
    else:
        # Catalog may list catalogRefs (subdirectories) instead of direct files
        refs = re.findall(r'href="([^"]+)"', inv_xml)
        hrefs = [r for r in refs if "catalog" in r.lower()]
        print(f"  No .nc files at top level. CatalogRef hrefs found: {hrefs[:10]}")
        # Also print any datasetScan or dataset name entries
        names_all = re.findall(r'name="([^"]*)"', inv_xml)
        print(f"  All name= entries: {names_all[:20]}")

    return an_files, fc_files, inv_files


# ---------------------------------------------------------------------------
# Section 5: Invariant file — discover correct path then open
# ---------------------------------------------------------------------------
def find_and_inspect_invariant(fc_files):
    section("5. Invariant file: geopotential z — discover correct path")

    candidate_paths = [
        # Path known from S3 — try as-is on THREDDS
        "e5.oper.invariant/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc",
        # Some NCAR mirrors put invariant under a year directory
        "e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc",
    ]

    # Also try fetching the raw catalog XML for invariant and print it
    print("\n  Raw invariant catalog XML (first 2000 chars):")
    try:
        resp = requests.get(f"{CATALOG_BASE}e5.oper.invariant/catalog.xml", timeout=30)
        print(resp.text[:2000])
    except Exception as e:
        print(f"  ERROR: {e}")

    # Try each candidate path
    for path in candidate_paths:
        print(f"\n  Trying: {path}")
        try:
            ds = open_opendap(path)
            show_coords(ds)
            print("\n  Variable details:")
            for vname in ds.data_vars:
                v = ds[vname]
                print(
                    f"    {vname!r}: dims={v.dims}, dtype={v.dtype}, units={v.attrs.get('units', '?')}"
                )
            ds.close()
            print(f"\n  SUCCESS: invariant file found at: {path}")
            return path
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n  Could not find invariant file — may need manual catalog exploration.")
    return None


# ---------------------------------------------------------------------------
# Section 6: Forecast second half — confirm split pattern and time continuity
# ---------------------------------------------------------------------------
def inspect_forecast_second_half(fc_files):
    section("6. Forecast-accumulated second half of Jan 2013")

    ssrd_files = sorted(f for f in fc_files if "169_ssrd" in f)
    print(f"\n  All ssrd files for Jan 2013: {ssrd_files}")

    if len(ssrd_files) >= 2:
        second = ssrd_files[1]
        url_path = f"e5.oper.fc.sfc.accumu/201301/{second}"
        print(f"\n  Opening second half: {second}")
        ds = open_opendap(url_path)
        print(f"  Dimensions: {dict(ds.sizes)}")
        for cname in ds.coords:
            if "time" in cname.lower() or "hour" in cname.lower():
                c = ds.coords[cname]
                print(f"  {cname}: {c.values[0]} .. {c.values[-1]}  (n={len(c)})")
        ds.close()

        print(f"\n  Half-month split pattern:")
        print(f"    First  half: {{YYYY}}{{MM}}0106_{{YYYY}}{{MM}}1606.nc")
        m = re.search(r"(\d{8})_(\d{8})\.nc$", second)
        if m:
            print(f"    Second half ends: {second}")
    else:
        print("  Only one ssrd file found.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(lat_coord, lon_coord, lat_desc, lon_0_360, inv_path):
    section("SUMMARY OF FINDINGS")

    lat_sel = (
        "slice(north, south)  [e.g. slice(62, 56)]"
        if lat_desc
        else "slice(south, north)"
    )
    lon_note = (
        "0..360 — two .sel() calls + concat when bbox crosses 0-meridian; reassign high coords - 360"
        if lon_0_360
        else "-180..180 — .sel() works directly"
    )

    print(f"""
  Coordinate names
  ----------------
  Latitude  : '{lat_coord}'   direction: {"N→S (descending)" if lat_desc else "S→N (ascending)"}
  Longitude : '{lon_coord}'  convention: {lon_note}
  Lat .sel() slice : {lat_sel}

  Variable naming
  ---------------
  Analysis surface vars: uppercase with VAR_ prefix  (e.g. VAR_10U, VAR_10V)
  Forecast accum vars  : uppercase, no prefix         (e.g. SSRD, SSR, FDIR)
  Invariant            : uppercase                    (e.g. Z)

  Forecast accumulated structure
  --------------------------------
  Dims: (forecast_initial_time, forecast_hour, latitude, longitude)
  forecast_initial_time: twice-daily (06:00, 18:00 UTC), ~15 per half-month file
  forecast_hour        : 1..12 (hourly accumulations from each init time)
  Accumulation         : CUMULATIVE from init time (J/m²) — must diff along hour axis
  To reconstruct hourly: diff(concat([0, hours 1..12], dim=forecast_hour), dim=forecast_hour)
  Then convert to W/m²: divide by 3600

  Filename patterns
  -----------------
  Analysis surface : e5.oper.an.sfc/{{YYYYMM}}/e5.oper.an.sfc.{{param}}.ll025sc.{{YYYYMM}}0100_{{YYYYMM}}<last_day>23.nc
  Forecast accumu  : split into two files per month —
    first  half: e5.oper.fc.sfc.accumu/{{YYYYMM}}/...{{YYYYMM}}0106_{{YYYYMM}}1606.nc
    second half: e5.oper.fc.sfc.accumu/{{YYYYMM}}/...{{YYYYMM}}1606_{{YYYY}}{{MM+1}}0106.nc
  Invariant        : {inv_path or "NOT FOUND — needs investigation"}

  Verified parameter codes (from catalog)
  -----------------------------------------
  u10   : 128_165_10u   (an.sfc)
  v10   : 128_166_10v   (an.sfc)
  u100  : 228_246_100u  (an.sfc)
  v100  : 228_247_100v  (an.sfc)
  fsr   : 128_244_fsr   (an.sfc)   [forecast surface roughness — confirmed]
  t2m   : 128_167_2t    (an.sfc)
  d2m   : 128_168_2d    (an.sfc)
  stl4  : 128_236_stl4  (an.sfc)
  ssrd  : 128_169_ssrd  (fc.sfc.accumu)
  ssr   : 128_176_ssr   (fc.sfc.accumu)
  fdir  : 228_021_fdir  (fc.sfc.accumu)  [was wrong in plan: 128_228_fdir → 228_021_fdir]
  tisr  : 128_212_tisr  (fc.sfc.accumu)  [confirmed]
  ro    : 128_205_ro    (fc.sfc.accumu)  [confirmed]
  tp    : NOT IN CATALOG — not present in fc.sfc.accumu (not needed: era5.py uses ro, not tp)
  z     : 128_129_z     (invariant)      [under 197901/ subdir — see section 5]
""")


def run_phase_b():
    print("ERA5-NCAR OPeNDAP exploration script — Phase B")
    print(f"Test bbox: x={X0}..{X1}, y={Y0}..{Y1}  (BOUNDS = {(X0, Y0, X1, Y1)})")

    lat_coord, lon_coord = inspect_analysis_surface()
    lat_desc, lon_0_360 = inspect_spatial_subset(lat_coord, lon_coord)
    inspect_forecast_accumulated()
    an_files, fc_files, inv_files = inspect_param_codes()
    inv_path = find_and_inspect_invariant(fc_files)
    inspect_forecast_second_half(fc_files)
    print_summary(lat_coord, lon_coord, lat_desc, lon_0_360, inv_path)


# ===========================================================================
# PHASE C: Feature-by-feature processing and comparison with CDS reference
# ===========================================================================

CACHE_PATH = Path("test-cache/cutout_era5.nc")

# Maps ERA5 short name → (product_dir, param_code, ncar_varname_in_file)
#
# Variable naming conventions discovered in Phase B:
#   Analysis surface : VAR_ prefix + uppercase ECMWF short name  (e.g. VAR_10U)
#   Forecast accumu  : uppercase short name, no prefix            (e.g. SSRD)
#   Invariant        : uppercase short name, no prefix            (e.g. Z)
VAR_INFO = {
    "u10":  ("e5.oper.an.sfc",        "128_165_10u",  "VAR_10U"),
    "v10":  ("e5.oper.an.sfc",        "128_166_10v",  "VAR_10V"),
    "u100": ("e5.oper.an.sfc",        "228_246_100u", "VAR_100U"),
    "v100": ("e5.oper.an.sfc",        "228_247_100v", "VAR_100V"),
    "fsr":  ("e5.oper.an.sfc",        "128_244_fsr",  "FSR"),
    "t2m":  ("e5.oper.an.sfc",        "128_167_2t",   "VAR_2T"),  # may need → 2T
    "d2m":  ("e5.oper.an.sfc",        "128_168_2d",   "VAR_2D"),  # may need → 2D
    "stl4": ("e5.oper.an.sfc",        "128_236_stl4", "STL4"),
    "ssrd": ("e5.oper.fc.sfc.accumu", "128_169_ssrd", "SSRD"),
    "ssr":  ("e5.oper.fc.sfc.accumu", "128_176_ssr",  "SSR"),
    "fdir": ("e5.oper.fc.sfc.accumu", "228_021_fdir", "FDIR"),
    "tisr": ("e5.oper.fc.sfc.accumu", "128_212_tisr", "TISR"),
    "ro":   ("e5.oper.fc.sfc.accumu", "128_205_ro",   "RO"),
    "z":    ("e5.oper.invariant",     "128_129_z",    "Z"),
}

INVARIANT_PATH = (
    "e5.oper.invariant/197901/"
    "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
)


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def _an_sfc_url(product_dir, param_code, year, month):
    """Full-month analysis surface file path (relative to OPENDAP_BASE)."""
    ym = f"{year}{month:02d}"
    last = calendar.monthrange(year, month)[1]
    fname = f"{product_dir}.{param_code}.ll025sc.{ym}0100_{ym}{last:02d}23.nc"
    return f"{product_dir}/{ym}/{fname}"


def _fc_half_urls(product_dir, param_code, year, month):
    """Both half-month forecast file paths for a given year/month."""
    ym = f"{year}{month:02d}"
    next_month = month % 12 + 1
    next_year = year + (1 if month == 12 else 0)
    next_ym = f"{next_year}{next_month:02d}"
    prefix = f"{product_dir}/{ym}/{product_dir}.{param_code}.ll025sc"
    return [
        f"{prefix}.{ym}0106_{ym}1606.nc",
        f"{prefix}.{ym}1606_{next_ym}0106.nc",
    ]


def _fc_prev_half_url(product_dir, param_code, year, month):
    """Second half of the previous month — needed for hours 01:00-06:00 of day 1.

    ERA5 forecasts are initialised at 06:00 and 18:00 UTC.  Hours 01-06 of
    the first day of month M come from the 18:00 init of the last day of M-1,
    which lives in the second-half file of month M-1.
    """
    prev_year = year - 1 if month == 1 else year
    prev_month = 12 if month == 1 else month - 1
    prev_ym = f"{prev_year}{prev_month:02d}"
    ym = f"{year}{month:02d}"
    prefix = f"{product_dir}/{prev_ym}/{product_dir}.{param_code}.ll025sc"
    return f"{prefix}.{prev_ym}1606_{ym}0106.nc"


# ---------------------------------------------------------------------------
# Forecast accumulation → hourly conversion
# ---------------------------------------------------------------------------

def _fc_to_hourly(subset, var_name):
    """Convert forecast cumulative-accumulated variable to hourly DataArray.

    Input subset has dims (forecast_initial_time, forecast_hour, latitude, longitude).
    forecast_hour values are 1..12; each is a running cumulative sum from the
    forecast init time.

    Steps:
      1. Prepend a zero slice at hour=0 (no accumulation at init time).
      2. diff() along forecast_hour → per-hour values.
      3. Compute actual timestamps: init_time + forecast_hour * 1h.
      4. Flatten (init, hour) → time and return a (time, lat, lon) DataArray.
    """
    data = subset[var_name].load()  # triggers OPeNDAP fetch; shape (n_init,12,lat,lon)

    # NOTE: despite being named "fc.sfc.accumu" (accumulated), NCAR has already
    # deaccumulated the data to per-forecast-hour values before hosting.  Each
    # forecast_hour entry is the J/m² (or m for runoff) for THAT single hour,
    # not a running total from the init time.  Do NOT diff — just reshape.
    vals = data.values  # (n_init, n_hour, lat, lon)

    # Compute actual wall-clock timestamps for every (init, hour) cell
    init_times = subset["forecast_initial_time"].values       # datetime64[ns], shape (n_init,)
    hours_td = subset["forecast_hour"].values.astype(int).astype("timedelta64[h]")
    actual_times = (
        init_times[:, np.newaxis] + hours_td[np.newaxis, :]
    ).reshape(-1)  # shape (n_init * n_hour,)

    n_init, n_hour = len(init_times), len(hours_td)
    flat = vals.reshape(n_init * n_hour, *vals.shape[2:])

    return xr.DataArray(
        flat,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": actual_times,
            "latitude": subset["latitude"].values,
            "longitude": subset["longitude"].values,
        },
        attrs=data.attrs,
    ).sortby("time")



# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def _to_xy(da):
    """Rename latitude/longitude → y/x and sort to S→N (ascending y)."""
    return da.rename({"latitude": "y", "longitude": "x"}).sortby("y")


def retrieve_analysis(short_name, year, month, target_times):
    """Fetch one an.sfc variable; return (time, y, x) DataArray at target_times."""
    product_dir, param_code, ncar_var = VAR_INFO[short_name]
    path = _an_sfc_url(product_dir, param_code, year, month)
    print(f"    {short_name:6s}: {path.split('/')[-1]}")
    ds = open_opendap(path)
    subset = sel_bbox(ds, "latitude", "longitude", X0, Y0, X1, Y1)
    da = subset[ncar_var].load()
    da = da.sel(time=target_times, method="nearest")
    da.coords["time"] = target_times  # snap to exact reference timestamps
    ds.close()
    return _to_xy(da)


def retrieve_forecast(short_name, year, month, target_times, divide_by_3600=False):
    """Fetch one fc.sfc.accumu variable; diff to hourly; return at target_times.

    Also fetches the previous month's second-half file when target_times
    includes the first 6 hours of day 1 (which come from the prior month's
    last forecast run).
    """
    product_dir, param_code, ncar_var = VAR_INFO[short_name]
    target_pd = pd.DatetimeIndex(target_times)

    # Check whether we need data from the previous month's boundary file
    need_prev = any(t.month == month and t.day == 1 and t.hour < 7 for t in target_pd)

    urls = []
    if need_prev:
        urls.append(_fc_prev_half_url(product_dir, param_code, year, month))
    urls.extend(_fc_half_urls(product_dir, param_code, year, month))

    parts = []
    for url in urls:
        print(f"    {short_name:6s}: {url.split('/')[-1]}")
        ds = open_opendap(url)
        subset = sel_bbox(ds, "latitude", "longitude", X0, Y0, X1, Y1)
        parts.append(_fc_to_hourly(subset, ncar_var))
        ds.close()

    hourly = xr.concat(parts, dim="time").sortby("time")

    # Remove duplicate timestamps at file boundaries (e.g. init Jan1 06:00 appears
    # in both the Dec second-half and Jan first-half files)
    _, idx = np.unique(hourly.time.values, return_index=True)
    hourly = hourly.isel(time=idx)

    if divide_by_3600:
        hourly = hourly / 3600.0

    da = hourly.sel(time=target_times, method="nearest")
    da.coords["time"] = target_times
    return _to_xy(da)


def retrieve_height():
    """Fetch invariant geopotential Z and convert to metres above geoid."""
    print(f"    z:     {INVARIANT_PATH.split('/')[-1]}")
    ds = open_opendap(INVARIANT_PATH)
    subset = sel_bbox(ds, "latitude", "longitude", X0, Y0, X1, Y1)
    z = subset["Z"].isel(time=0, drop=True).load()
    height = z / 9.80665
    ds.close()
    return _to_xy(height)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_var(ncar_da, ref_da, label):
    """Print max/mean absolute and max relative difference; return max abs diff."""
    a = ncar_da.values.astype(np.float64)
    b = ref_da.values.astype(np.float64)
    diff = np.abs(a - b)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.abs(b) > 0, diff / np.abs(b), 0.0)
    print(
        f"    {label:30s}  "
        f"max_abs={diff.max():.4g}  mean_abs={diff.mean():.4g}  "
        f"max_rel={rel.max():.4g}"
    )
    return diff.max()


# ---------------------------------------------------------------------------
# Feature sections
# ---------------------------------------------------------------------------

def section_wind(ref):
    section("7. WIND — processing comparison")
    y, m, t = 2013, 1, ref.time.values
    print()
    u10  = retrieve_analysis("u10",  y, m, t)
    v10  = retrieve_analysis("v10",  y, m, t)
    u100 = retrieve_analysis("u100", y, m, t)
    v100 = retrieve_analysis("v100", y, m, t)
    fsr  = retrieve_analysis("fsr",  y, m, t)

    wnd10m        = np.sqrt(u10**2 + v10**2)
    wnd100m       = np.sqrt(u100**2 + v100**2)
    wnd_shear_exp = np.log(wnd10m / wnd100m) / np.log(10.0 / 100.0)
    az            = np.arctan2(u100.values, v100.values)
    wnd_azimuth   = xr.DataArray(
        np.where(az >= 0, az, az + 2 * np.pi),
        dims=wnd100m.dims, coords=wnd100m.coords,
    )
    roughness = fsr.where(fsr >= 0.0, 2e-4)

    print("\n  max absolute and relative differences vs CDS reference:")
    compare_var(wnd100m,       ref["wnd100m"],      "wnd100m [m/s]")
    compare_var(wnd_shear_exp, ref["wnd_shear_exp"], "wnd_shear_exp [-]")
    compare_var(wnd_azimuth,   ref["wnd_azimuth"],   "wnd_azimuth [rad]")
    compare_var(roughness,     ref["roughness"],      "roughness [m]")


def section_influx(ref):
    section("8. INFLUX — processing comparison")
    y, m, t = 2013, 1, ref.time.values
    print()
    ssrd = retrieve_forecast("ssrd", y, m, t, divide_by_3600=True)
    ssr  = retrieve_forecast("ssr",  y, m, t, divide_by_3600=True)
    fdir = retrieve_forecast("fdir", y, m, t, divide_by_3600=True)
    tisr = retrieve_forecast("tisr", y, m, t, divide_by_3600=True)

    albedo        = ((ssrd - ssr) / ssrd.where(ssrd != 0)).fillna(0.0)
    influx_direct  = fdir.clip(min=0.0)
    influx_toa     = tisr.clip(min=0.0)
    influx_diffuse = (ssrd - fdir).clip(min=0.0)

    print("\n  max absolute and relative differences vs CDS reference:")
    compare_var(influx_direct,  ref["influx_direct"],  "influx_direct [W/m²]")
    compare_var(influx_diffuse, ref["influx_diffuse"], "influx_diffuse [W/m²]")
    compare_var(influx_toa,     ref["influx_toa"],     "influx_toa [W/m²]")
    compare_var(albedo,         ref["albedo"],          "albedo [-]")
    print("  (solar_altitude/solar_azimuth are purely geometric — identical, skipped)")

    # Diagnostic: hour-by-hour tisr at centre of bbox to diagnose any time offset
    print("\n  DIAGNOSTIC — tisr (influx_toa) hour by hour at bbox centre:")
    yi = influx_toa.sizes["y"] // 2
    xi = influx_toa.sizes["x"] // 2
    lat_c = float(influx_toa.y[yi])
    lon_c = float(influx_toa.x[xi])
    print(f"  Location: lat={lat_c:.2f}, lon={lon_c:.2f}")
    print(f"  {'hour':>6}  {'NCAR':>10}  {'CDS':>10}  {'diff':>10}")
    for i, ts in enumerate(ref.time.values):
        h = pd.Timestamp(ts).hour
        ncar_val = float(influx_toa.isel(y=yi, x=xi).values[i])
        cds_val  = float(ref["influx_toa"].isel(y=yi, x=xi).values[i])
        print(f"  {h:02d}:00  {ncar_val:10.2f}  {cds_val:10.2f}  {ncar_val-cds_val:10.2f}")


def section_temperature(ref):
    section("9. TEMPERATURE — processing comparison")
    y, m, t = 2013, 1, ref.time.values
    print()
    t2m  = retrieve_analysis("t2m",  y, m, t)
    stl4 = retrieve_analysis("stl4", y, m, t)
    d2m  = retrieve_analysis("d2m",  y, m, t)

    print("\n  max absolute and relative differences vs CDS reference:")
    compare_var(t2m,  ref["temperature"],          "temperature [K]")
    compare_var(stl4, ref["soil temperature"],     "soil temperature [K]")
    compare_var(d2m,  ref["dewpoint temperature"], "dewpoint temperature [K]")


def section_runoff(ref):
    section("10. RUNOFF — processing comparison")
    y, m, t = 2013, 1, ref.time.values
    print()
    # Runoff units are metres (not J/m²); diff step gives per-hour metres — no /3600.
    ro = retrieve_forecast("ro", y, m, t, divide_by_3600=False)
    ro = ro.clip(min=0.0)

    print("\n  max absolute and relative differences vs CDS reference:")
    compare_var(ro, ref["runoff"], "runoff [m]")


def section_height(ref):
    section("11. HEIGHT — processing comparison")
    print()
    height = retrieve_height()

    print("\n  max absolute and relative differences vs CDS reference:")
    compare_var(height, ref["height"], "height [m]")


# ---------------------------------------------------------------------------
# Phase C entry point
# ---------------------------------------------------------------------------

def run_phase_c():
    section("PHASE C: Feature-by-feature processing and comparison with CDS")

    if not CACHE_PATH.exists():
        print(f"\n  ERROR: Reference cutout not found: {CACHE_PATH}")
        print("  Run the test suite with --cache-path first.")
        return

    print(f"\n  Loading reference: {CACHE_PATH}")
    ref = xr.open_dataset(CACHE_PATH)
    print(f"  vars : {list(ref.data_vars)}")
    print(f"  time : {pd.Timestamp(ref.time.values[0])} .. {pd.Timestamp(ref.time.values[-1])}  (n={len(ref.time)})")
    print(f"  x    : {float(ref.x.min()):.2f} .. {float(ref.x.max()):.2f}")
    print(f"  y    : {float(ref.y.min()):.2f} .. {float(ref.y.max()):.2f}")

    section_wind(ref)
    section_influx(ref)
    section_temperature(ref)
    section_runoff(ref)
    section_height(ref)

    ref.close()
    section("PHASE C COMPLETE")


# ===========================================================================
# PHASE D: Regridding — verify xr.interp() matches CDS coarse-grid output
# ===========================================================================

COARSE_CACHE_PATH = Path("test-cache/cutout_era5_coarse.nc")


def run_phase_d():
    section("PHASE D: Regridding — xr.interp() vs CDS coarse cutout")

    if not COARSE_CACHE_PATH.exists():
        print(f"\n  ERROR: Coarse reference cutout not found: {COARSE_CACHE_PATH}")
        return

    print(f"\n  Loading coarse reference: {COARSE_CACHE_PATH}")
    ref = xr.open_dataset(COARSE_CACHE_PATH)
    print(f"  vars : {list(ref.data_vars)}")
    print(f"  x    : {float(ref.x.min()):.4f} .. {float(ref.x.max()):.4f}  "
          f"(n={len(ref.x)}, dx≈{float(ref.x[1]-ref.x[0]):.4f})")
    print(f"  y    : {float(ref.y.min()):.4f} .. {float(ref.y.max()):.4f}  "
          f"(n={len(ref.y)}, dy≈{float(ref.y[1]-ref.y[0]):.4f})")
    print(f"  time : {pd.Timestamp(ref.time.values[0])} .. "
          f"{pd.Timestamp(ref.time.values[-1])}  (n={len(ref.time)})")

    # Target grid from the coarse reference
    target_x = ref.x.values
    target_y = ref.y.values
    t = ref.time.values
    y, m = 2013, 1

    # -----------------------------------------------------------------------
    # Test variables: one analysis (u100) and one forecast-accumulated (tisr)
    # -----------------------------------------------------------------------
    print("\n  Step 1: Fetch u100 at native 0.25° resolution, interpolate, compare")
    u100_native = retrieve_analysis("u100", y, m, t)
    # u100_native has dims (time, y, x) on the fine 0.25° grid
    u100_interp = u100_native.interp(x=target_x, y=target_y, method="linear")

    # Reference wnd100m requires v100 too — compare u100 directly against
    # the raw analysis values embedded in the coarse wnd100m.
    # Instead, test via wnd100m (magnitude) since that's what the ref stores.
    v100_native = retrieve_analysis("v100", y, m, t)
    v100_interp = v100_native.interp(x=target_x, y=target_y, method="linear")
    wnd100m_interp = np.sqrt(u100_interp**2 + v100_interp**2)
    compare_var(wnd100m_interp, ref["wnd100m"], "wnd100m (interp) [m/s]")

    print("\n  Step 2: Fetch tisr at native 0.25° resolution, interpolate, compare")
    tisr_native = retrieve_forecast("tisr", y, m, t, divide_by_3600=True)
    tisr_interp = tisr_native.interp(x=target_x, y=target_y, method="linear")
    tisr_interp = tisr_interp.clip(min=0.0)
    compare_var(tisr_interp, ref["influx_toa"], "influx_toa (interp) [W/m²]")

    print("\n  Step 3: Fetch height (invariant), interpolate, compare")
    height_native = retrieve_height()
    height_interp = height_native.interp(x=target_x, y=target_y, method="linear")
    compare_var(height_interp, ref["height"], "height (interp) [m]")

    print()
    ref.close()
    section("PHASE D COMPLETE")


# ===========================================================================
# main
# ===========================================================================

def main():
    if "--influx" in sys.argv:
        if not CACHE_PATH.exists():
            print(f"ERROR: Reference cutout not found: {CACHE_PATH}")
            return
        ref = xr.open_dataset(CACHE_PATH)
        section_influx(ref)
        ref.close()
    elif "--phase-d" in sys.argv:
        run_phase_d()
    elif "--phase-c" in sys.argv:
        run_phase_c()
    else:
        run_phase_b()


if __name__ == "__main__":
    main()
