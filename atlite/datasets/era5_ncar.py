# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading ERA5 data from NCAR's THREDDS/OPeNDAP server.

Fetches ERA5 data from NCAR's Research Data Archive (RDA, dataset d633000)
via OPeNDAP server-side subsetting.  This eliminates the multi-hour CDS queue
while producing numerically identical output to the ``era5`` module.
No authentication is required.

Usage
-----
    cutout = atlite.Cutout(
        path="my_cutout.nc",
        module="era5-ncar",
        x=slice(-10, 5),
        y=slice(35, 44),
        time="2024",
    )
    cutout.prepare()

Technical notes
---------------
- OPeNDAP endpoint: https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/
- Native resolution: 0.25° global grid, latitude N→S (90→−90), longitude 0→360.
- Analysis surface variables (``e5.oper.an.sfc``): one file per variable per month.
- Forecast accumulated variables (``e5.oper.fc.sfc.accumu``): two files per month
  (~15-day halves).  Values are *already per-hour* (J m⁻²) — no deaccumulation
  needed; divide by 3600 to convert to W m⁻².
- Bounding boxes crossing the 0° meridian are handled with two ``.sel()`` calls
  and coordinate reassignment; ``.roll()`` is never called on lazy pydap datasets
  to avoid triggering a full global download (THREDDS hard limit: 500 MB/request).
- Requires ``pydap`` (``pip install pydap``); netCDF4 is typically not compiled
  with OPeNDAP support.
"""

import calendar
import logging
import os
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random_exponential

import numpy as np
import pandas as pd
import xarray as xr

from atlite.datasets.era5 import sanitize_influx, sanitize_runoff, sanitize_wind
from atlite.pv.solar_position import SolarPosition

logger = logging.getLogger(__name__)

MAX_WORKERS = 8  # concurrent OPeNDAP requests

# Fix 1: module-level semaphore caps total concurrent THREDDS connections
# across all features that atlite prepares in parallel.
_semaphore = threading.Semaphore(MAX_WORKERS)

# Fix 2: per-(url, tmpdir) cache so the prev-half of month N+1 (== second-half
# of month N) is not downloaded twice.
_url_cache: dict[tuple[str, str], str] = {}
_url_cache_lock = threading.Lock()          # guards _url_cache and _url_locks
_url_locks: dict[tuple[str, str], threading.Lock] = {}

# Fix 4: thread-local requests.Session for HTTP keep-alive reuse.
_thread_local = threading.local()


def _get_url_lock(cache_key: tuple[str, str]) -> threading.Lock:
    """Return the per-URL lock, creating it if necessary (thread-safe)."""
    with _url_cache_lock:
        if cache_key not in _url_locks:
            _url_locks[cache_key] = threading.Lock()
        return _url_locks[cache_key]


def _get_session():
    """Return a thread-local requests.Session (created on first access)."""
    if not hasattr(_thread_local, "session"):
        import requests
        _thread_local.session = requests.Session()
    return _thread_local.session

# ---------------------------------------------------------------------------
# Module-level constants required by atlite
# ---------------------------------------------------------------------------

crs = 4326
features = {
    "height": ["height"],
    "wind": ["wnd100m", "wnd_shear_exp", "wnd_azimuth", "roughness"],
    "influx": [
        "influx_toa",
        "influx_direct",
        "influx_diffuse",
        "albedo",
        "solar_altitude",
        "solar_azimuth",
    ],
    "temperature": ["temperature", "soil temperature", "dewpoint temperature"],
    "runoff": ["runoff"],
}
static_features = {"height"}

OPENDAP_BASE = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/"

# Maps ERA5 short names → (product_dir, param_code, ncar_varname).
# Variable naming in the NCAR NetCDF files (verified empirically):
#   - Most an.sfc vars:  uppercase with VAR_ prefix  (VAR_10U, VAR_2T, …)
#   - fsr, stl4:         uppercase, no prefix         (FSR, STL4)
#   - All fc.sfc.accumu: uppercase, no prefix         (SSRD, SSR, FDIR, …)
#   - Invariant:         uppercase, no prefix         (Z)
VAR_MAP = {
    "u10":  ("e5.oper.an.sfc",        "128_165_10u",  "VAR_10U"),
    "v10":  ("e5.oper.an.sfc",        "128_166_10v",  "VAR_10V"),
    "u100": ("e5.oper.an.sfc",        "228_246_100u", "VAR_100U"),
    "v100": ("e5.oper.an.sfc",        "228_247_100v", "VAR_100V"),
    "fsr":  ("e5.oper.an.sfc",        "128_244_fsr",  "FSR"),
    "t2m":  ("e5.oper.an.sfc",        "128_167_2t",   "VAR_2T"),
    "d2m":  ("e5.oper.an.sfc",        "128_168_2d",   "VAR_2D"),
    "stl4": ("e5.oper.an.sfc",        "128_236_stl4", "STL4"),
    "ssrd": ("e5.oper.fc.sfc.accumu", "128_169_ssrd", "SSRD"),
    "ssr":  ("e5.oper.fc.sfc.accumu", "128_176_ssr",  "SSR"),
    "fdir": ("e5.oper.fc.sfc.accumu", "228_021_fdir", "FDIR"),
    "tisr": ("e5.oper.fc.sfc.accumu", "128_212_tisr", "TISR"),
    "ro":   ("e5.oper.fc.sfc.accumu", "128_205_ro",   "RO"),
    "z":    ("e5.oper.invariant",     "128_129_z",    "Z"),
}

_INVARIANT_PATH = (
    "e5.oper.invariant/197901/"
    "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
)

# Raw ERA5 short names required for each feature.
_FEATURE_VARS = {
    "wind":        ["u10", "v10", "u100", "v100", "fsr"],
    "influx":      ["ssrd", "ssr", "fdir", "tisr"],
    "temperature": ["t2m", "stl4", "d2m"],
    "runoff":      ["ro"],
    "height":      ["z"],
}

# Forecast radiation vars stored as J/m² — convert to W/m².
_FC_DIVIDE_BY_3600 = {"ssrd", "ssr", "fdir", "tisr"}

# Extra degrees added to each side of the bbox when fetching native-resolution
# data, so bilinear interpolation has support even at the target grid edges.
_BBOX_PAD = 0.5


# ---------------------------------------------------------------------------
# OPeNDAP open + spatial subset
# ---------------------------------------------------------------------------

@retry(
    wait=wait_random_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _open_opendap(path):
    """Open an OPeNDAP dataset (lazy — metadata only).

    Uses ``engine="pydap"`` because netCDF4 is typically not compiled with
    OPeNDAP support.  The pydap DAP2 deprecation warning is suppressed.
    Retried up to 5 times with random exponential backoff on transient errors.
    """
    url = OPENDAP_BASE + path
    logger.debug("era5-ncar OPeNDAP: %s", url)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xr.open_dataset(url, engine="pydap", session=_get_session())


def _sel_bbox(ds, x0, y0, x1, y1):
    """Subset *ds* to (x0, y0, x1, y1) handling 0–360 longitudes.

    If the bbox crosses the 0° meridian (e.g. x0=-4, x1=1.5), two ``.sel()``
    calls are made and the results concatenated.  ``.roll()`` is never used
    on lazy pydap datasets — it would force a full global download.
    """
    lat = ds.coords["latitude"].values
    lat_slice = slice(y1, y0) if lat[0] > lat[-1] else slice(y0, y1)

    x0_360, x1_360 = x0 % 360, x1 % 360

    if x0_360 <= x1_360:
        return ds.sel(latitude=lat_slice, longitude=slice(x0_360, x1_360))

    # Bbox wraps across 0°. High segment (e.g. 356–360) → negative coords.
    high = ds.sel(latitude=lat_slice, longitude=slice(x0_360, 360))
    low  = ds.sel(latitude=lat_slice, longitude=slice(0, x1_360))
    high = high.assign_coords(longitude=high.coords["longitude"].values - 360)
    vars_with_lon = [v for v in ds.data_vars if "longitude" in ds[v].dims]
    return (
        xr.concat(
            [high[vars_with_lon], low[vars_with_lon]],
            dim="longitude",
            data_vars="all",
        )
        .sortby("longitude")
    )


def _to_xy(da):
    """Rename latitude/longitude → y/x, sort S→N, round coords to 5 d.p."""
    da = da.rename({"latitude": "y", "longitude": "x"}).sortby("y")
    return da.assign_coords(
        x=np.round(da.x.values.astype(float), 5),
        y=np.round(da.y.values.astype(float), 5),
    )


def _grids_align(ds, coords, tol=1e-4):
    """Return True if ds and the target grid share the same resolution.

    The fetched dataset is always larger than the target (padded by _BBOX_PAD),
    so a shape/value equality check never passes.  Matching resolution is
    sufficient: when True, `.sel(method="nearest")` extracts the target points
    exactly rather than interpolating between them.
    """
    src_x = ds.coords["x"].values
    src_y = ds.coords["y"].values
    tgt_x = coords["x"].values
    tgt_y = coords["y"].values
    if len(src_x) < 2 or len(tgt_x) < 2 or len(src_y) < 2 or len(tgt_y) < 2:
        return False
    return (
        abs(np.diff(src_x[:2]).item() - np.diff(tgt_x[:2]).item()) < tol
        and abs(np.diff(src_y[:2]).item() - np.diff(tgt_y[:2]).item()) < tol
    )


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def _an_sfc_url(product_dir, param_code, year, month):
    ym   = f"{year}{month:02d}"
    last = calendar.monthrange(year, month)[1]
    fname = f"{product_dir}.{param_code}.ll025sc.{ym}0100_{ym}{last:02d}23.nc"
    return f"{product_dir}/{ym}/{fname}"


def _fc_half_urls(product_dir, param_code, year, month):
    ym  = f"{year}{month:02d}"
    ny  = year + (1 if month == 12 else 0)
    nm  = month % 12 + 1
    nym = f"{ny}{nm:02d}"
    prefix = f"{product_dir}/{ym}/{product_dir}.{param_code}.ll025sc"
    return [
        f"{prefix}.{ym}0106_{ym}1606.nc",
        f"{prefix}.{ym}1606_{nym}0106.nc",
    ]


def _fc_prev_half_url(product_dir, param_code, year, month):
    py  = year - 1 if month == 1 else year
    pm  = 12 if month == 1 else month - 1
    pym = f"{py}{pm:02d}"
    ym  = f"{year}{month:02d}"
    prefix = f"{product_dir}/{pym}/{product_dir}.{param_code}.ll025sc"
    return f"{prefix}.{pym}1606_{ym}0106.nc"


# ---------------------------------------------------------------------------
# Forecast: flatten (forecast_initial_time, forecast_hour) → (time,)
# ---------------------------------------------------------------------------

def _fc_to_hourly(subset, ncar_varname):
    """Load forecast data and return a (time, latitude, longitude) DataArray.

    Despite the ``fc.sfc.accumu`` product name, NCAR stores *per-forecast-hour*
    values (J m⁻² or m for runoff), not running totals.  No differencing
    is needed.  Timestamps: ``forecast_initial_time + forecast_hour × 1 h``.
    """
    data = subset[ncar_varname].load()  # (n_init, n_hour, lat, lon)
    vals = data.values

    init_times = subset["forecast_initial_time"].values
    hours_td   = subset["forecast_hour"].values.astype(int).astype("timedelta64[h]")
    actual_times = (
        init_times[:, np.newaxis] + hours_td[np.newaxis, :]
    ).reshape(-1)

    n_init, n_hour = len(init_times), len(hours_td)
    flat = vals.reshape(n_init * n_hour, *vals.shape[2:])

    return xr.DataArray(
        flat,
        dims=["time", "latitude", "longitude"],
        coords={
            "time":      actual_times,
            "latitude":  subset["latitude"].values,
            "longitude": subset["longitude"].values,
        },
        attrs=data.attrs,
    )


# ---------------------------------------------------------------------------
# Forecast half-file helpers
# ---------------------------------------------------------------------------

def _retrieve_fc_half(url, ncar_var, x0, y0, x1, y1, tmpdir):
    """Fetch one forecast half-file, write to a temp NetCDF file.

    Results are cached by (url, tmpdir): the prev-half of month N+1 is the
    same file as the second-half of month N, so the second caller returns the
    cached path without a second OPeNDAP round-trip.

    The DataArray has ``dims=["time", "latitude", "longitude"]``
    (``_to_xy`` not yet applied).  The file has a single variable ``"data"``.
    Assembly's ``sel(time=mt)`` filters to the correct month's timestamps, so
    no init-time subsetting is required here.
    """
    cache_key = (url, str(tmpdir))
    url_lock = _get_url_lock(cache_key)

    with url_lock:
        if cache_key in _url_cache:
            return _url_cache[cache_key]

        with _semaphore:
            with _open_opendap(url) as ds:
                subset = _sel_bbox(ds, x0, y0, x1, y1)
                da = _fc_to_hourly(subset, ncar_var)

        fd, path = tempfile.mkstemp(suffix=".nc", dir=tmpdir)
        os.close(fd)
        da.to_dataset(name="data").to_netcdf(path)
        _url_cache[cache_key] = path
        return path


# ---------------------------------------------------------------------------
# Analysis/invariant fetch helper
# ---------------------------------------------------------------------------

def _retrieve_var(short_name, year, month, x0, y0, x1, y1, tmpdir):
    """Fetch one invariant or an.sfc variable, write to a temp NetCDF file.

    Returns the path to the temp file.  The DataArray inside has y/x coords
    (``_to_xy`` already applied).  The file has a single variable named
    ``"data"``.
    """
    product_dir, param_code, ncar_var = VAR_MAP[short_name]

    if product_dir == "e5.oper.invariant":
        with _semaphore:
            with _open_opendap(_INVARIANT_PATH) as ds:
                subset = _sel_bbox(ds, x0, y0, x1, y1)
                z = subset["Z"].isel(time=0, drop=True).load()
        da = _to_xy(z / 9.80665)
    else:
        # e5.oper.an.sfc
        url = _an_sfc_url(product_dir, param_code, year, month)
        with _semaphore:
            with _open_opendap(url) as ds:
                subset = _sel_bbox(ds, x0, y0, x1, y1)
                da = _to_xy(subset[ncar_var].load())

    fd, path = tempfile.mkstemp(suffix=".nc", dir=tmpdir)
    os.close(fd)
    da.to_dataset(name="data").to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# Concurrent fetch helpers
# ---------------------------------------------------------------------------

def _months(coords):
    t = pd.DatetimeIndex(coords["time"].values)
    return sorted(set(zip(t.year, t.month)))


def _bbox(coords):
    """Padded bounding box for fetching native-resolution data."""
    return (
        float(coords["x"].min()) - _BBOX_PAD,
        float(coords["y"].min()) - _BBOX_PAD,
        float(coords["x"].max()) + _BBOX_PAD,
        float(coords["y"].max()) + _BBOX_PAD,
    )


def _fetch_vars(short_names, coords, tmpdir=None):
    """Fetch all requested variables concurrently.

    Each task downloads its data to a temporary NetCDF file, then all files
    are opened lazily (dask-backed) so assembly and interpolation do not hold
    multiple variables in RAM simultaneously.

    Each forecast half-file is its own parallel task (one task per URL), so
    the thread pool parallelises across both (var, month) and the three
    half-files within each forecast month.  Results are assembled and
    interpolated to the cutout grid as a single Dataset per group (static /
    time-varying) to avoid redundant grid-weight computation.

    Returns a dict mapping short_name → DataArray on the cutout's (time, y, x)
    grid (or (y, x) for the invariant height variable).
    """
    x0, y0, x1, y1 = _bbox(coords)
    months = _months(coords)
    t = pd.DatetimeIndex(coords["time"].values)

    # Build task list.
    # Each task is a tuple: (kind, sn, year, month, url)
    #   kind ∈ {"invariant", "an_sfc", "fc_half"}
    tasks = []
    for sn in short_names:
        product_dir, param_code, ncar_var = VAR_MAP[sn]
        if product_dir == "e5.oper.invariant":
            tasks.append(("invariant", sn, None, None, None))
        elif product_dir == "e5.oper.an.sfc":
            for year, month in months:
                tasks.append(("an_sfc", sn, year, month, None))
        else:
            for year, month in months:
                urls = [
                    _fc_prev_half_url(product_dir, param_code, year, month),
                    *_fc_half_urls(product_dir, param_code, year, month),
                ]
                for url in urls:
                    tasks.append(("fc_half", sn, year, month, url))

    n_total = len(tasks)
    logger.info(
        "era5-ncar: submitting %d tasks for [%s] with %d workers",
        n_total, ", ".join(short_names), MAX_WORKERS,
    )
    t_start = time.time()

    # raw storage (workers return paths to temp NetCDF files):
    #   ("invariant", sn)            → str path
    #   ("an_sfc", sn, year, month)  → str path
    #   ("fc_half", sn, year, month) → list of str paths (one per half-file)
    raw = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {}
        for kind, sn, yr, mo, url in tasks:
            if kind == "invariant":
                f = pool.submit(_retrieve_var, sn, None, None, x0, y0, x1, y1, tmpdir)
            elif kind == "an_sfc":
                f = pool.submit(_retrieve_var, sn, yr, mo, x0, y0, x1, y1, tmpdir)
            else:
                _, _, ncar_var = VAR_MAP[sn]
                f = pool.submit(_retrieve_fc_half, url, ncar_var, x0, y0, x1, y1, tmpdir)
            future_map[f] = (kind, sn, yr, mo)

        for n_done, future in enumerate(as_completed(future_map), 1):
            kind, sn, yr, mo = future_map[future]
            result = future.result()  # raises immediately on worker error
            elapsed = time.time() - t_start
            rate = elapsed / n_done
            eta = rate * (n_total - n_done)
            logger.info(
                "era5-ncar: [%d/%d] %-6s %s-%02d  (%.0fs elapsed, ETA %.0fs)",
                n_done, n_total, sn, yr if yr else "—", mo or 0, elapsed, eta,
            )
            if kind == "invariant":
                raw[("invariant", sn)] = result
            elif kind == "an_sfc":
                raw[("an_sfc", sn, yr, mo)] = result
            else:
                fc_key = ("fc_half", sn, yr, mo)
                if fc_key not in raw:
                    raw[fc_key] = []
                raw[fc_key].append(result)

    # Open temp files lazily (dask-backed) and assemble per-variable.
    assembled_static = {}
    assembled_tv = {}

    for sn in short_names:
        product_dir = VAR_MAP[sn][0]

        if product_dir == "e5.oper.invariant":
            path = raw[("invariant", sn)]
            # Invariant is small (y, x) — open without time chunking.
            assembled_static[sn] = xr.open_dataset(path, chunks={})["data"]

        elif product_dir == "e5.oper.an.sfc":
            parts = []
            for year, month in months:
                path = raw[("an_sfc", sn, year, month)]
                da = xr.open_dataset(path, chunks={"time": 24})["data"]
                mt = t[(t.year == year) & (t.month == month)]
                if len(mt):
                    sel = (
                        da.sel(time=mt, method="nearest", tolerance=pd.Timedelta("30min"))
                        .assign_coords(time=mt.values)
                    )
                    parts.append(sel)
            assembled_tv[sn] = xr.concat(parts, dim="time")

        else:
            # fc.sfc.accumu — assemble half-files per month, then concat months.
            # Temp files have latitude/longitude coords; _to_xy applied below.
            month_parts = []
            for year, month in months:
                paths = raw[("fc_half", sn, year, month)]
                parts = [
                    xr.open_dataset(p, chunks={"time": 24})["data"]
                    for p in paths
                ]
                hourly = xr.concat(parts, dim="time")
                # Deduplicate on the time coordinate (small, safe to load).
                _, idx = np.unique(hourly.time.values, return_index=True)
                hourly = hourly.isel(time=idx)
                if sn in _FC_DIVIDE_BY_3600:
                    hourly = hourly / 3600.0
                hourly = _to_xy(hourly)
                mt = t[(t.year == year) & (t.month == month)]
                if len(mt):
                    sel = (
                        hourly.sel(time=mt, method="nearest", tolerance=pd.Timedelta("30min"))
                        .assign_coords(time=mt.values)
                    )
                    month_parts.append(sel)
            assembled_tv[sn] = xr.concat(month_parts, dim="time")

    # Batch interpolation: one .interp() call per group to reuse grid weights.
    # .load() is called on each interpolated Dataset to materialise the result
    # at the (smaller) target-grid resolution, which releases the file handles
    # opened above and allows tmpdir cleanup on all platforms.
    assembled = {}

    if assembled_static:
        ds_static = xr.Dataset(assembled_static)
        if _grids_align(ds_static, coords):
            ds_static_out = ds_static.sel(
                x=coords["x"].values, y=coords["y"].values, method="nearest"
            ).load()
        else:
            ds_static_out = ds_static.interp(
                x=coords["x"].values, y=coords["y"].values, method="linear"
            ).load()
        for sn in assembled_static:
            assembled[sn] = ds_static_out[sn]

    if assembled_tv:
        ds_tv = xr.Dataset(assembled_tv)
        if _grids_align(ds_tv, coords):
            ds_tv_out = ds_tv.sel(
                x=coords["x"].values, y=coords["y"].values, method="nearest"
            ).load()
        else:
            ds_tv_out = ds_tv.interp(
                x=coords["x"].values, y=coords["y"].values, method="linear"
            ).load()
        for sn in assembled_tv:
            assembled[sn] = ds_tv_out[sn]

    # Clear temp-file encoding (contiguous, source, original_shape) so it
    # does not conflict with compression settings applied by cutout_prepare.
    for da in assembled.values():
        da.encoding.clear()

    # Clean up cache entries for this tmpdir now that all downloads and
    # assembly are done.  Entries from other tmpdirs (concurrent features
    # sharing a different prepare() call) are left untouched.
    if tmpdir is not None:
        tmpdir_str = str(tmpdir)
        with _url_cache_lock:
            stale = [k for k in _url_cache if k[1] == tmpdir_str]
            for k in stale:
                del _url_cache[k]
                _url_locks.pop(k, None)

    return assembled


# ---------------------------------------------------------------------------
# Feature assemblers  (receive pre-fetched vars dict, compute derived fields)
# ---------------------------------------------------------------------------

def get_data_wind(coords, tmpdir=None):
    v = _fetch_vars(_FEATURE_VARS["wind"], coords, tmpdir=tmpdir)
    wnd10m  = np.sqrt(v["u10"]**2  + v["v10"]**2)
    wnd100m = np.sqrt(v["u100"]**2 + v["v100"]**2)
    wnd_shear_exp = (
        np.log(wnd10m / wnd100m) / np.log(10.0 / 100.0)
    ).assign_attrs(units="", long_name="wind shear exponent")
    az = np.arctan2(v["u100"], v["v100"])  # stays lazy via __array_ufunc__
    wnd_azimuth = az.where(az >= 0, az + 2 * np.pi)
    return xr.Dataset(
        {
            "wnd100m":       wnd100m.rename("wnd100m"),
            "wnd_shear_exp": wnd_shear_exp.rename("wnd_shear_exp"),
            "wnd_azimuth":   wnd_azimuth.rename("wnd_azimuth"),
            "roughness":     v["fsr"].rename("roughness"),
        }
    )


def get_data_influx(coords, tmpdir=None):
    v = _fetch_vars(_FEATURE_VARS["influx"], coords, tmpdir=tmpdir)
    ssrd, ssr, fdir, tisr = v["ssrd"], v["ssr"], v["fdir"], v["tisr"]
    albedo         = ((ssrd - ssr) / ssrd.where(ssrd != 0)).fillna(0.0)
    influx_diffuse = ssrd - fdir
    ds = xr.Dataset(
        {
            "influx_direct":  fdir.rename("influx_direct"),
            "influx_diffuse": influx_diffuse.rename("influx_diffuse"),
            "influx_toa":     tisr.rename("influx_toa"),
            "albedo":         albedo.rename("albedo"),
        }
    )
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift=pd.to_timedelta("-30 minutes"))
    sp = sp.rename({name: f"solar_{name}" for name in sp.data_vars})
    return xr.merge([ds, sp])


def get_data_temperature(coords, tmpdir=None):
    v = _fetch_vars(_FEATURE_VARS["temperature"], coords, tmpdir=tmpdir)
    return xr.Dataset(
        {
            "temperature":          v["t2m"].rename("temperature"),
            "soil temperature":     v["stl4"].rename("soil temperature"),
            "dewpoint temperature": v["d2m"].rename("dewpoint temperature"),
        }
    )


def get_data_runoff(coords, tmpdir=None):
    v = _fetch_vars(_FEATURE_VARS["runoff"], coords, tmpdir=tmpdir)
    return xr.Dataset({"runoff": v["ro"].rename("runoff")})


def get_data_height(coords, tmpdir=None):
    v = _fetch_vars(_FEATURE_VARS["height"], coords, tmpdir=tmpdir)
    return xr.Dataset({"height": v["z"].rename("height")})


# ---------------------------------------------------------------------------
# Entry point — same signature as era5.get_data()
# ---------------------------------------------------------------------------

def get_data(cutout, feature, tmpdir=None, lock=None, **creation_parameters):
    """Retrieve ERA5 data from NCAR THREDDS/OPeNDAP.

    Same interface as ``atlite.datasets.era5.get_data()``.
    """
    coords = cutout.coords
    sanitize = creation_parameters.get("sanitize", True)

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    if func is None:
        raise ValueError(
            f"era5-ncar: no retrieval function for feature {feature!r}. "
            f"Available: {list(features)}"
        )

    logger.info("era5-ncar: fetching feature '%s'...", feature)

    if tmpdir is not None:
        # Normal workflow: atlite's prepare machinery manages tmpdir lifecycle.
        ds = func(coords, tmpdir=tmpdir)
    else:
        # Direct call with no tmpdir: use a TemporaryDirectory so temp files
        # are cleaned up automatically.  _fetch_vars already .load()s after
        # interpolation, so ds is in-memory when the context exits.
        with tempfile.TemporaryDirectory() as _tmpdir:
            ds = func(coords, tmpdir=_tmpdir)

    if sanitize and sanitize_func is not None:
        ds = sanitize_func(ds)

    if feature in static_features:
        return ds.squeeze()
    return ds
