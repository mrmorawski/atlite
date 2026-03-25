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
import hashlib
import logging
import os
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import traceback

import requests.exceptions
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import numpy as np
import pandas as pd
import xarray as xr

from atlite.datasets.era5 import sanitize_influx, sanitize_runoff, sanitize_wind
from atlite.pv.solar_position import SolarPosition

logger = logging.getLogger(__name__)

MAX_WORKERS = 8  # concurrent OPeNDAP requests

# Module-level pool caps total concurrent THREDDS connections across all
# features that dask runs in parallel.  Threads are created on demand.
_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_nc_write_lock = threading.Lock()

# Thread-local requests.Session for HTTP keep-alive reuse.
_thread_local = threading.local()


_REQUEST_TIMEOUT = (30, 300)  # (connect, read) seconds


def _get_session():
    """Return a thread-local requests.Session (created on first access).

    A default timeout is set via a mounted adapter so that stalled THREDDS
    connections don't block a pool thread indefinitely.
    """
    if not hasattr(_thread_local, "session"):
        from requests.adapters import HTTPAdapter

        class _TimeoutAdapter(HTTPAdapter):
            def send(self, *args, **kwargs):
                kwargs.setdefault("timeout", _REQUEST_TIMEOUT)
                return super().send(*args, **kwargs)

        s = requests.Session()
        s.mount("http://", _TimeoutAdapter())
        s.mount("https://", _TimeoutAdapter())
        _thread_local.session = s
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
    "u10": ("e5.oper.an.sfc", "128_165_10u", "VAR_10U"),
    "v10": ("e5.oper.an.sfc", "128_166_10v", "VAR_10V"),
    "u100": ("e5.oper.an.sfc", "228_246_100u", "VAR_100U"),
    "v100": ("e5.oper.an.sfc", "228_247_100v", "VAR_100V"),
    "fsr": ("e5.oper.an.sfc", "128_244_fsr", "FSR"),
    "t2m": ("e5.oper.an.sfc", "128_167_2t", "VAR_2T"),
    "d2m": ("e5.oper.an.sfc", "128_168_2d", "VAR_2D"),
    "stl4": ("e5.oper.an.sfc", "128_236_stl4", "STL4"),
    "ssrd": ("e5.oper.fc.sfc.accumu", "128_169_ssrd", "SSRD"),
    "ssr": ("e5.oper.fc.sfc.accumu", "128_176_ssr", "SSR"),
    "fdir": ("e5.oper.fc.sfc.accumu", "228_021_fdir", "FDIR"),
    "tisr": ("e5.oper.fc.sfc.accumu", "128_212_tisr", "TISR"),
    "ro": ("e5.oper.fc.sfc.accumu", "128_205_ro", "RO"),
    "z": ("e5.oper.invariant", "128_129_z", "Z"),
}

_INVARIANT_PATH = (
    "e5.oper.invariant/197901/"
    "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
)

# Raw ERA5 short names required for each feature.
_FEATURE_VARS = {
    "wind": ["u10", "v10", "u100", "v100", "fsr"],
    "influx": ["ssrd", "ssr", "fdir", "tisr"],
    "temperature": ["t2m", "stl4", "d2m"],
    "runoff": ["ro"],
    "height": ["z"],
}

# Forecast radiation vars stored as J/m² — convert to W/m².
_FC_DIVIDE_BY_3600 = {"ssrd", "ssr", "fdir", "tisr"}

# Extra degrees added to each side of the bbox when fetching native-resolution
# data, so bilinear interpolation has support even at the target grid edges.
_BBOX_PAD = 0.5


# ---------------------------------------------------------------------------
# OPeNDAP open + spatial subset
# ---------------------------------------------------------------------------


def _open_opendap(path):
    """Open an OPeNDAP dataset (lazy — metadata only).

    Uses ``engine="pydap"`` because netCDF4 is typically not compiled with
    OPeNDAP support.  The pydap DAP2 deprecation warning is suppressed.
    Transient errors are retried by the calling functions.
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
    low = ds.sel(latitude=lat_slice, longitude=slice(0, x1_360))
    high = high.assign_coords(longitude=high.coords["longitude"].values - 360)
    vars_with_lon = [v for v in ds.data_vars if "longitude" in ds[v].dims]
    return xr.concat(
        [high[vars_with_lon], low[vars_with_lon]],
        dim="longitude",
        data_vars="all",
    ).sortby("longitude")


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
    ym = f"{year}{month:02d}"
    last = calendar.monthrange(year, month)[1]
    fname = f"{product_dir}.{param_code}.ll025sc.{ym}0100_{ym}{last:02d}23.nc"
    return f"{product_dir}/{ym}/{fname}"


def _fc_half_urls(product_dir, param_code, year, month):
    ym = f"{year}{month:02d}"
    ny = year + (1 if month == 12 else 0)
    nm = month % 12 + 1
    nym = f"{ny}{nm:02d}"
    prefix = f"{product_dir}/{ym}/{product_dir}.{param_code}.ll025sc"
    return [
        f"{prefix}.{ym}0106_{ym}1606.nc",
        f"{prefix}.{ym}1606_{nym}0106.nc",
    ]


def _fc_prev_half_url(product_dir, param_code, year, month):
    py = year - 1 if month == 1 else year
    pm = 12 if month == 1 else month - 1
    pym = f"{py}{pm:02d}"
    ym = f"{year}{month:02d}"
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
    hours_td = subset["forecast_hour"].values.astype(int).astype("timedelta64[h]")
    actual_times = (init_times[:, np.newaxis] + hours_td[np.newaxis, :]).reshape(-1)

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
    )


# ---------------------------------------------------------------------------
# Forecast half-file fetch
# ---------------------------------------------------------------------------


def _cache_key(short_name, x0, y0, x1, y1, year=None, month=None, url=None):
    """Return a deterministic filename for caching a raw download."""
    if url is not None:
        # fc half-files: hash the URL (unique per half-month)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"era5_ncar_{short_name}_fc_{url_hash}.nc"
    elif year is not None and month is not None:
        # Analysis surface: variable + year-month + bbox
        bbox_hash = hashlib.md5(
            f"{x0:.3f}_{y0:.3f}_{x1:.3f}_{y1:.3f}".encode()
        ).hexdigest()[:8]
        return f"era5_ncar_{short_name}_{year}_{month:02d}_{bbox_hash}.nc"
    else:
        # Invariant
        bbox_hash = hashlib.md5(
            f"{x0:.3f}_{y0:.3f}_{x1:.3f}_{y1:.3f}".encode()
        ).hexdigest()[:8]
        return f"era5_ncar_{short_name}_inv_{bbox_hash}.nc"


def _retrieve_var(short_name, x0, y0, x1, y1, tmpdir, year=None, month=None, url=None):
    """Fetch one ERA5 variable from NCAR OPeNDAP, write to a temp NetCDF file.

    Handles all three product types based on ``VAR_MAP[short_name]``:
    - ``e5.oper.invariant``: single static file, no year/month needed.
    - ``e5.oper.an.sfc``: one file per variable per month, needs year/month.
    - ``e5.oper.fc.sfc.accumu``: forecast half-file, needs a pre-built *url*
      (because callers handle half-file URL construction and deduplication).

    Downloads are cached by deterministic filename in *tmpdir*.  If a cached
    file already exists and is non-empty, the download is skipped.  This makes
    interrupted runs resumable — just re-run with the same tmpdir.

    Returns the path to the temp file containing a single variable ``"data"``.
    """
    # Check cache first
    cache_name = _cache_key(short_name, x0, y0, x1, y1, year, month, url)
    path = os.path.join(tmpdir, cache_name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        logger.info("era5-ncar: cache hit for %s", cache_name)
        return path

    product_dir, param_code, ncar_var = VAR_MAP[short_name]
    _retrieve_var_inner(
        short_name, x0, y0, x1, y1, tmpdir, path, product_dir, param_code, ncar_var,
        year=year, month=month, url=url,
    )
    return path


@retry(
    wait=wait_random_exponential(multiplier=1, min=2, max=120),
    stop=stop_after_attempt(8),
    retry=retry_if_exception_type(
        (requests.exceptions.RequestException, OSError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _retrieve_var_inner(
    short_name, x0, y0, x1, y1, tmpdir, path, product_dir, param_code, ncar_var,
    year=None, month=None, url=None,
):
    """Download + write, with retries. Called by _retrieve_var after cache check."""
    logger.debug(
        "era5-ncar: fetching %s year=%s month=%s url=%s",
        short_name, year, month, url,
    )

    if product_dir == "e5.oper.invariant":
        with _open_opendap(_INVARIANT_PATH) as ds:
            subset = _sel_bbox(ds, x0, y0, x1, y1)
            z = subset["Z"].isel(time=0, drop=True).load()
        da = _to_xy(z / 9.80665)
    elif product_dir == "e5.oper.an.sfc":
        an_url = _an_sfc_url(product_dir, param_code, year, month)
        with _open_opendap(an_url) as ds:
            subset = _sel_bbox(ds, x0, y0, x1, y1)
            da = _to_xy(subset[ncar_var].load())
    else:
        # e5.oper.fc.sfc.accumu — url must be provided by caller
        with _open_opendap(url) as ds:
            subset = _sel_bbox(ds, x0, y0, x1, y1)
            da = _fc_to_hourly(subset, ncar_var)

    # Write to a temp file first, then atomically rename to cache path.
    # This avoids leaving a partial file if the process is interrupted.
    fd, tmp_path = tempfile.mkstemp(suffix=".nc.tmp", dir=tmpdir)
    os.close(fd)
    try:
        da.to_dataset(name="data").to_netcdf(tmp_path)
        os.rename(tmp_path, path)
    except BaseException:
        # Clean up partial file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
    """Fetch variables with parallel downloads, then consolidate to target grid.

    Architecture:
      1. **Download** — all variables' tasks submitted to the module-level
         thread pool in parallel (I/O bound, full concurrency).  Each task
         writes a raw temp file at native resolution, cached by deterministic
         filename so interrupted runs are resumable.
      2. **Assemble & consolidate** — for each variable, open raw temp files
         lazily, concat months, interpolate/sel to target grid, then eagerly
         compute and write one consolidated NetCDF per variable at target
         resolution.  This collapses ~222 raw files into ~13 consolidated
         files, giving ``cutout_prepare`` a simple dask graph for its final
         write.
      3. **Return** — lazy dask-backed DataArrays backed by consolidated
         files.  Peak RAM ≈ one dask chunk (24 hours × target grid) at a time.

    All temp files (raw + consolidated) are cleaned up by
    ``maybe_remove_tmpdir`` after ``cutout_prepare`` finishes writing.

    Returns a dict mapping short_name → lazy DataArray on the cutout's
    (time, y, x) grid (or (y, x) for invariant height).
    """
    x0, y0, x1, y1 = _bbox(coords)
    months = _months(coords)
    t = pd.DatetimeIndex(coords["time"].values)

    # ------------------------------------------------------------------
    # File-level debug logging (written to tmpdir/era5_ncar.log).
    # ------------------------------------------------------------------
    # Attach a file handler once (first call); subsequent calls reuse it.
    _already = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "_era5_ncar_logfile", False)
        for h in logger.handlers
    )
    if tmpdir is not None and not _already:
        log_path = os.path.join(tmpdir, "era5_ncar.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(threadName)s %(levelname)s %(message)s")
        )
        file_handler._era5_ncar_logfile = True
        logger.addHandler(file_handler)
        # Lower the logger gate so DEBUG reaches the file handler.  Console
        # handlers keep their own level filters, so this won't spam stdout.
        if logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)
        logger.debug("era5-ncar: logfile %s, bbox=(%.3f,%.3f,%.3f,%.3f), months=%s",
                      log_path, x0, y0, x1, y1, months)

    # ------------------------------------------------------------------
    # Phase 1: Submit all download tasks in parallel.
    # ------------------------------------------------------------------
    # Invariant/analysis futures: keyed by (sn,) or (sn, year, month).
    # Forecast futures: keyed by URL for deduplication (prev-half of
    # month N+1 == second-half of month N).
    inv_futures = {}   # sn → Future
    an_futures = {}    # (sn, year, month) → Future
    fc_futures = {}    # url → Future
    fc_urls = {}       # (sn, year, month) → [url, ...]
    all_futures = {}   # Future → label (for progress logging)

    for sn in short_names:
        product_dir, param_code, ncar_var = VAR_MAP[sn]

        if product_dir == "e5.oper.invariant":
            f = _pool.submit(_retrieve_var, sn, x0, y0, x1, y1, tmpdir)
            inv_futures[sn] = f
            all_futures[f] = sn

        elif product_dir == "e5.oper.an.sfc":
            for year, month in months:
                f = _pool.submit(
                    _retrieve_var, sn, x0, y0, x1, y1, tmpdir,
                    year=year, month=month,
                )
                an_futures[(sn, year, month)] = f
                all_futures[f] = f"{sn} {year}-{month:02d}"

        else:
            for year, month in months:
                urls = [
                    _fc_prev_half_url(product_dir, param_code, year, month),
                    *_fc_half_urls(product_dir, param_code, year, month),
                ]
                fc_urls[(sn, year, month)] = urls
                for url in urls:
                    if url not in fc_futures:
                        f = _pool.submit(
                            _retrieve_var, sn, x0, y0, x1, y1, tmpdir,
                            url=url,
                        )
                        fc_futures[url] = f
                        all_futures[f] = f"{sn} {year}-{month:02d}"

    logger.info(
        "era5-ncar: submitting %d download tasks for [%s] with %d workers",
        len(all_futures), ", ".join(short_names), MAX_WORKERS,
    )
    t_start = time.time()

    try:
        for n_done, future in enumerate(as_completed(all_futures), 1):
            try:
                future.result()
            except Exception:
                label = all_futures[future]
                logger.error(
                    "era5-ncar: FAILED %s after retries:\n%s",
                    label, traceback.format_exc(),
                )
                raise
            elapsed = time.time() - t_start
            eta = (elapsed / n_done) * (len(all_futures) - n_done)
            logger.info(
                "era5-ncar: [%d/%d] %s  (%.0fs elapsed, ETA %.0fs)",
                n_done, len(all_futures), all_futures[future], elapsed, eta,
            )
    except BaseException:
        for f in all_futures:
            f.cancel()
        raise

    # ------------------------------------------------------------------
    # Phase 2: Consolidate per-variable — merge many small temp files into
    # one file per variable.  This collapses ~222 dask graph nodes into ~13,
    # matching the simplicity of the ERA5/CDS pipeline and dramatically
    # speeding up the final to_netcdf() write in cutout_prepare.
    #
    # The consolidation writes uncompressed at target resolution.  Raw
    # temp files cover a padded bbox and span full months; consolidated
    # files are trimmed to the exact cutout grid and time range.  The
    # trade-off is extra I/O now in exchange for a simple, fast final write.
    # ------------------------------------------------------------------
    assembled = {}

    # The entire consolidation phase must be serialized because the netCDF4/HDF5
    # C library is not thread-safe — concurrent open_dataset() calls from
    # multiple dask threads cause "double free" crashes.  Downloads (Phase 1)
    # remain fully parallel since they use pydap (pure Python, thread-safe).
    with _nc_write_lock:
        for sn in short_names:
            # Check consolidated cache first (before opening raw files).
            consolidated_path = os.path.join(tmpdir, f"consolidated_{sn}.nc")
            if os.path.exists(consolidated_path) and os.path.getsize(consolidated_path) > 0:
                logger.info("era5-ncar: consolidated cache hit for %s", sn)
                da_out = xr.open_dataset(consolidated_path, chunks={"time": 24})["data"]
                da_out.encoding.clear()
                assembled[sn] = da_out
                continue

            product_dir = VAR_MAP[sn][0]
            raw_datasets = []  # track for cleanup

            if product_dir == "e5.oper.invariant":
                path = inv_futures[sn].result()
                raw_ds = xr.open_dataset(path, chunks={})
                raw_datasets.append(raw_ds)
                da = raw_ds["data"]

            elif product_dir == "e5.oper.an.sfc":
                parts = []
                for year, month in months:
                    path = an_futures[(sn, year, month)].result()
                    raw_ds = xr.open_dataset(path, chunks={"time": 24})
                    raw_datasets.append(raw_ds)
                    chunk = raw_ds["data"]
                    mt = t[(t.year == year) & (t.month == month)]
                    if len(mt):
                        parts.append(
                            chunk.sel(
                                time=mt, method="nearest", tolerance=pd.Timedelta("30min")
                            ).assign_coords(time=mt.values)
                        )
                da = xr.concat(parts, dim="time")

            else:
                # fc.sfc.accumu
                month_parts = []
                for year, month in months:
                    paths = [fc_futures[url].result() for url in fc_urls[(sn, year, month)]]
                    chunks = []
                    for p in paths:
                        raw_ds = xr.open_dataset(p, chunks={"time": 24})
                        raw_datasets.append(raw_ds)
                        chunks.append(raw_ds["data"])
                    hourly = xr.concat(chunks, dim="time")
                    _, idx = np.unique(hourly.time.values, return_index=True)
                    hourly = hourly.isel(time=idx)
                    if sn in _FC_DIVIDE_BY_3600:
                        hourly = hourly / 3600.0
                    hourly = _to_xy(hourly)
                    mt = t[(t.year == year) & (t.month == month)]
                    if len(mt):
                        month_parts.append(
                            hourly.sel(
                                time=mt, method="nearest", tolerance=pd.Timedelta("30min")
                            ).assign_coords(time=mt.values)
                        )
                da = xr.concat(month_parts, dim="time")

            # Interpolate to target grid.
            ds_one = xr.Dataset({sn: da})
            if _grids_align(ds_one, coords):
                result = ds_one.sel(
                    x=coords["x"].values, y=coords["y"].values, method="nearest"
                )[sn]
            else:
                result = ds_one.interp(
                    x=coords["x"].values, y=coords["y"].values, method="linear"
                )[sn]
            result.encoding.clear()

            # Consolidate: eagerly compute this variable and write one clean
            # uncompressed NetCDF at target resolution.
            logger.info("era5-ncar: consolidating %s to %s", sn, consolidated_path)
            to_write = xr.Dataset({"data": result})
            for v in to_write.data_vars:
                to_write[v].encoding.clear()
            tmp_consolidated = consolidated_path + ".tmp"
            to_write.compute(scheduler="synchronous").to_netcdf(tmp_consolidated)

            # Close raw temp file handles now that data is materialized.
            for raw_ds in raw_datasets:
                raw_ds.close()

            os.rename(tmp_consolidated, consolidated_path)
            da_out = xr.open_dataset(consolidated_path, chunks={"time": 24})["data"]
            da_out.encoding.clear()
            assembled[sn] = da_out

    return assembled


# ---------------------------------------------------------------------------
# Feature assemblers  (receive pre-fetched vars dict, compute derived fields)
# ---------------------------------------------------------------------------


def get_data_wind(coords, tmpdir=None):
    v = _fetch_vars(_FEATURE_VARS["wind"], coords, tmpdir=tmpdir)
    wnd10m = np.sqrt(v["u10"] ** 2 + v["v10"] ** 2)
    wnd100m = np.sqrt(v["u100"] ** 2 + v["v100"] ** 2)
    wnd_shear_exp = (np.log(wnd10m / wnd100m) / np.log(10.0 / 100.0)).assign_attrs(
        units="", long_name="wind shear exponent"
    )
    az = np.arctan2(v["u100"], v["v100"])  # stays lazy via __array_ufunc__
    wnd_azimuth = az.where(az >= 0, az + 2 * np.pi)
    return xr.Dataset(
        {
            "wnd100m": wnd100m.rename("wnd100m"),
            # np.log promotes float32→float64; cast back to match input precision
            "wnd_shear_exp": wnd_shear_exp.astype("float32").rename("wnd_shear_exp"),
            "wnd_azimuth": wnd_azimuth.rename("wnd_azimuth"),
            "roughness": v["fsr"].rename("roughness"),
        }
    )


def get_data_influx(coords, tmpdir=None):
    v = _fetch_vars(_FEATURE_VARS["influx"], coords, tmpdir=tmpdir)
    ssrd, ssr, fdir, tisr = v["ssrd"], v["ssr"], v["fdir"], v["tisr"]
    albedo = ((ssrd - ssr) / ssrd.where(ssrd != 0)).fillna(0.0)
    influx_diffuse = ssrd - fdir
    ds = xr.Dataset(
        {
            "influx_direct": fdir.rename("influx_direct"),
            "influx_diffuse": influx_diffuse.rename("influx_diffuse"),
            "influx_toa": tisr.rename("influx_toa"),
            "albedo": albedo.rename("albedo"),
        }
    )
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift=pd.to_timedelta("-30 minutes"))
    sp = sp.rename({name: f"solar_{name}" for name in sp.data_vars})
    # SolarPosition trig promotes float32→float64; cast back to save disk/RAM
    sp = sp.astype("float32")
    return xr.merge([ds, sp])


def get_data_temperature(coords, tmpdir=None):
    v = _fetch_vars(_FEATURE_VARS["temperature"], coords, tmpdir=tmpdir)
    return xr.Dataset(
        {
            "temperature": v["t2m"].rename("temperature"),
            "soil temperature": v["stl4"].rename("soil temperature"),
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
