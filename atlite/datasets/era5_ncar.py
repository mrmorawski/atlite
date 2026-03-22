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

MAX_WORKERS = 16  # concurrent OPeNDAP requests

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

def _open_opendap(path):
    """Open an OPeNDAP dataset (lazy — metadata only).

    Uses ``engine="pydap"`` because netCDF4 is typically not compiled with
    OPeNDAP support.  The pydap DAP2 deprecation warning is suppressed.
    """
    url = OPENDAP_BASE + path
    logger.debug("era5-ncar OPeNDAP: %s", url)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xr.open_dataset(url, engine="pydap")


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
    ).sortby("time")


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


@retry(
    wait=wait_random_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _retrieve_var(short_name, year, month, x0, y0, x1, y1):
    """Fetch one variable for one month at native 0.25° resolution.

    Returns a (time, y, x) DataArray, or (y, x) for the invariant height var.
    Retried up to 5 times with random exponential backoff (2–60 s) on any
    error, so transient THREDDS 500s and timeouts are handled automatically.
    """
    product_dir, param_code, ncar_var = VAR_MAP[short_name]

    if product_dir == "e5.oper.invariant":
        ds = _open_opendap(_INVARIANT_PATH)
        subset = _sel_bbox(ds, x0, y0, x1, y1)
        z = subset["Z"].isel(time=0, drop=True).load()
        ds.close()
        return _to_xy(z / 9.80665)

    if product_dir == "e5.oper.an.sfc":
        path = _an_sfc_url(product_dir, param_code, year, month)
        ds = _open_opendap(path)
        subset = _sel_bbox(ds, x0, y0, x1, y1)
        da = _to_xy(subset[ncar_var].load())
        ds.close()
        return da

    # fc.sfc.accumu — fetch prev half + both halves of the month
    urls = [
        _fc_prev_half_url(product_dir, param_code, year, month),
        *_fc_half_urls(product_dir, param_code, year, month),
    ]
    parts = []
    for url in urls:
        ds = _open_opendap(url)
        subset = _sel_bbox(ds, x0, y0, x1, y1)
        parts.append(_fc_to_hourly(subset, ncar_var))
        ds.close()
    hourly = xr.concat(parts, dim="time").sortby("time")
    _, idx = np.unique(hourly.time.values, return_index=True)
    hourly = hourly.isel(time=idx)
    if short_name in _FC_DIVIDE_BY_3600:
        hourly = hourly / 3600.0
    return _to_xy(hourly)


def _fetch_vars(short_names, coords):
    """Fetch all requested variables concurrently.

    Submits one task per (short_name, year, month) combination to a thread
    pool, concatenates monthly results, and interpolates to the cutout grid.

    Returns a dict mapping short_name → DataArray on the cutout's (time, y, x)
    grid (or (y, x) for the invariant height variable).
    """
    x0, y0, x1, y1 = _bbox(coords)
    months = _months(coords)
    t = pd.DatetimeIndex(coords["time"].values)

    # Build task list; invariant vars need only one fetch (no month loop).
    tasks = []
    for sn in short_names:
        if VAR_MAP[sn][0] == "e5.oper.invariant":
            tasks.append((sn, None, None))
        else:
            for year, month in months:
                tasks.append((sn, year, month))

    n_total = len(tasks)
    logger.info(
        "era5-ncar: submitting %d tasks for [%s] with %d workers",
        n_total, ", ".join(short_names), MAX_WORKERS,
    )
    t_start = time.time()

    raw = {}  # (short_name, year, month) → DataArray
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_retrieve_var, sn, yr, mo, x0, y0, x1, y1): (sn, yr, mo)
            for sn, yr, mo in tasks
        }
        for n_done, future in enumerate(as_completed(futures), 1):
            sn, yr, mo = futures[future]
            elapsed = time.time() - t_start
            rate = elapsed / n_done
            eta = rate * (n_total - n_done)
            logger.info(
                "era5-ncar: [%d/%d] %-6s %s-%02d  (%.0fs elapsed, ETA %.0fs)",
                n_done, n_total, sn, yr if yr else "—", mo or 0, elapsed, eta,
            )
            raw[(sn, yr, mo)] = future.result()

    # Assemble per-variable: select target times, concatenate months, interpolate.
    assembled = {}
    for sn in short_names:
        if VAR_MAP[sn][0] == "e5.oper.invariant":
            assembled[sn] = _interp(raw[(sn, None, None)], coords)
        else:
            parts = []
            for year, month in months:
                da = raw[(sn, year, month)]
                mt = t[(t.year == year) & (t.month == month)]
                if len(mt):
                    sel = da.sel(time=mt, method="nearest").assign_coords(time=mt.values)
                    parts.append(sel)
            assembled[sn] = _interp(xr.concat(parts, dim="time"), coords)

    return assembled


def _interp(da, coords):
    """Bilinearly interpolate *da* (y, x) to the cutout's target grid."""
    return da.interp(
        x=coords["x"].values,
        y=coords["y"].values,
        method="linear",
    )


# ---------------------------------------------------------------------------
# Feature assemblers  (receive pre-fetched vars dict, compute derived fields)
# ---------------------------------------------------------------------------

def get_data_wind(coords):
    v = _fetch_vars(_FEATURE_VARS["wind"], coords)
    wnd10m  = np.sqrt(v["u10"]**2  + v["v10"]**2)
    wnd100m = np.sqrt(v["u100"]**2 + v["v100"]**2)
    wnd_shear_exp = (
        np.log(wnd10m / wnd100m) / np.log(10.0 / 100.0)
    ).assign_attrs(units="", long_name="wind shear exponent")
    az = np.arctan2(v["u100"].values, v["v100"].values)
    wnd_azimuth = xr.DataArray(
        np.where(az >= 0, az, az + 2 * np.pi),
        dims=wnd100m.dims,
        coords=wnd100m.coords,
    )
    return xr.Dataset(
        {
            "wnd100m":       wnd100m.rename("wnd100m"),
            "wnd_shear_exp": wnd_shear_exp.rename("wnd_shear_exp"),
            "wnd_azimuth":   wnd_azimuth.rename("wnd_azimuth"),
            "roughness":     v["fsr"].rename("roughness"),
        }
    )


def get_data_influx(coords):
    v = _fetch_vars(_FEATURE_VARS["influx"], coords)
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


def get_data_temperature(coords):
    v = _fetch_vars(_FEATURE_VARS["temperature"], coords)
    return xr.Dataset(
        {
            "temperature":          v["t2m"].rename("temperature"),
            "soil temperature":     v["stl4"].rename("soil temperature"),
            "dewpoint temperature": v["d2m"].rename("dewpoint temperature"),
        }
    )


def get_data_runoff(coords):
    v = _fetch_vars(_FEATURE_VARS["runoff"], coords)
    return xr.Dataset({"runoff": v["ro"].rename("runoff")})


def get_data_height(coords):
    v = _fetch_vars(_FEATURE_VARS["height"], coords)
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
    ds = func(coords)

    if sanitize and sanitize_func is not None:
        ds = sanitize_func(ds)

    if feature in static_features:
        return ds.squeeze()
    return ds
