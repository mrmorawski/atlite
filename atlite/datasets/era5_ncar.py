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
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from atlite.datasets.era5 import sanitize_influx, sanitize_runoff, sanitize_wind
from atlite.pv.solar_position import SolarPosition

logger = logging.getLogger(__name__)

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
# Per-month retrieval
# ---------------------------------------------------------------------------

def _retrieve_analysis(short_name, year, month, x0, y0, x1, y1):
    """Return (time, y, x) DataArray at native 0.25° for one an.sfc month."""
    product_dir, param_code, ncar_var = VAR_MAP[short_name]
    path = _an_sfc_url(product_dir, param_code, year, month)
    ds = _open_opendap(path)
    subset = _sel_bbox(ds, x0, y0, x1, y1)
    da = _to_xy(subset[ncar_var].load())
    ds.close()
    return da


def _retrieve_forecast_month(short_name, year, month, x0, y0, x1, y1,
                              divide_by_3600=False):
    """Return (time, y, x) DataArray for one fc.sfc.accumu variable and month.

    Fetches the previous month's second-half file as well, which supplies
    hours 00:00–06:00 of the first day of the month.
    """
    product_dir, param_code, ncar_var = VAR_MAP[short_name]
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

    if divide_by_3600:
        hourly = hourly / 3600.0

    return _to_xy(hourly)


def _retrieve_height(x0, y0, x1, y1):
    """Return (y, x) geopotential height [m]."""
    ds = _open_opendap(_INVARIANT_PATH)
    subset = _sel_bbox(ds, x0, y0, x1, y1)
    z = subset["Z"].isel(time=0, drop=True).load()
    ds.close()
    return _to_xy(z / 9.80665)


# ---------------------------------------------------------------------------
# Multi-month collection + interpolation helpers
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


def _collect_analysis(short_name, coords, x0, y0, x1, y1):
    t = pd.DatetimeIndex(coords["time"].values)
    parts = []
    for year, month in _months(coords):
        da = _retrieve_analysis(short_name, year, month, x0, y0, x1, y1)
        mt = t[(t.year == year) & (t.month == month)]
        if len(mt):
            sel = da.sel(time=mt, method="nearest").assign_coords(time=mt.values)
            parts.append(sel)
    return xr.concat(parts, dim="time")


def _collect_forecast(short_name, coords, x0, y0, x1, y1, divide_by_3600=False):
    t = pd.DatetimeIndex(coords["time"].values)
    parts = []
    for year, month in _months(coords):
        da = _retrieve_forecast_month(
            short_name, year, month, x0, y0, x1, y1, divide_by_3600
        )
        mt = t[(t.year == year) & (t.month == month)]
        if len(mt):
            sel = da.sel(time=mt, method="nearest").assign_coords(time=mt.values)
            parts.append(sel)
    return xr.concat(parts, dim="time")


def _interp(da, coords):
    """Bilinearly interpolate *da* (y, x) to the cutout's target grid."""
    return da.interp(
        x=coords["x"].values,
        y=coords["y"].values,
        method="linear",
    )


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def get_data_wind(coords):
    x0, y0, x1, y1 = _bbox(coords)

    u10  = _interp(_collect_analysis("u10",  coords, x0, y0, x1, y1), coords)
    v10  = _interp(_collect_analysis("v10",  coords, x0, y0, x1, y1), coords)
    u100 = _interp(_collect_analysis("u100", coords, x0, y0, x1, y1), coords)
    v100 = _interp(_collect_analysis("v100", coords, x0, y0, x1, y1), coords)
    fsr  = _interp(_collect_analysis("fsr",  coords, x0, y0, x1, y1), coords)

    wnd10m  = np.sqrt(u10**2  + v10**2)
    wnd100m = np.sqrt(u100**2 + v100**2)
    wnd_shear_exp = (
        np.log(wnd10m / wnd100m) / np.log(10.0 / 100.0)
    ).assign_attrs(units="", long_name="wind shear exponent")

    az = np.arctan2(u100.values, v100.values)
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
            "roughness":     fsr.rename("roughness"),
        }
    )


def get_data_influx(coords):
    x0, y0, x1, y1 = _bbox(coords)

    ssrd = _interp(_collect_forecast("ssrd", coords, x0, y0, x1, y1, True), coords)
    ssr  = _interp(_collect_forecast("ssr",  coords, x0, y0, x1, y1, True), coords)
    fdir = _interp(_collect_forecast("fdir", coords, x0, y0, x1, y1, True), coords)
    tisr = _interp(_collect_forecast("tisr", coords, x0, y0, x1, y1, True), coords)

    albedo        = ((ssrd - ssr) / ssrd.where(ssrd != 0)).fillna(0.0)
    influx_direct  = fdir
    influx_diffuse = ssrd - fdir
    influx_toa     = tisr

    ds = xr.Dataset(
        {
            "influx_direct":  influx_direct.rename("influx_direct"),
            "influx_diffuse": influx_diffuse.rename("influx_diffuse"),
            "influx_toa":     influx_toa.rename("influx_toa"),
            "albedo":         albedo.rename("albedo"),
        }
    )
    # SolarPosition requires lon/lat coordinates on the dataset.
    # ERA5 convention: value at T = mean irradiance for T-1h..T → shift by -30 min.
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift=pd.to_timedelta("-30 minutes"))
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})
    return xr.merge([ds, sp])


def get_data_temperature(coords):
    x0, y0, x1, y1 = _bbox(coords)
    t2m  = _interp(_collect_analysis("t2m",  coords, x0, y0, x1, y1), coords)
    stl4 = _interp(_collect_analysis("stl4", coords, x0, y0, x1, y1), coords)
    d2m  = _interp(_collect_analysis("d2m",  coords, x0, y0, x1, y1), coords)
    return xr.Dataset(
        {
            "temperature":          t2m.rename("temperature"),
            "soil temperature":     stl4.rename("soil temperature"),
            "dewpoint temperature": d2m.rename("dewpoint temperature"),
        }
    )


def get_data_runoff(coords):
    x0, y0, x1, y1 = _bbox(coords)
    ro = _interp(
        _collect_forecast("ro", coords, x0, y0, x1, y1, divide_by_3600=False),
        coords,
    )
    return xr.Dataset({"runoff": ro.rename("runoff")})


def get_data_height(coords):
    x0, y0, x1, y1 = _bbox(coords)
    height = _interp(_retrieve_height(x0, y0, x1, y1), coords)
    return xr.Dataset({"height": height.rename("height")})


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
