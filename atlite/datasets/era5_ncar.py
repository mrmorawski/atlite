# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and processing data from ECMWFs ERA5 dataset (via NSF NCAR)

NSF NCAR hosts a mirror of ERA5 on their Research Data Archive (RDA, dataset d633000).
However, unlike the original source of ERA5 (the Copernicus Data Store), they do not
require authentication and do not have a download queue.

This module mirrors processing from atlite/datasets/era5.py, and should deliver identical results.
To use it, or to replace era5.py in old code, simply replace `module="era5"` with
`module="era5-ncar"` in `atlite.Cutout` like this:

```
cutout = atlite.Cutout(
    path="my_cutout.nc",
    module="era5-ncar",
    x=slice(-10, 5),
    y=slice(35, 44),
    time="2024",
)
cutout.prepare()
```

Caveats:
- the data has some delay vs. ERA5, ERA5T, so it is not a best choice for very recent data
- data is downloaded uncompressed and unprocessed, so this module requires ~1.5x the bandwith and disk space of era5.py
"""

import calendar
import logging
import tempfile
import datetime
import hashlib
import warnings
import xarray as xr
import pandas as pd
import numpy as np
import os
from pathlib import Path
from atlite.datasets.era5 import sanitize_influx, sanitize_runoff, sanitize_wind

# Logging setup
logger = logging.getLogger(__name__)
# logging.getLogger("pydap").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.ERROR)

# Model and CRS Settings
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


class ERA5NCARdownloader:
    """
    TODO:
    squeeze if static!
    tmpdir cleanup
    check if xr has a round function
    look into locking session
    """

    def __init__(
        self,
        cutout,
        feature: str,
        sanitize: bool = False,
        tmpdir: str | Path | None = None,
    ) -> None:
        if feature not in features.keys():
            raise ValueError(
                f"era5-ncar: no retrieval function for feature {feature!r}. "
                f"Available: {list(features)}"
            )

        self.cutout = cutout
        self.feature = feature
        self.sanitize = sanitize
        self.tmpdir = self._get_tmpdir(tmpdir)
        # determine which months are needed in the cutout (used for download logic)
        t = pd.DatetimeIndex(cutout.coords["time"].values)
        self.years_months = sorted(set(zip(t.year, t.month)))
        self.cutout_boundary = self._bbox(cutout.coords)
        # map atlite features to ERA5 variables
        self.atlite_to_era5_map = {
            "wind": ("u10", "v10", "u100", "v100", "fsr"),
            "influx": ("ssrd", "ssr", "fdir", "tisr"),
            "temperature": ("t2m", "stl4", "d2m"),
            "runoff": ("ro"),
            "height": ("z"),
        }

        # map ERA5 variables to NCAR product dir, parameter code, variable name
        self.era5_to_ncar_map = {
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

    @staticmethod
    def _bbox(coords):
        bbox_pad = 0.5
        """pad bounding box coordinates, to make sure we have all the data"""
        return (
            float(coords["x"].min()) - bbox_pad,
            float(coords["y"].min()) - bbox_pad,
            float(coords["x"].max()) + bbox_pad,
            float(coords["y"].max()) + bbox_pad,
        )

    @staticmethod
    def _get_tmpdir(tmpdir: str | Path | None) -> Path:
        """
        If the user doesn't provide a tmpdir, atlite creates one in /tmp, which can be too small for storing bit cutouts.
        If this is the case, we need to create a separate tmpdir for storing downloads. We make a directory in .cache for this.
        """
        # check if provided path is from tempfile or from the user
        if tmpdir is not None:
            system_tmp = Path(tempfile.gettempdir()).resolve()
            tmpdir_resolved = Path(tmpdir).resolve()
            if not tmpdir_resolved.is_relative_to(system_tmp):
                tmpdir_path = Path(tmpdir)
        # if not, generate
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_root = Path(".cache")
        cache_root.mkdir(exist_ok=True)
        tmpdir_path = Path(tempfile.mkdtemp(prefix=f"era5_ncar_{ts}_", dir=cache_root))
        logger.info("era5-ncar: no user tmpdir; downloading to %s", tmpdir_path)

        return tmpdir_path

    @staticmethod
    def _open_opendap(path):
        """
        Open an OPeNDAP dataset

        Uses ``engine="pydap"`` because netCDF4 is typically not compiled with
        OPeNDAP support.  The pydap DAP2 deprecation warning is suppressed.
        Transient errors are retried by the calling functions.
        """
        opendap_base = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/"
        url = opendap_base + path
        logger.debug("era5-ncar OPeNDAP: %s", url)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return xr.open_dataset(url, engine="pydap")

    @staticmethod
    def _sel_bbox(ds, x0, y0, x1, y1):
        """
                Subset *ds* to (x0, y0, x1, y1) handling 0–360 longitudes.

        look into locking session
                Serves two functions. Firstly, maps the way that atlite handles coordinates to
                the way that the NCAR dataset is structured.

                Secondly, chooses a geogrphic subset of the data.The NCAR THREDDS server handles geographic subsetting and serves only the required
                geographic slices of the data. Howver, there is an issue when the boundary box crosses
                the crosses the 0° meridian (e.g. x0=-4, x1=1.5). Then the data must be split into two
                downloads and concatenated.
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

    @staticmethod
    def _to_xy(da):
        """Rename latitude/longitude → y/x, sort S→N to match atlite convention, round coords to 5 d.p."""
        da = da.rename({"latitude": "y", "longitude": "x"}).sortby("y")
        return da.assign_coords(
            x=np.round(da.x.values.astype(float), 5),
            y=np.round(da.y.values.astype(float), 5),
        )

    @staticmethod
    def _an_sfc_url(product_dir, param_code, year, month):
        ym = f"{year}{month:02d}"
        last = calendar.monthrange(year, month)[1]
        fname = f"{product_dir}.{param_code}.ll025sc.{ym}0100_{ym}{last:02d}23.nc"
        return f"{product_dir}/{ym}/{fname}"

    @staticmethod
    def _cache_key(x0, y0, x1, y1, url):
        """Return a deterministic filename for caching a raw download."""
        bbox_str = f"{x0:.3f}_{y0:.3f}_{x1:.3f}_{y1:.3f}"
        combined = f"{bbox_str}_{url}"
        key_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
        return f"era5_ncar_{key_hash}.nc"

    def _get_wind(self):
        ncar_download_params = [
            self.era5_to_ncar_map.get(era5_var)
            for era5_var in self.atlite_to_era5_map["wind"]
        ]
        for year, month in self.years_months:
            logger.info(
                f"downloading wind for {year}-{month} in {ncar_download_params}"
            )
            for product_code, param_code, ncar_var in ncar_download_params:
                # check if file exists in cache. if not - download
                print(product_code, param_code, ncar_var)
                an_url = self._an_sfc_url(product_code, param_code, year, month)
                cache_name = self._cache_key(*self.cutout_boundary, an_url)
                file_path = Path(self.tmpdir) / cache_name
                if file_path.exists() and file_path.stat().st_size > 0:
                    logger.info("era5-ncar: cache hit for %s", cache_name)
                else:
                    with self._open_opendap(an_url) as ds:
                        subset = self._sel_bbox(ds, *self.cutout_boundary)
                        da = self._to_xy(subset[ncar_var].load())
                    # Write to a temp file first, then atomically rename to cache path.
                    # This avoids leaving a partial file if the process is interrupted.
                    fd, tmp_path = tempfile.mkstemp(suffix=".nc.tmp", dir=self.tmpdir)
                    tmp_path = Path(tmp_path)
                    os.close(fd)
                    try:
                        da.to_dataset(name="data").to_netcdf(tmp_path)
                        tmp_path.rename(file_path)
                    except BaseException:
                        # Clean up partial file on failure
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                        raise

                return xr.open_dataset(file_path, chunks={"time": 720})

    def get_ds(self):
        ds = self._get_wind()

        sanitize_map = {
            "influx": sanitize_influx,
            "runoff": sanitize_runoff,
            "wind": sanitize_wind,
        }
        if self.sanitize:
            ds = sanitize_map[self.feature](ds)

        raise NotImplementedError


def get_data(cutout, feature: str, tmpdir=None, **creation_parameters):
    """
    Retrieve ERA5 data from NCAR THREDDS/OPeNDAP.

    Similar interface to ``atlite.datasets.era5.get_data()``.
    """

    logger.info("era5-ncar: fetching feature '%s'...", feature)
    downloader = ERA5NCARdownloader(
        cutout, feature, creation_parameters.get("sanitize", True)
    )

    return downloader.get_ds()
