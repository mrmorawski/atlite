"""
Profile cutout.prepare() on Germany cutout — clean test with new settings.

Deletes consolidated caches so consolidation is re-timed.
Tests final write at complevel=1 (new default in scripts) and complevel=9 (atlite default).
"""

import logging
import os
import sys
import time
from pathlib import Path
from tempfile import mkstemp

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-24s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("profile_germany")

TMPDIR = "./tmp2"
OUTPUT = "germany_2013_profile.nc"
BOUNDS = (5.5, 47.0, 15.5, 55.5)

ALL_FEATURES = ["height", "wind", "influx", "temperature", "runoff"]


def time_block(label):
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            logger.info("START: %s", label)
            return self
        def __exit__(self, *args):
            self.elapsed = time.time() - self.t0
            logger.info("DONE:  %s — %.1f s (%.1f min)", label, self.elapsed, self.elapsed / 60)
    return Timer()


def main():
    import atlite

    os.makedirs(TMPDIR, exist_ok=True)
    output_path = Path(OUTPUT)
    if output_path.exists():
        output_path.unlink()

    # Delete old consolidated caches
    for f in Path(TMPDIR).glob("consolidated_*.nc"):
        logger.info("Removing old cache: %s", f.name)
        f.unlink()

    # ---- Full prepare with complevel=1 ----
    cutout = atlite.Cutout(
        path=output_path,
        module="era5-ncar",
        x=slice(BOUNDS[0], BOUNDS[2]),
        y=slice(BOUNDS[1], BOUNDS[3]),
        time="2013",
    )
    nx = len(cutout.coords["x"])
    ny = len(cutout.coords["y"])
    nt = len(cutout.coords["time"])
    logger.info("Grid: %d x %d, %d timesteps", nx, ny, nt)

    with time_block("cutout.prepare (complevel=1, chunks=720)"):
        cutout.prepare(
            features=ALL_FEATURES,
            tmpdir=TMPDIR,
            compression={"zlib": True, "complevel": 1, "shuffle": True},
        )

    fsize = output_path.stat().st_size
    logger.info("Output file: %.2f GB", fsize / 1e9)

    # Report consolidated sizes
    for f in sorted(Path(TMPDIR).glob("consolidated_*.nc")):
        logger.info("  %s: %.1f MB", f.name, f.stat().st_size / 1e6)

    cutout.data.close()
    output_path.unlink()

    # ---- Re-profile final write at different complevels (cache hits) ----
    cutout = atlite.Cutout(
        path=output_path,
        module="era5-ncar",
        x=slice(BOUNDS[0], BOUNDS[2]),
        y=slice(BOUNDS[1], BOUNDS[3]),
        time="2013",
    )

    from atlite.data import get_features, available_features, non_bool_dict
    from numpy import atleast_1d

    modules = atleast_1d(cutout.module)
    target = available_features(modules).loc[:, ALL_FEATURES].drop_duplicates()
    missing_vars = target[modules[0]].values

    with time_block("get_features (consolidated cache hits)"):
        ds = get_features(cutout, modules[0], target[modules[0]].index.unique("feature"),
                          tmpdir=TMPDIR, data_format="grib")

    total_size = sum(np.prod(ds[v].shape) * ds[v].dtype.itemsize for v in ds.data_vars)
    logger.info("Dataset: %.2f GB uncompressed", total_size / 1e9)

    for v in list(ds.data_vars)[:2]:
        da = ds[v]
        if da.chunks:
            logger.info("  %s: shape=%s, n_chunks=%d, chunk_time=%d",
                        v, da.shape, len(da.chunks[0]), da.chunks[0][0])

    prepared = set(target[modules[0]].index.unique("feature"))
    cutout.data.attrs.update(dict(prepared_features=list(prepared)))
    attrs = non_bool_dict(cutout.data.attrs)
    attrs.update(ds.attrs)

    results = {}
    for complevel in [1, 9]:
        for v in missing_vars:
            ds[v].encoding.clear()
            ds[v].encoding.update({"zlib": True, "complevel": complevel, "shuffle": True})

        ds_merged = cutout.data.merge(ds[missing_vars]).assign_attrs(**attrs)

        fd, tmp = mkstemp(suffix=".nc", dir=".")
        os.close(fd)
        try:
            with time_block(f"final write (complevel={complevel})") as timer:
                write_job = ds_merged.to_netcdf(tmp, compute=False)
                write_job.compute()
            fsize = os.path.getsize(tmp)
            results[complevel] = (timer.elapsed, fsize)
            logger.info("  => %.2f GB, ratio: %.1fx",
                        fsize / 1e9, total_size / fsize if fsize else 0)
        finally:
            os.unlink(tmp)

    print()
    print("=" * 70)
    print("RESULTS (Germany, 41x35, 8760h, 0.70 GB uncompressed)")
    print("=" * 70)
    print(f"  {'complevel':>10}  {'Write time':>12}  {'File size':>12}  {'Ratio':>8}")
    for cl, (elapsed, fsize) in results.items():
        ratio = total_size / fsize if fsize else 0
        print(f"  {cl:>10}  {elapsed:>10.1f} s  {fsize/1e9:>10.2f} GB  {ratio:>6.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
