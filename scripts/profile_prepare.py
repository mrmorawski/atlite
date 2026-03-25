"""
Profile the cutout.prepare() pipeline with pre-cached consolidated files.

Measures:
  1. get_features() — loading consolidated files + computing derived fields
  2. Compression encoding overhead (zlib complevel=9 vs lower levels)
  3. Final to_netcdf() write (the main bottleneck suspected)

Usage:
    python scripts/profile_prepare.py
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
logger = logging.getLogger("profile_prepare")

TMPDIR = "./tmp"
OUTPUT = "europe_2013_ncar_profile.nc"
BOUNDS = (-25.0, 34.0, 45.0, 72.0)

ALL_FEATURES = ["height", "wind", "influx", "temperature", "runoff"]


def time_block(label):
    """Context manager to time a block and print results."""
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

    # ---- Step 1: Create cutout object (no I/O) ----
    output_path = Path(OUTPUT)
    if output_path.exists():
        output_path.unlink()

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

    # ---- Step 2: Time get_features (loads consolidated caches + derived fields) ----
    from atlite.data import get_features, available_features, non_bool_dict
    from numpy import atleast_1d
    from dask.utils import SerializableLock

    modules = atleast_1d(cutout.module)
    features_list = atleast_1d(ALL_FEATURES)
    target = available_features(modules).loc[:, features_list].drop_duplicates()

    with time_block("get_features (load consolidated + compute derived)"):
        ds = get_features(cutout, modules[0], target[modules[0]].index.unique("feature"),
                          tmpdir=TMPDIR, data_format="grib")

    missing_vars = target[modules[0]].values
    logger.info("Variables: %s", list(ds.data_vars))
    logger.info("Dataset size: %.2f GB (uncompressed float32 estimate)",
                sum(ds[v].size * 4 for v in ds.data_vars) / 1e9)

    # Check chunking
    for v in list(ds.data_vars)[:3]:
        da = ds[v]
        logger.info("  %s: shape=%s, chunks=%s, dtype=%s",
                     v, da.shape, da.chunks if da.chunks else "None", da.dtype)

    # ---- Step 3: Profile compression at different levels ----
    # Pick one representative variable to benchmark compression
    test_var = "temperature"
    logger.info("=== Compression benchmarks on '%s' ===", test_var)

    da_test = ds[test_var]
    # Load a chunk into memory for compression testing
    with time_block(f"load {test_var} into memory"):
        data_mem = da_test.compute()

    ds_test = xr.Dataset({test_var: data_mem})
    data_bytes = data_mem.nbytes
    logger.info("  %s in-memory size: %.2f GB", test_var, data_bytes / 1e9)

    for complevel in [0, 1, 4, 9]:
        if complevel == 0:
            enc = {}
            label = "no compression"
        else:
            enc = {"zlib": True, "complevel": complevel, "shuffle": True}
            label = f"zlib complevel={complevel} shuffle=True"

        ds_test[test_var].encoding.update(enc)

        fd, tmp = mkstemp(suffix=".nc", dir=".")
        os.close(fd)
        try:
            with time_block(f"write {test_var} ({label})"):
                ds_test.to_netcdf(tmp)
            fsize = os.path.getsize(tmp)
            ratio = data_bytes / fsize if fsize > 0 else 0
            logger.info("  => file size: %.2f GB, compression ratio: %.1fx", fsize / 1e9, ratio)
        finally:
            os.unlink(tmp)
        ds_test[test_var].encoding.clear()

    # ---- Step 4: Profile full write at complevel=9 (current default) ----
    # Set up compression like cutout_prepare does
    compression = {"zlib": True, "complevel": 9, "shuffle": True}
    prepared = set()
    prepared |= set(target[modules[0]].index.unique("feature"))
    cutout.data.attrs.update(dict(prepared_features=list(prepared)))
    attrs = non_bool_dict(cutout.data.attrs)
    attrs.update(ds.attrs)

    for v in missing_vars:
        ds[v].encoding.update(compression)

    ds_full = cutout.data.merge(ds[missing_vars]).assign_attrs(**attrs)

    # Profile the delayed write
    logger.info("=== Full dataset write (complevel=9) ===")
    fd, tmp = mkstemp(suffix=".nc", dir=".")
    os.close(fd)

    with time_block("to_netcdf (compute=False) — build dask graph"):
        write_job = ds_full.to_netcdf(tmp, compute=False)

    with time_block("write_job.compute() — actual write to disk"):
        write_job.compute()

    fsize = os.path.getsize(tmp)
    logger.info("Final file size: %.2f GB", fsize / 1e9)
    os.unlink(tmp)

    # ---- Step 5: Profile full write at complevel=1 for comparison ----
    logger.info("=== Full dataset write (complevel=1) ===")
    for v in missing_vars:
        ds[v].encoding.update({"zlib": True, "complevel": 1, "shuffle": True})
    ds_full2 = cutout.data.merge(ds[missing_vars]).assign_attrs(**attrs)

    fd, tmp = mkstemp(suffix=".nc", dir=".")
    os.close(fd)
    with time_block("write_job.compute() — complevel=1"):
        write_job2 = ds_full2.to_netcdf(tmp, compute=False)
        write_job2.compute()
    fsize2 = os.path.getsize(tmp)
    logger.info("Final file size (complevel=1): %.2f GB", fsize2 / 1e9)
    os.unlink(tmp)

    # ---- Step 6: Profile full write uncompressed ----
    logger.info("=== Full dataset write (no compression) ===")
    for v in missing_vars:
        ds[v].encoding.clear()
    ds_full3 = cutout.data.merge(ds[missing_vars]).assign_attrs(**attrs)

    fd, tmp = mkstemp(suffix=".nc", dir=".")
    os.close(fd)
    with time_block("write_job.compute() — no compression"):
        write_job3 = ds_full3.to_netcdf(tmp, compute=False)
        write_job3.compute()
    fsize3 = os.path.getsize(tmp)
    logger.info("Final file size (no compression): %.2f GB", fsize3 / 1e9)
    os.unlink(tmp)

    # ---- Step 7: Profile the consolidation phase (re-consolidate one variable) ----
    # Delete one consolidated file and re-run to time it
    logger.info("=== Consolidation phase benchmark (single variable: u10) ===")
    consol_path = os.path.join(TMPDIR, "consolidated_u10.nc")
    backup_path = consol_path + ".bak"
    if os.path.exists(consol_path):
        os.rename(consol_path, backup_path)
    try:
        from atlite.datasets.era5_ncar import _fetch_vars
        with time_block("consolidate u10 from raw files"):
            result = _fetch_vars(["u10"], cutout.coords, tmpdir=TMPDIR)
        logger.info("  u10 result: shape=%s, chunks=%s", result["u10"].shape, result["u10"].chunks)
    finally:
        # Restore backup
        if os.path.exists(backup_path):
            if os.path.exists(consol_path):
                os.unlink(consol_path)
            os.rename(backup_path, consol_path)

    # ---- Summary ----
    print()
    print("=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
