"""
Focused benchmark: consolidated file compression impact.

Tests:
  A) Consolidation write speed: uncompressed vs complevel=1
  B) Final write speed reading from compressed vs uncompressed consolidated files
     (separate subprocess to avoid lock/fd issues)
"""

import logging
import os
import sys
import time
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-24s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("profile_consolidation")

TMPDIR = "./tmp2"
BOUNDS = (5.5, 47.0, 15.5, 55.5)


def time_block(label):
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            logger.info("START: %s", label)
            return self
        def __exit__(self, *args):
            self.elapsed = time.time() - self.t0
            logger.info("DONE:  %s — %.1f s", label, self.elapsed)
    return Timer()


def run_test_a():
    """Test consolidation write speed: uncompressed vs complevel=1."""
    import atlite
    from atlite.datasets import era5_ncar
    from atlite.datasets.era5_ncar import _fetch_vars

    output_path = Path("germany_consol_test.nc")
    if output_path.exists():
        output_path.unlink()

    cutout = atlite.Cutout(
        path=output_path,
        module="era5-ncar",
        x=slice(BOUNDS[0], BOUNDS[2]),
        y=slice(BOUNDS[1], BOUNDS[3]),
        time="2013",
    )
    coords = cutout.coords

    test_vars = ["u10", "ssrd", "fsr"]

    print("\n" + "=" * 70)
    print("TEST A: Consolidation write speed (from cached raw files)")
    print("=" * 70)

    for compress in [False, True]:
        label = "complevel=1" if compress else "uncompressed"

        for sn in test_vars:
            p = os.path.join(TMPDIR, f"consolidated_{sn}.nc")
            if os.path.exists(p):
                os.unlink(p)

        if not compress:
            saved = dict(era5_ncar._CONSOLIDATED_ENCODING)
            era5_ncar._CONSOLIDATED_ENCODING = {}

        with time_block(f"consolidate {test_vars} ({label})") as timer:
            _fetch_vars(test_vars, coords, tmpdir=TMPDIR)

        if not compress:
            era5_ncar._CONSOLIDATED_ENCODING = saved

        total = 0
        for sn in test_vars:
            p = os.path.join(TMPDIR, f"consolidated_{sn}.nc")
            sz = os.path.getsize(p)
            total += sz
            logger.info("  %s: %.1f MB", sn, sz / 1e6)
        logger.info("  Total: %.1f MB in %.2f s", total / 1e6, timer.elapsed)

    cutout.data.close()
    if output_path.exists():
        output_path.unlink()


def run_test_b_single(consol_compress: bool):
    """Run a single final-write test. Called as subprocess to avoid lock issues."""
    import atlite
    from atlite.datasets import era5_ncar
    from atlite.data import get_features, available_features, non_bool_dict
    from numpy import atleast_1d

    consol_label = "compressed" if consol_compress else "uncompressed"
    output_path = Path("germany_consol_test.nc")
    if output_path.exists():
        output_path.unlink()

    # Delete all consolidated caches and rebuild
    for f in Path(TMPDIR).glob("consolidated_*.nc"):
        f.unlink()

    if not consol_compress:
        era5_ncar._CONSOLIDATED_ENCODING = {}

    cutout = atlite.Cutout(
        path=output_path,
        module="era5-ncar",
        x=slice(BOUNDS[0], BOUNDS[2]),
        y=slice(BOUNDS[1], BOUNDS[3]),
        time="2013",
    )

    modules = atleast_1d(cutout.module)
    all_features = ["height", "wind", "influx", "temperature", "runoff"]
    target = available_features(modules).loc[:, all_features].drop_duplicates()
    missing_vars = target[modules[0]].values

    with time_block(f"get_features + consolidate all ({consol_label})") as t_load:
        ds = get_features(cutout, modules[0],
                          target[modules[0]].index.unique("feature"),
                          tmpdir=TMPDIR, data_format="grib")

    total_consol = sum(f.stat().st_size for f in Path(TMPDIR).glob("consolidated_*.nc"))
    logger.info("  Consolidated total: %.1f MB", total_consol / 1e6)

    prepared = set(target[modules[0]].index.unique("feature"))
    cutout.data.attrs.update(dict(prepared_features=list(prepared)))
    attrs = non_bool_dict(cutout.data.attrs)
    attrs.update(ds.attrs)

    for v in missing_vars:
        ds[v].encoding.clear()
        ds[v].encoding.update({"zlib": True, "complevel": 1, "shuffle": True})

    ds_merged = cutout.data.merge(ds[missing_vars]).assign_attrs(**attrs)

    from tempfile import mkstemp
    fd, tmp = mkstemp(suffix=".nc", dir=".")
    os.close(fd)
    try:
        with time_block(f"final write complevel=1 (from {consol_label} consolidated)") as t_write:
            write_job = ds_merged.to_netcdf(tmp, compute=False)
            write_job.compute()
        fsize = os.path.getsize(tmp)
        logger.info("  Output: %.2f GB", fsize / 1e9)
    finally:
        os.unlink(tmp)

    cutout.data.close()
    if output_path.exists():
        output_path.unlink()

    print(f"\nRESULT: consol={consol_label} load={t_load.elapsed:.1f}s write={t_write.elapsed:.1f}s total={t_load.elapsed+t_write.elapsed:.1f}s consol_size={total_consol/1e6:.0f}MB")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test-b":
        compress = sys.argv[2] == "compressed"
        run_test_b_single(compress)
        return

    # Test A: runs in this process
    run_test_a()

    # Test B: each variant in a fresh subprocess to avoid lock/fd issues
    print("\n" + "=" * 70)
    print("TEST B: Full pipeline (consolidate + final write) from each source")
    print("=" * 70)

    for variant in ["uncompressed", "compressed"]:
        logger.info("--- Running test B: %s consolidated ---", variant)
        result = subprocess.run(
            [sys.executable, __file__, "--test-b", variant],
            capture_output=False,
            timeout=600,
        )
        if result.returncode != 0:
            logger.error("Test B (%s) failed with exit code %d", variant, result.returncode)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
