"""
Download an ERA5-NCAR cutout for all of Europe, full year 2013.

Usage
-----
    python scripts/download_europe_2013.py [--features wind influx ...]

The cutout is written to europe_2013_ncar.nc (or --output PATH).
Progress and timing are logged to stdout; a summary is printed at the end.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import atlite

# ---------------------------------------------------------------------------
# Logging — timestamps + module name so per-feature timing is visible
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-24s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("download_europe_2013")

# ---------------------------------------------------------------------------
# Europe bounding box (lon_min, lat_min, lon_max, lat_max)
# ---------------------------------------------------------------------------
BOUNDS = (-25, 34, 45, 72)  # West Atlantic coast → Ural foothills, Med → N Norway

ALL_FEATURES = ["height", "wind", "influx", "temperature", "runoff"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="europe_2013_ncar.nc", type=Path)
    parser.add_argument(
        "--features", nargs="+", default=ALL_FEATURES,
        choices=ALL_FEATURES, metavar="FEATURE",
        help="Features to prepare (default: all)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Cutout definition
    # ------------------------------------------------------------------
    logger.info("Creating cutout: %s", args.output)
    cutout = atlite.Cutout(
        path=args.output,
        module="era5-ncar",
        x=slice(BOUNDS[0], BOUNDS[2]),
        y=slice(BOUNDS[1], BOUNDS[3]),
        time="2013",
    )

    nx = len(cutout.coords["x"])
    ny = len(cutout.coords["y"])
    nt = len(cutout.coords["time"])
    logger.info(
        "Grid: %d × %d (lon × lat), %d timesteps (%.1f°×%.1f° at 0.25° native)",
        nx, ny, nt,
        float(cutout.coords["x"].max()) - float(cutout.coords["x"].min()),
        float(cutout.coords["y"].max()) - float(cutout.coords["y"].min()),
    )

    # ------------------------------------------------------------------
    # Prepare — time each feature
    # ------------------------------------------------------------------
    feature_times = {}
    t_total = time.time()

    for feature in args.features:
        if cutout.prepared and feature in cutout.prepared:
            logger.info("Feature '%s' already in cutout — skipping", feature)
            continue

        logger.info("=== preparing feature: %s ===", feature)
        t0 = time.time()
        cutout.prepare(features=[feature])
        dt = time.time() - t0
        feature_times[feature] = dt
        logger.info("=== feature '%s' done in %.1f min ===", feature, dt / 60)

    elapsed = time.time() - t_total

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    output_size_mb = args.output.stat().st_size / 1e6 if args.output.exists() else float("nan")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Output:          {args.output}  ({output_size_mb:.0f} MB)")
    print(f"  Grid:            {nx} × {ny} lon/lat,  {nt} timesteps")
    print()
    if feature_times:
        print("  Feature times:")
        for feat, dt in feature_times.items():
            print(f"    {feat:<14} {dt/60:5.1f} min")
        print()
    print(f"  Total wall time: {elapsed/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
