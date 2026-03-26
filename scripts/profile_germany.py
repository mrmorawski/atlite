"""
Profile cutout.prepare() on Germany cutout — full year 2013.
"""

import logging
import os
import sys
import time
from pathlib import Path

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


def main():
    import atlite

    os.makedirs(TMPDIR, exist_ok=True)
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

    t0 = time.time()
    cutout.prepare(
        features=ALL_FEATURES,
        tmpdir=TMPDIR,
        compression={"zlib": True, "complevel": 1, "shuffle": True},
    )
    elapsed = time.time() - t0

    fsize = output_path.stat().st_size
    logger.info(
        "Output: %.2f GB in %.1f s (%.1f min)", fsize / 1e9, elapsed, elapsed / 60
    )


if __name__ == "__main__":
    main()
