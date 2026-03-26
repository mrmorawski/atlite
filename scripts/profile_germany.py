"""
Profile cutout.prepare() on Germany cutout — full year 2013.
"""

import logging
import os
import sys
import time
from pathlib import Path


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

    cutout.prepare(
        features=ALL_FEATURES,
        tmpdir=TMPDIR,
        compression={"zlib": True, "complevel": 1, "shuffle": True},
        show_progress=True,
    )


if __name__ == "__main__":
    main()
