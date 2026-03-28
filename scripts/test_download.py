import atlite
from pathlib import Path
import logging
import sys

BOUNDS = (5.5, 5.75, 15.5, 15.75)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)-24s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def main():
    cutout = atlite.Cutout(
        path=Path.cwd() / "test.nc",
        module="era5-ncar",
        x=slice(BOUNDS[0], BOUNDS[2]),
        y=slice(BOUNDS[1], BOUNDS[3]),
        time="2013",
        show_progress=True,
    )
    cutout.prepare(features="wind")


if __name__ == "__main__":
    main()
