"""
Download an ERA5-NCAR cutout for Europe, full year 2013.

Usage
-----
    python scripts/download_europe_2013.py

The cutout is written to europe_2013_ncar.nc (or --output PATH).
Progress and timing are logged to stdout; a summary is printed at the end.
"""

import logging
import sys
import time
import threading
from pathlib import Path

import atlite

# ---------------------------------------------------------------------------
# Logging — timestamps + module name
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-24s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("download_europe_2013")

# ---------------------------------------------------------------------------
# Germany bounding box (lon_min, lat_min, lon_max, lat_max)
# ---------------------------------------------------------------------------
BOUNDS = (-25.0, 34.0, 45.0, 72.0)  # Europe bounding box

ALL_FEATURES = ["height", "wind", "influx", "temperature", "runoff"]


def _net_rx_bytes() -> int:
    """Return total bytes received across all non-loopback interfaces."""
    total = 0
    with open("/proc/net/dev") as f:
        for line in f:
            if ":" not in line:
                continue
            iface, data = line.split(":", 1)
            if iface.strip() == "lo":
                continue
            total += int(data.split()[0])
    return total


def _progress_reporter(
    output_path: Path,
    stop_event: threading.Event,
    interval: float = 10.0,
):
    """Periodically log download progress via /proc/net/dev."""
    baseline = _net_rx_bytes()
    prev_bytes = baseline
    prev_time = time.time()

    while not stop_event.wait(interval):
        now = time.time()
        dt = max(now - prev_time, 1e-3)

        current = _net_rx_bytes()
        total = current - baseline
        rate_mb_s = (current - prev_bytes) / 1e6 / dt
        logger.info(
            "Downloaded: %.1f MB  (%.2f MB/s)",
            total / 1e6,
            rate_mb_s,
        )
        prev_bytes = current
        prev_time = now


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="europe_2013_ncar.nc", type=Path)
    parser.add_argument(
        "--tmpdir",
        default=None,
        type=Path,
        help="Persistent cache dir for resumable downloads",
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
        nx,
        ny,
        nt,
        float(cutout.coords["x"].max()) - float(cutout.coords["x"].min()),
        float(cutout.coords["y"].max()) - float(cutout.coords["y"].min()),
    )

    # ------------------------------------------------------------------
    # Prepare all features, reporting download progress every 10 s
    # ------------------------------------------------------------------
    stop_event = threading.Event()
    reporter = threading.Thread(
        target=_progress_reporter,
        args=(args.output, stop_event),
        daemon=True,
    )

    logger.info("=== preparing all features: %s ===", ", ".join(ALL_FEATURES))
    t0 = time.time()
    reporter.start()
    try:
        cutout.prepare(
            features=ALL_FEATURES,
            tmpdir=str(args.tmpdir) if args.tmpdir else None,
            compression={"zlib": True, "complevel": 1, "shuffle": True},
        )
    finally:
        stop_event.set()
        reporter.join()

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    output_size_mb = (
        args.output.stat().st_size / 1e6 if args.output.exists() else float("nan")
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Output:          {args.output}  ({output_size_mb:.0f} MB)")
    print(f"  Grid:            {nx} × {ny} lon/lat,  {nt} timesteps")
    print(f"  Total wall time: {elapsed / 60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
