"""
Inspect the NCAR OSDF ERA5 Zarr store chunk layout and estimate download sizes.

The NCAR OSDF (Open Science Data Federation) serves ERA5 as Zarr stores.
The intake-esm catalog lists POSIX paths (/glade/campaign/...) which must be
converted to OSDF HTTP URLs for remote access.

Path mapping:
  /glade/campaign/collections/gdex/data/d633000/...
  -> https://osdf-director.osg-htc.org/ncar/gdex/d633000/...

We need to understand:
  1. What variables are available?
  2. How is data chunked (time, lat, lon)?
  3. For a typical EU bounding box, how many chunks must we read per variable?
  4. What's the total estimated download for 1 year x 16 variables?

Usage:
    pip install xarray zarr requests fsspec aiohttp pandas
    python scripts/internal_chunking_zarr.py

No authentication required.
"""

import json
import requests
import xarray as xr
import pandas as pd
from urllib.parse import urljoin

CATALOG_JSON_URL = (
    "https://osdf-director.osg-htc.org/ncar/gdex/d633000/catalogs/d633000-osdf.json"
)

# POSIX prefix on NCAR's GLADE filesystem -> OSDF HTTP prefix
GLADE_PREFIX = "/glade/campaign/collections/gdex/data/d633000/"
OSDF_PREFIX = "https://osdf-director.osg-htc.org/ncar/gdex/d633000/"

# EU bounding box for download size estimation
EU_LAT = (35, 72)  # degrees N
EU_LON = (-25, 45)  # degrees E
HOURS_PER_YEAR = 8760

# Variables we care about (ERA5 short names)
TARGET_VARS = ["10u", "2t", "ssrd", "z"]


def posix_to_osdf(posix_path):
    """Convert a GLADE POSIX path to an OSDF HTTP URL."""
    if posix_path.startswith(GLADE_PREFIX):
        return OSDF_PREFIX + posix_path[len(GLADE_PREFIX):]
    # Try a more relaxed match
    for marker in ["/d633000/", "/ds633.0/"]:
        idx = posix_path.find(marker)
        if idx >= 0:
            return OSDF_PREFIX + posix_path[idx + len(marker):]
    return posix_path  # Return as-is if no match


def fetch_catalog():
    """Fetch the intake-esm catalog JSON and its CSV listing."""
    print("=" * 70)
    print("  Step 1: Fetch catalog")
    print("=" * 70)

    resp = requests.get(CATALOG_JSON_URL, timeout=30)
    resp.raise_for_status()
    meta = resp.json()

    print(f"\nCatalog ID: {meta.get('id', 'N/A')}")
    print(f"Description: {meta.get('description', 'N/A')}")
    print(f"Last updated: {meta.get('last_updated', 'N/A')}")

    # Fetch the CSV catalog (path is relative to the JSON URL)
    csv_filename = meta["catalog_file"]
    csv_url = urljoin(CATALOG_JSON_URL, csv_filename)
    print(f"\nCSV URL: {csv_url}")

    df = pd.read_csv(csv_url)
    print(f"Total entries: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Filter to zarr-format entries only
    if "format" in df.columns:
        zarr_df = df[df["format"] == "zarr"]
        print(f"Zarr entries: {len(zarr_df)} (of {len(df)} total)")
        other_formats = df[df["format"] != "zarr"]["format"].unique()
        if len(other_formats) > 0:
            print(f"Other formats: {list(other_formats)}")
    else:
        zarr_df = df

    # Show available short_names
    if "short_name" in zarr_df.columns:
        names = sorted(zarr_df["short_name"].dropna().unique())
        print(f"\nAvailable short_names ({len(names)}): {names}")

    # Show example paths and their OSDF conversions
    print(f"\nExample path conversion:")
    sample = zarr_df.iloc[0]["path"]
    print(f"  POSIX: {sample}")
    print(f"  OSDF:  {posix_to_osdf(sample)}")

    return meta, zarr_df


def inspect_zarr_store(zarr_url, label):
    """Open a single Zarr store and inspect its chunk layout."""
    print(f"\n{'-'*70}")
    print(f"  {label}")
    print(f"  {zarr_url}")
    print(f"{'-'*70}")

    # Try different open strategies
    ds = None
    errors = []
    for kwargs in [
        {"consolidated": True, "zarr_format": 2},
        {"consolidated": True},
        {"consolidated": False},
        {},
    ]:
        try:
            ds = xr.open_dataset(
                zarr_url,
                engine="zarr",
                backend_kwargs=kwargs,
                chunks={},  # Use Zarr's native chunking (lazy/dask)
            )
            print(f"  Opened with backend_kwargs={kwargs}")
            break
        except Exception as e:
            errors.append(f"  {kwargs}: {e}")
            continue

    if ds is None:
        print(f"  FAILED to open. Errors:")
        for err in errors:
            print(err)
        return None

    print(f"\n  Dimensions: {dict(ds.dims)}")
    print(f"  Coordinates: {list(ds.coords)}")
    print(f"  Data variables: {list(ds.data_vars)}")

    # Check coordinate ranges
    for coord_name in ["latitude", "lat"]:
        if coord_name in ds.coords:
            c = ds.coords[coord_name]
            step = float(c[1] - c[0]) if c.size > 1 else "N/A"
            print(f"\n  {coord_name}: {float(c.min()):.2f} to {float(c.max()):.2f}, "
                  f"step={step}, n={c.size}")
    for coord_name in ["longitude", "lon"]:
        if coord_name in ds.coords:
            c = ds.coords[coord_name]
            step = float(c[1] - c[0]) if c.size > 1 else "N/A"
            print(f"  {coord_name}: {float(c.min()):.2f} to {float(c.max()):.2f}, "
                  f"step={step}, n={c.size}")
    if "time" in ds.coords:
        t = ds.coords["time"]
        print(f"  time: {t.values[0]} ... {t.values[-1]}, n={t.size}")

    # Inspect chunking of each data variable
    print(f"\n  Variable chunk layouts:")
    for var_name in sorted(ds.data_vars):
        var = ds[var_name]
        if var.chunks:
            chunk_summary = {}
            for dim, chunks in zip(var.dims, var.chunks):
                unique_sizes = sorted(set(chunks))
                if len(unique_sizes) == 1:
                    chunk_summary[dim] = unique_sizes[0]
                elif len(unique_sizes) <= 3:
                    chunk_summary[dim] = unique_sizes
                else:
                    chunk_summary[dim] = (
                        f"{unique_sizes[0]}-{unique_sizes[-1]} "
                        f"({len(unique_sizes)} sizes)"
                    )

            # Chunk size in bytes (use first chunk)
            chunk_elems = 1
            for chunks in var.chunks:
                chunk_elems *= chunks[0]
            chunk_bytes = chunk_elems * var.dtype.itemsize
            chunk_kb = chunk_bytes / 1024

            print(f"\n    {var_name}:")
            print(f"      shape={var.shape}, dtype={var.dtype}")
            print(f"      chunks={chunk_summary}")
            print(f"      chunk_size={chunk_kb:.1f} KB ({chunk_bytes} bytes)")

            estimate_eu_download(var_name, var, chunk_summary)
        else:
            print(f"\n    {var_name}: shape={var.shape}, dtype={var.dtype}, unchunked")

    ds.close()
    return True


def estimate_eu_download(var_name, var, chunk_summary):
    """Estimate bytes we'd need to download for an EU bounding box, 1 year."""
    dims = var.dims

    lat_dim = next((d for d in dims if d in ("latitude", "lat")), None)
    lon_dim = next((d for d in dims if d in ("longitude", "lon")), None)
    time_dim = "time" if "time" in dims else None

    if not lat_dim or not lon_dim:
        return

    lat_idx = dims.index(lat_dim)
    lon_idx = dims.index(lon_dim)

    lat_size = var.shape[lat_idx]
    lon_size = var.shape[lon_idx]

    # Assume 0.25deg grid
    eu_lat_points = int((EU_LAT[1] - EU_LAT[0]) / 0.25) + 1  # 149
    eu_lon_points = int((EU_LON[1] - EU_LON[0]) / 0.25) + 1  # 281

    lat_chunk = chunk_summary.get(lat_dim)
    lon_chunk = chunk_summary.get(lon_dim)
    time_chunk = chunk_summary.get("time", 1) if time_dim else 1

    if not isinstance(lat_chunk, int) or not isinstance(lon_chunk, int):
        print(f"      (non-uniform chunks, skipping estimate)")
        return
    if not isinstance(time_chunk, int):
        print(f"      (non-uniform time chunks, skipping estimate)")
        return

    n_lat_chunks = (eu_lat_points + lat_chunk - 1) // lat_chunk
    n_lon_chunks = (eu_lon_points + lon_chunk - 1) // lon_chunk

    dl_lat = min(n_lat_chunks * lat_chunk, lat_size)
    dl_lon = min(n_lon_chunks * lon_chunk, lon_size)

    overhead = (dl_lat * dl_lon) / (eu_lat_points * eu_lon_points)

    time_size = var.shape[dims.index(time_dim)] if time_dim else 1
    target_hours = min(HOURS_PER_YEAR, time_size)
    n_time_chunks = (target_hours + time_chunk - 1) // time_chunk

    total_chunks = n_lat_chunks * n_lon_chunks * n_time_chunks
    chunk_bytes = int(lat_chunk * lon_chunk * time_chunk * var.dtype.itemsize)
    total_bytes = total_chunks * chunk_bytes
    total_mb = total_bytes / (1024 * 1024)

    needed_bytes = eu_lat_points * eu_lon_points * target_hours * var.dtype.itemsize
    needed_mb = needed_bytes / (1024 * 1024)

    print(f"      EU 1yr estimate: {n_time_chunks} time x {n_lat_chunks} lat x "
          f"{n_lon_chunks} lon = {total_chunks} chunks")
    print(f"      Download: {total_mb:.1f} MB (needed: {needed_mb:.1f} MB, "
          f"overhead: {overhead:.1f}x spatial)")


def main():
    print("NCAR OSDF ERA5 Zarr Store Inspector")
    print("=" * 70)

    # Step 1: Fetch catalog
    meta, zarr_df = fetch_catalog()

    if zarr_df is None or len(zarr_df) == 0:
        print("\nNo Zarr catalog entries found. Cannot proceed.")
        return

    # Step 2: Pick representative stores to inspect
    print(f"\n{'='*70}")
    print("  Step 2: Inspect Zarr store chunk layouts")
    print("=" * 70)

    inspected = 0
    for target in TARGET_VARS:
        if "short_name" not in zarr_df.columns:
            break
        matches = zarr_df[zarr_df["short_name"] == target]
        if len(matches) == 0:
            print(f"\n  No entry for short_name='{target}'")
            continue

        row = matches.iloc[0]
        posix_path = row["path"]
        zarr_url = posix_to_osdf(posix_path)
        label = f"{row.get('short_name', '?')} ({row.get('variable', '?')})"

        result = inspect_zarr_store(zarr_url, label)
        if result:
            inspected += 1

    # If nothing worked, try first 3 entries with converted URLs
    if inspected == 0:
        print("\nTarget variables not found. Trying first 3 entries:")
        for _, row in zarr_df.head(3).iterrows():
            posix_path = row["path"]
            zarr_url = posix_to_osdf(posix_path)
            label = f"{row.get('short_name', '?')} ({row.get('variable', '?')})"
            inspect_zarr_store(zarr_url, label)

    # Step 3: Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print("=" * 70)
    print("""
Key questions answered by this output:

1. Chunk layout: (time, lat, lon) dimensions and sizes
   - Ideal for us: small spatial chunks (e.g. 32x32 or 64x64)
   - Worst case: full grid per chunk (721x1440)

2. Overhead ratio: how much extra data we download due to chunk boundaries
   - 1.0x = perfect (only read what we need)
   - 10x+ = most of each chunk is wasted

3. Total download for our use case (1 year x 16 vars x EU bbox):
   Compare against OPeNDAP (~300 MB) and S3 whole-file (~40-90 GB)
""")


if __name__ == "__main__":
    main()
