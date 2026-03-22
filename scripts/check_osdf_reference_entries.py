"""
Check what variables are available in the OSDF "reference" format entries.
These may be kerchunk references pointing to the forecast-accumulated NetCDF files,
which would give us chunk-level access to ssrd, fdir, tp, ro, etc.

Usage:
    python scripts/check_osdf_reference_entries.py
"""

import requests
import pandas as pd
from urllib.parse import urljoin

CATALOG_JSON_URL = (
    "https://osdf-director.osg-htc.org/ncar/gdex/d633000/catalogs/d633000-osdf.json"
)

GLADE_PREFIX = "/glade/campaign/collections/gdex/data/d633000/"
OSDF_PREFIX = "https://osdf-director.osg-htc.org/ncar/gdex/d633000/"

# Variables we need for atlite that are forecast-accumulated in ERA5
FORECAST_TARGETS = ["ssrd", "strd", "tp", "ro", "fdir", "ssr", "tisr"]


def posix_to_osdf(posix_path):
    if posix_path.startswith(GLADE_PREFIX):
        return OSDF_PREFIX + posix_path[len(GLADE_PREFIX):]
    idx = posix_path.find("/d633000/")
    if idx >= 0:
        return OSDF_PREFIX + posix_path[idx + len("/d633000/"):]
    return posix_path


def main():
    # Fetch catalog
    resp = requests.get(CATALOG_JSON_URL, timeout=30)
    resp.raise_for_status()
    meta = resp.json()

    csv_url = urljoin(CATALOG_JSON_URL, meta["catalog_file"])
    df = pd.read_csv(csv_url)

    print(f"Total entries: {len(df)}")
    print(f"Format counts:\n{df['format'].value_counts()}\n")

    # Reference entries
    refs = df[df["format"] == "reference"]
    print(f"Reference format short_names ({len(refs)} entries):")
    print(sorted(refs["short_name"].dropna().unique()))

    print(f"\nReference format example paths:")
    for _, row in refs.head(5).iterrows():
        posix = row["path"]
        osdf = posix_to_osdf(posix)
        print(f"  {row['short_name']:8s} POSIX: {posix}")
        print(f"           OSDF:  {osdf}")

    # Check for our forecast targets
    print(f"\nForecast-accumulated variables we need:")
    for target in FORECAST_TARGETS:
        matches = refs[refs["short_name"] == target]
        if len(matches) > 0:
            row = matches.iloc[0]
            osdf = posix_to_osdf(row["path"])
            print(f"  {target:6s} FOUND  {osdf}")
        else:
            # Also check zarr entries
            zarr_matches = df[(df["short_name"] == target) & (df["format"] == "zarr")]
            if len(zarr_matches) > 0:
                print(f"  {target:6s} FOUND (zarr only)")
            else:
                print(f"  {target:6s} NOT FOUND in any format")

    # If we found reference entries, try to fetch one to see what's inside
    print(f"\nAttempting to read a reference entry...")
    sample_refs = refs[refs["short_name"].isin(FORECAST_TARGETS)]
    if len(sample_refs) == 0:
        sample_refs = refs.head(1)

    if len(sample_refs) > 0:
        row = sample_refs.iloc[0]
        osdf = posix_to_osdf(row["path"])
        print(f"  Fetching: {osdf}")
        try:
            # Try reading as parquet (kerchunk reference)
            r = requests.get(osdf, timeout=30, stream=True)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "unknown")
            first_bytes = r.content[:500]
            print(f"  Content-Type: {content_type}")
            print(f"  Size: {len(r.content)} bytes")
            print(f"  First 200 bytes (repr): {first_bytes[:200]!r}")

            # If it's parquet, try reading with pandas
            if b"PAR1" in first_bytes[:4]:
                import io
                ref_df = pd.read_parquet(io.BytesIO(r.content))
                print(f"  Parquet columns: {list(ref_df.columns)}")
                print(f"  Parquet shape: {ref_df.shape}")
                print(f"  First few rows:\n{ref_df.head()}")
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
