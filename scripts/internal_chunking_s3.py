"""
Inspect internal HDF5/NetCDF4 chunking of ERA5 files on the NCAR S3 mirror.

This tells us whether partial spatial reads via HTTP range requests are feasible.
If the files are chunked spatially (e.g. 64x64 lat/lon tiles), we can read only
the chunks overlapping our bounding box. If they're stored as one big chunk per
time step (721x1440), we must download the full global grid for each access.

Uses xarray + netCDF4 engine with an s3fs filesystem — this reads only metadata
via HTTP range requests without downloading the whole file.

Usage:
    pip install s3fs netCDF4 xarray
    python scripts/internal_chunking_s3.py

No authentication required (AWS Open Data Program).
"""

import s3fs
import xarray as xr
import numpy as np

BUCKET = "nsf-ncar-era5"

# One analysis surface file, one forecast accumulated file, and one invariant,
# since they may have different chunking strategies.
FILES = {
    "an.sfc (u10, Jan 2013)": (
        f"{BUCKET}/e5.oper.an.sfc/201301/"
        "e5.oper.an.sfc.128_165_10u.ll025sc.2013010100_2013013123.nc"
    ),
    "fc.sfc.accumu (ssrd, Jan 2013 first half)": (
        f"{BUCKET}/e5.oper.fc.sfc.accumu/201301/"
        "e5.oper.fc.sfc.accumu.128_169_ssrd.ll025sc.2013010106_2013011606.nc"
    ),
    "invariant (geopotential)": (
        f"{BUCKET}/e5.oper.invariant/"
        "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
    ),
}


def inspect_file(fs, path, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  s3://{path}")
    print(f"{'='*70}")

    # Get file size
    info = fs.info(path)
    size_mb = info["size"] / (1024 * 1024)
    print(f"\nFile size: {size_mb:.1f} MB")

    # Open with xarray using the s3fs filesystem.
    # netCDF4 engine + s3fs does range-request reads for metadata only.
    fobj = fs.open(path, "rb")
    ds = xr.open_dataset(fobj, engine="scipy")

    print(f"\nDimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Data variables: {list(ds.data_vars)}")

    # Show coordinate ranges
    for coord_name in ds.coords:
        c = ds.coords[coord_name]
        if c.size <= 1:
            print(f"\n  {coord_name}: {c.values}")
        elif c.size > 1:
            step = float(c[1] - c[0])
            print(f"  {coord_name}: {float(c.min()):.2f} to {float(c.max()):.2f}, "
                  f"step={step:.4f}, n={c.size}")

    # scipy engine doesn't expose HDF5 chunking, so we report encoding
    print(f"\nVariable details:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        enc = var.encoding
        print(f"\n  {var_name}:")
        print(f"    shape: {var.shape}")
        print(f"    dims: {var.dims}")
        print(f"    dtype: {var.dtype}")
        if "chunksizes" in enc:
            chunks = enc["chunksizes"]
            print(f"    chunksizes: {chunks}")
            chunk_bytes = int(np.prod(chunks)) * var.dtype.itemsize
            print(f"    chunk_size: {chunk_bytes / 1024:.1f} KB")
            n_chunks = tuple(
                (s + c - 1) // c for s, c in zip(var.shape, chunks)
            )
            print(f"    n_chunks_per_dim: {n_chunks}")
            print(f"    total_chunks: {int(np.prod(n_chunks))}")
        else:
            print(f"    chunksizes: not available (scipy engine)")
            print(f"    encoding keys: {list(enc.keys())}")

    ds.close()
    fobj.close()
    print()


def inspect_file_h5netcdf(fs, path, label):
    """Alternative: use h5netcdf which exposes HDF5 chunking directly."""
    print(f"\n{'='*70}")
    print(f"  {label} [h5netcdf]")
    print(f"  s3://{path}")
    print(f"{'='*70}")

    info = fs.info(path)
    size_mb = info["size"] / (1024 * 1024)
    print(f"\nFile size: {size_mb:.1f} MB")

    fobj = fs.open(path, "rb")
    try:
        ds = xr.open_dataset(fobj, engine="h5netcdf")
    except Exception as e:
        print(f"  h5netcdf failed: {e}")
        fobj.close()
        return

    print(f"\nDimensions: {dict(ds.dims)}")

    print(f"\nVariable details:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        enc = var.encoding
        print(f"\n  {var_name}:")
        print(f"    shape: {var.shape}")
        print(f"    dims: {var.dims}")
        print(f"    dtype: {var.dtype}")
        # h5netcdf exposes chunksizes in encoding
        for key in ["chunksizes", "chunks", "compression", "shuffle",
                     "complevel", "fletcher32"]:
            if key in enc:
                print(f"    {key}: {enc[key]}")
        if "chunksizes" not in enc and "chunks" not in enc:
            print(f"    encoding keys: {list(enc.keys())}")

    ds.close()
    fobj.close()
    print()


def main():
    print("Connecting to S3 (anonymous)...")
    fs = s3fs.S3FileSystem(anon=True)

    # Verify bucket is accessible
    try:
        top_level = fs.ls(BUCKET, detail=False)[:5]
        print(f"Bucket accessible. Top-level entries: {top_level}")
    except Exception as e:
        print(f"ERROR: Cannot access bucket: {e}")
        return

    # Try h5netcdf first (exposes chunking), fall back to scipy
    for label, path in FILES.items():
        try:
            inspect_file_h5netcdf(fs, path, label)
        except Exception as e:
            print(f"\nh5netcdf failed for {label}: {e}")
            try:
                inspect_file(fs, path, label)
            except Exception as e2:
                print(f"scipy also failed for {label}: {e2}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: What this means for partial reads")
    print("=" * 70)
    print("""
If chunks are (1, 721, 1440) or similar full-grid:
  -> Each chunk = entire global field for one timestep
  -> Must download ~4 MB per timestep even for a small region
  -> 1 year x 1 variable = ~35 GB downloaded to extract ~80 MB of EU data
  -> S3 partial reads are NOT efficient for spatial subsets

If chunks are (1, 64, 64) or similar spatial tiles:
  -> Each chunk = one spatial tile for one timestep
  -> Can read only tiles overlapping the bounding box
  -> Much more efficient for regional subsets
  -> Feasibility depends on tile size vs region size
""")


if __name__ == "__main__":
    main()
