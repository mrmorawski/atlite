"""Profile consolidation: read vs compute vs write."""
import os, time
import numpy as np, xarray as xr, pandas as pd

tmpdir = "tmp"
months = [(2013, m) for m in range(1, 13)]
t = pd.date_range("2013-01-01", "2013-12-31 23:00", freq="h")

# --- Phase A: open + concat (lazy) ---
t0 = time.time()
raw_datasets = []
parts = []
for year, month in months:
    path = os.path.join(tmpdir, f"era5_ncar_v10_{year}_{month:02d}_2df5d529.nc")
    raw_ds = xr.open_dataset(path, chunks={"time": 24})
    raw_datasets.append(raw_ds)
    chunk = raw_ds["data"]
    mt = t[(t.year == year) & (t.month == month)]
    if len(mt):
        parts.append(
            chunk.sel(time=mt, method="nearest", tolerance=pd.Timedelta("30min"))
            .assign_coords(time=mt.values)
        )
da = xr.concat(parts, dim="time")
t_open = time.time() - t0
print(f"Open + concat (lazy):  {t_open:.2f}s  shape={da.shape} chunks={da.chunks[0][:3]}...")

# --- Phase B: compute (synchronous) ---
t0 = time.time()
to_write = xr.Dataset({"data": da})
for v in to_write.data_vars:
    to_write[v].encoding.clear()
computed = to_write.compute(scheduler="synchronous")
t_compute = time.time() - t0
nbytes = computed["data"].nbytes
print(f"Compute (synchronous): {t_compute:.2f}s  ({nbytes/1e6:.0f} MB, {nbytes/1e6/t_compute:.0f} MB/s)")

# --- Phase C: write uncompressed ---
out = os.path.join(tmpdir, "test_bench.nc")
t0 = time.time()
computed.to_netcdf(out)
t_write = time.time() - t0
fsize = os.path.getsize(out) / 1e6
print(f"Write (uncompressed):  {t_write:.2f}s  ({fsize:.0f} MB, {fsize/t_write:.0f} MB/s)")
os.unlink(out)

# --- Phase D: write with zlib 9 ---
t0 = time.time()
enc = {"data": {"zlib": True, "complevel": 9, "shuffle": True}}
computed.to_netcdf(out, encoding=enc)
t_write_z = time.time() - t0
fsize_z = os.path.getsize(out) / 1e6
print(f"Write (zlib 9):        {t_write_z:.2f}s  ({fsize_z:.0f} MB, {nbytes/1e6/t_write_z:.0f} MB/s raw)")
os.unlink(out)

# --- Phase E: compute with threaded scheduler (for comparison) ---
# Re-open to get fresh dask graph
for ds in raw_datasets:
    ds.close()
raw_datasets = []
parts = []
for year, month in months:
    path = os.path.join(tmpdir, f"era5_ncar_v10_{year}_{month:02d}_2df5d529.nc")
    raw_ds = xr.open_dataset(path, chunks={"time": 24})
    raw_datasets.append(raw_ds)
    chunk = raw_ds["data"]
    mt = t[(t.year == year) & (t.month == month)]
    if len(mt):
        parts.append(
            chunk.sel(time=mt, method="nearest", tolerance=pd.Timedelta("30min"))
            .assign_coords(time=mt.values)
        )
da = xr.concat(parts, dim="time")
to_write = xr.Dataset({"data": da})
for v in to_write.data_vars:
    to_write[v].encoding.clear()

t0 = time.time()
computed2 = to_write.compute(scheduler="threads", num_workers=4)
t_compute_threaded = time.time() - t0
print(f"Compute (4 threads):   {t_compute_threaded:.2f}s  ({nbytes/1e6/t_compute_threaded:.0f} MB/s)")

# --- Phase F: larger chunks ---
for ds in raw_datasets:
    ds.close()
raw_datasets = []
parts = []
for year, month in months:
    path = os.path.join(tmpdir, f"era5_ncar_v10_{year}_{month:02d}_2df5d529.nc")
    raw_ds = xr.open_dataset(path, chunks={"time": -1})  # whole file = one chunk
    raw_datasets.append(raw_ds)
    chunk = raw_ds["data"]
    mt = t[(t.year == year) & (t.month == month)]
    if len(mt):
        parts.append(
            chunk.sel(time=mt, method="nearest", tolerance=pd.Timedelta("30min"))
            .assign_coords(time=mt.values)
        )
da = xr.concat(parts, dim="time")
to_write = xr.Dataset({"data": da})
for v in to_write.data_vars:
    to_write[v].encoding.clear()

t0 = time.time()
computed3 = to_write.compute(scheduler="synchronous")
t_compute_big = time.time() - t0
print(f"Compute (big chunks):  {t_compute_big:.2f}s  ({nbytes/1e6/t_compute_big:.0f} MB/s)")

for ds in raw_datasets:
    ds.close()

print(f"\nTotal per variable: {t_open + t_compute + t_write:.1f}s (uncompressed), {t_open + t_compute + t_write_z:.1f}s (zlib 9)")
