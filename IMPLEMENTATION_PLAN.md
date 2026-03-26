<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: MIT
-->

# era5-ncar Dataset Module for Atlite — Implementation Plan

## Status: REFACTOR IN PROGRESS — consolidation removal

All phases (A–E) are implemented and tested.  15/16 tests pass on the first run
(the 16th, `test_compare_with_era5`, requires the CDS reference cutout to also
be cached — run `TestERA5` first, or run both classes together).

### Latest change: Remove intermediate consolidation step

The consolidation phase (Phase 2 of `_fetch_vars`) has been removed.  Instead
of eagerly materialising each variable into an intermediate NetCDF file, the
raw temp files are opened lazily and the full dask graph is returned directly
to `cutout_prepare`.  See [Consolidation Removal](#consolidation-removal)
below for details and testing instructions.

**Quick start for a future contributor:**

```python
cutout = atlite.Cutout(
    path="my_cutout.nc",
    module="era5-ncar",   # only change from "era5"
    x=slice(-10, 5),
    y=slice(35, 44),
    time="2024",
)
cutout.prepare()   # fetches from NCAR THREDDS — no CDS queue
```

---

## Context

ERA5 data downloads via CDS (Copernicus Climate Data Store) can take >4 hours
due to queue times.  The same ERA5 dataset is available on NCAR's THREDDS
server.  This module fetches from there instead.

## Data Source

### NCAR RDA d633000 — chosen source

NCAR's Research Data Archive hosts ERA5 as dataset **d633000**.  Data is
CF-compliant NetCDF4 on a 0.25° global grid.  No account required.

**File organisation:**
- `e5.oper.an.sfc/{YYYYMM}/` — surface analysis (u10, v10, t2m, …).  One
  variable per file, one month per file.
- `e5.oper.fc.sfc.accumu/{YYYYMM}/` — forecast accumulated (ssrd, ro, …).
  Split into two ~15-day half-month files per month.
- `e5.oper.invariant/197901/` — time-invariant geopotential z.  Single file.

**Filename pattern:**
```
e5.oper.{product}.{table}_{paramid}_{shortname}.ll025sc.{YYYYMMDDHH}_{YYYYMMDDHH}.nc
```

**OPeNDAP endpoint:**
```
https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/
```

**THREDDS hard limit:** 500 MB per request.  One variable × one month ×
EU-scale bbox ≈ 124 MB — well within limit.

### Access method: OPeNDAP

OPeNDAP is the only viable option.  It provides efficient server-side spatial
subsetting for all variable types.

| Method | an.sfc | fc.sfc.accumu | Verdict |
|--------|--------|---------------|---------|
| **OPeNDAP** | ~124 MB/var/mo | ~124 MB/var/mo | **Use this** |
| S3 range reads | ~2-3 GB (ok) | ~35 GB (full-grid chunks) | Bad for forecast |
| OSDF/Zarr | ~3-5 GB | **Not available** | Missing vars |
| GDEX Subset API | ~124 MB/var/mo | ~124 MB/var/mo | Requires auth |

S3 forecast chunks are `(1, 12, 721, 1440)` — full global grid per init time.
Unusable for spatial subsets.

---

## Verified Findings (resolved during implementation)

These were open questions before Phase B; all are now answered.

### Variable names in NCAR NetCDF files

Not uniform — depends on the variable:

| Short name | NCAR variable name | Notes |
|------------|-------------------|-------|
| u10, v10   | `VAR_10U`, `VAR_10V` | VAR_ prefix + uppercase |
| u100, v100 | `VAR_100U`, `VAR_100V` | VAR_ prefix + uppercase |
| t2m, d2m   | `VAR_2T`, `VAR_2D` | VAR_ prefix + uppercase |
| fsr        | `FSR` | **No VAR_ prefix** |
| stl4       | `STL4` | **No VAR_ prefix** |
| ssrd, ssr, fdir, tisr, ro | `SSRD`, `SSR`, `FDIR`, `TISR`, `RO` | No prefix (all fc vars) |
| z          | `Z` | No prefix (invariant) |

Rule of thumb: analysis vars with a numeric level in the name get `VAR_` prefix;
others don't.  Always verify against the file rather than guessing.

### Verified parameter codes

```python
VAR_MAP = {
    "u10":  ("e5.oper.an.sfc",        "128_165_10u",  "VAR_10U"),
    "v10":  ("e5.oper.an.sfc",        "128_166_10v",  "VAR_10V"),
    "u100": ("e5.oper.an.sfc",        "228_246_100u", "VAR_100U"),
    "v100": ("e5.oper.an.sfc",        "228_247_100v", "VAR_100V"),
    "fsr":  ("e5.oper.an.sfc",        "128_244_fsr",  "FSR"),
    "t2m":  ("e5.oper.an.sfc",        "128_167_2t",   "VAR_2T"),
    "d2m":  ("e5.oper.an.sfc",        "128_168_2d",   "VAR_2D"),
    "stl4": ("e5.oper.an.sfc",        "128_236_stl4", "STL4"),
    "ssrd": ("e5.oper.fc.sfc.accumu", "128_169_ssrd", "SSRD"),
    "ssr":  ("e5.oper.fc.sfc.accumu", "128_176_ssr",  "SSR"),
    "fdir": ("e5.oper.fc.sfc.accumu", "228_021_fdir", "FDIR"),  # NOT 128_228
    "tisr": ("e5.oper.fc.sfc.accumu", "128_212_tisr", "TISR"),
    "ro":   ("e5.oper.fc.sfc.accumu", "128_205_ro",   "RO"),
    "z":    ("e5.oper.invariant",     "128_129_z",    "Z"),
}
```

Note: `fdir` is `228_021_fdir`, **not** `128_228_fdir` as one might guess.
`tp` (total precipitation) is not present in the NCAR catalog; use `ro` for
runoff as in the existing `era5` module.

### Invariant file path

```
e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc
```

The `197901/` subdirectory is required — it is not at the top level.

### Coordinates

- **Latitude**: N→S (90→−90, step=−0.25).  Use `slice(north, south)` with `.sel()`.
- **Longitude**: 0→360.  A bbox crossing the 0° meridian (e.g. x0=−4, x1=1.5)
  needs two `.sel()` calls:
  ```python
  high = ds.sel(longitude=slice(x0 % 360, 360))   # e.g. 356–360
  low  = ds.sel(longitude=slice(0, x1 % 360))      # e.g.   0–1.5
  high = high.assign_coords(longitude=high.longitude.values - 360)
  result = xr.concat([high[vars_with_lon], low[vars_with_lon]], dim="longitude")
  ```
  **Never call `.roll()` on a lazy pydap dataset** — it forces a full global
  download (4+ GB) and hits the THREDDS 500 MB limit with HTTP 403.

### Forecast accumulated convention — CRITICAL

Despite the `fc.sfc.accumu` product name, **NCAR stores per-forecast-hour
values, not running totals**.  Each `forecast_hour` entry already contains the
J/m² (or metres for runoff) for that single 1-hour period.  No differencing
(`diff()`) is needed.

- Divide by 3600 to convert J/m² → W/m² for radiation variables.
- Do **not** divide for runoff (units are metres).
- Timestamps: `actual_time = forecast_initial_time + forecast_hour × 1h`.
- Hours 00:00–06:00 of day 1 of each month come from the previous month's
  second-half file (init Dec 31 18:00 → hours 1–12 cover Jan 1 00:00–06:00).
  Always fetch `_fc_prev_half_url()` alongside the two half-month files.

### OPeNDAP engine

`netCDF4` is typically **not** compiled with OPeNDAP support in standard venvs.
Use `engine="pydap"` instead:
```python
xr.open_dataset(url, engine="pydap")
```
Install with `pip install pydap`.  Suppress the DAP2 deprecation warning with
`warnings.catch_warnings()`.

### Regridding

`xr.interp(method="linear")` matches CDS coarse-grid output within float32
precision.  Always fetch with 0.5° bbox padding so interpolation has support at
grid edges:
```python
x0_fetch = x0 - 0.5;  x1_fetch = x1 + 0.5
y0_fetch = y0 - 0.5;  y1_fetch = y1 + 0.5
```
Interpolate each raw variable (u100, v100, …) to the target grid **before**
computing derived quantities (wnd100m, etc.) for consistency with how CDS
handles coarse grids.

### Numerical accuracy vs CDS reference

All variables match to float32 precision or better (tested on BOUNDS=(−4, 56,
1.5, 62), TIME="2013-01-01"):

| Variable | Max abs diff | Notes |
|----------|-------------|-------|
| wnd100m | 0 | exact |
| wnd_shear_exp | 0 | exact |
| wnd_azimuth | 2.4e-7 rad | float32 rounding |
| roughness | 0 | exact |
| influx_direct | 0 | exact |
| influx_diffuse | 1.1e-5 W/m² | float32 rounding |
| influx_toa | 0 | exact |
| albedo | 9.7e-8 | float32 rounding |
| temperature | 6.1e-5 K | float32 rounding |
| soil temperature | 6.1e-5 K | float32 rounding |
| dewpoint temperature | 6.1e-5 K | float32 rounding |
| runoff | 0 | exact |
| height | 0 | exact |

---

## Consolidation Removal

### Problem

`_fetch_vars` had a two-phase architecture:
1. **Download** — ~200+ raw files fetched from OPeNDAP in parallel (pydap,
   thread-safe), written to temp NetCDF at native resolution.
2. **Consolidate** — raw files opened with netCDF4/HDF5 (not thread-safe),
   concatenated, interpolated to target grid, eagerly `.compute()`-ed, and
   written as ~13 consolidated NetCDF files.  All under a module-level
   `_nc_write_lock`, fully serialised.

The consolidation then re-opened those files lazily for the final
`cutout_prepare` write — an unnecessary round-trip through disk.

### Why consolidation was there (and why it's no longer needed)

The consolidation existed for two reasons:

1. **HDF5 thread safety** — the netCDF4/HDF5 C library crashes on concurrent
   reads from multiple dask threads.  The workaround was a global
   `threading.Lock()` around the entire Phase 2.

   **Solution**: xarray's `open_dataset` natively accepts a `lock` parameter
   that serialises individual chunk reads.  The caller (`cutout_prepare` in
   `data.py`) already creates a `dask.utils.SerializableLock()` and passes it
   to `get_data()` — era5_ncar was simply ignoring it.

2. **Dask graph complexity** — concern that ~200 nodes would be too many.

   **Not a real issue**: dask routinely handles thousands of graph nodes.  The
   overhead is negligible vs. I/O time.

### What changed in `era5_ncar.py`

**Removed:**
- `_nc_write_lock` (`threading.Lock`) — replaced by xarray's per-chunk lock
- `_CONSOLIDATED_CHUNKS` constant — no longer needed
- Entire consolidation block (~80 lines): the eager `.compute()`, intermediate
  NetCDF write/re-open cycle, and consolidated file caching

**Added/Changed:**
- `_fetch_vars(short_names, coords, tmpdir=None, lock=None)` — new `lock`
  parameter, passed to every `xr.open_dataset(..., lock=lock)` call
- `lock` threaded through: `get_data()` → `get_data_*()` → `_fetch_vars()`
- Phase 2 is now purely lazy: open files → concat → time select → spatial
  interp → return.  No materialisation.
- `get_data()` no-tmpdir fallback path: added `.load()` since temp files are
  deleted when the `TemporaryDirectory` context exits

**Unchanged:**
- Phase 1 (parallel OPeNDAP downloads) — identical
- Spatial interpolation logic (`_grids_align` / `interp` / `sel`) — still
  needed because OPeNDAP cannot do server-side regridding (unlike CDS which
  accepts a `"grid"` parameter).  The standard `era5.py` module avoids interp
  by requesting data at target resolution from CDS; era5_ncar downloads at
  native 0.25° and interpolates client-side.
- Raw file caching (deterministic filenames, resumable downloads)
- Forecast dedup, unit conversion (`/3600`), `_to_xy` coordinate rename
- `_retrieve_var` / `_retrieve_var_inner` download logic

**Net diff:** ~80 lines removed, ~0 lines of new logic.

### Why spatial interpolation stays

`cutout_prepare()` (in `data.py`) calls `xr.merge(datasets, compat="equals")`
after collecting results from all modules — it expects every module's
`get_data()` to return data already on the target grid.  There is **no
post-merge spatial alignment**.

The standard `era5.py` avoids client-side interp by passing
`"grid": f"{cutout.dx}/{cutout.dy}"` to the CDS API.  OPeNDAP has no
equivalent — data always comes back at native 0.25° resolution.  So
era5_ncar must interp to the target grid before returning.

In the common case (cutout at 0.25°), `_grids_align` returns `True` and
the code falls through to `sel(method="nearest")` — no actual interpolation,
just coordinate alignment.

### Testing instructions

The existing test suite covers this module end-to-end.  Tests hit the live
NCAR THREDDS server (no mocks), so they require network access.

**Test file:** `test/test_era5_ncar.py` (or similar — find with
`find . -name '*era5_ncar*' -path '*/test*'`)

**What to verify:**

1. **All existing tests still pass:**
   ```bash
   pytest test/test_era5_ncar.py -v
   ```
   The key tests exercise each feature (wind, influx, temperature, runoff,
   height) and compare against CDS reference values to float32 precision.

2. **No consolidation files are created:**
   After a test run, check the tmpdir (logged as `era5_ncar.log` in the temp
   directory).  There should be **no** `consolidated_*.nc` files — only the
   raw `era5_ncar_*.nc` download cache files.

3. **Lock is threaded correctly:**
   Verify that the `lock` parameter from `cutout_prepare` reaches
   `xr.open_dataset`.  A quick way: add a temporary `assert lock is not None`
   at the top of `_fetch_vars` and run a test that goes through
   `cutout.prepare()`.

4. **No HDF5 crashes under concurrency:**
   The most important thing to verify.  If the lock isn't working, you'll see
   segfaults or "double free" errors from the HDF5 C library.  Run the full
   test suite multiple times:
   ```bash
   for i in $(seq 5); do pytest test/test_era5_ncar.py -v || break; done
   ```

5. **No-tmpdir fallback still works:**
   Test the direct-call path (no `cutout.prepare()`):
   ```python
   from atlite.datasets.era5_ncar import get_data_height
   # Should work without tmpdir — uses TemporaryDirectory internally
   # Data must be eagerly loaded (.load()) since temp files are cleaned up
   ```

6. **Numerical accuracy unchanged:**
   The `test_compare_with_era5` test (if present) validates that NCAR output
   matches CDS output to float32 precision.  This should still pass since the
   interp logic is unchanged.

### Architecture after this change

```
cutout.prepare()
  └─ get_features()                    # data.py — creates SerializableLock
       └─ get_data(lock=lock)          # era5_ncar.py
            └─ get_data_wind(lock=lock)
                 └─ _fetch_vars(lock=lock)
                      ├─ Phase 1: parallel OPeNDAP downloads → raw .nc files
                      └─ Phase 2: open_dataset(lock=lock) → lazy dask graph
  └─ ds.to_netcdf(compute=False)       # dask writes final cutout
       └─ dask scheduler reads chunks through the lock (HDF5-safe)
```

