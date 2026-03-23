<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: MIT
-->

# era5-ncar Dataset Module for Atlite — Implementation Plan

## Status: COMPLETE

All phases (A–E) are implemented and tested.  15/16 tests pass on the first run
(the 16th, `test_compare_with_era5`, requires the CDS reference cutout to also
be cached — run `TestERA5` first, or run both classes together).

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

## Implementation

### Files created/modified

| File | Action |
|------|--------|
| `atlite/datasets/era5_ncar.py` | **Created**: new dataset module (~270 lines) |
| `atlite/datasets/__init__.py` | Added import + registered as `"era5-ncar"` |
| `test/conftest.py` | Added `THREDDS_AVAILABLE` probe + 2 fixtures |
| `test/test_preparation_and_conversion.py` | Added `TestERA5NCAR` (16 tests) |
| `scripts/dataset_comparison.py` | Exploration + verification script |
| `atlite/datasets/era5.py` | **No changes** |
| `pyproject.toml` | **No changes** |

### Exploration script

`scripts/dataset_comparison.py` has four modes:

```
python scripts/dataset_comparison.py            # Phase B: metadata exploration
python scripts/dataset_comparison.py --phase-c  # Phase C: full feature comparison
python scripts/dataset_comparison.py --phase-d  # Phase D: regridding test
python scripts/dataset_comparison.py --influx   # influx section only (fast debug)
```

Requires internet access and `test-cache/cutout_era5.nc` (for phases C/D).

### Module architecture (`atlite/datasets/era5_ncar.py`)

Key internal functions:

```
_open_opendap(path)                    Opens OPeNDAP URL with pydap engine
_sel_bbox(ds, x0, y0, x1, y1)         Spatial subset; handles 0-meridian wrap
_to_xy(da)                             latitude/longitude → y/x; round coords
_an_sfc_url(product_dir, param_code, year, month)
_fc_half_urls(product_dir, param_code, year, month)   → [first_half, second_half]
_fc_prev_half_url(product_dir, param_code, year, month)
_fc_to_hourly(subset, ncar_varname)    Flatten (init, hour, lat, lon) → (time, lat, lon)
_retrieve_var(short_name, year, month, x0, y0, x1, y1)
                                       Unified fetch: dispatches to an.sfc / fc.sfc.accumu
                                       / invariant path; inlines all download logic
_fetch_vars(short_names, coords)       Submits all (var, month) tasks concurrently via
                                       ThreadPoolExecutor; assembles + interpolates results
_interp(da, coords)                    xr.interp to target x/y grid
get_data_wind(coords)                  Pure assembler: calls _fetch_vars, computes derived
get_data_influx(coords)                Pure assembler: calls _fetch_vars, computes derived
get_data_temperature(coords)           Pure assembler: calls _fetch_vars, computes derived
get_data_runoff(coords)                Pure assembler: calls _fetch_vars, computes derived
get_data_height(coords)                Pure assembler: calls _fetch_vars, computes derived
get_data(cutout, feature, ...)         Entry point; same signature as era5.get_data()
```

### Testing

```bash
# Run era5-ncar tests (downloads from NCAR, ~8 min for one day)
python -m pytest test/test_preparation_and_conversion.py::TestERA5NCAR -v --cache-path=test-cache

# Run comparison test (needs CDS cutout cached too)
python -m pytest test/test_preparation_and_conversion.py::TestERA5 \
                 test/test_preparation_and_conversion.py::TestERA5NCAR \
                 -v --cache-path=test-cache
```

---

## Known issues and future work

### Performance

Concurrent downloads are implemented via `ThreadPoolExecutor` (default
`MAX_WORKERS = 8`).  All `(short_name, year, month)` tasks for a feature are
submitted to the pool simultaneously; results are assembled once all futures
complete.  For a full year × all features this gives ~8× wall-time reduction
over the previous sequential implementation.

### THREDDS reliability

THREDDS can be slow or return 500 errors under load.  The module has no retry
logic.  Consider wrapping `_open_opendap()` with `tenacity` retries for
production use.

### `test_compare_with_era5` skip

This test requires both `cutout_era5` (CDS) and `cutout_era5_ncar` to be
cached.  If running `TestERA5NCAR` in isolation with `--cache-path`, the CDS
fixture may not be available.  Run `TestERA5` first (or together) to populate
the CDS cache.

### GDEX Subset API as fallback

If THREDDS becomes unreliable, GDEX (`https://gdex.ucar.edu/api/`) provides the
same data with async queuing.  It requires a free ORCID-based auth token.  A
GDEX backend could be added as `module="era5-ncar-gdex"` following the same
module pattern.

---

## Phase F — Code quality fixes

Seven issues identified by code review, ordered by implementation dependency.
Changes are confined to `atlite/datasets/era5_ncar.py` unless noted.

### F1. Context managers for OPeNDAP datasets (high priority)

**Problem:** `_open_opendap` returns a dataset that is later closed with an
explicit `ds.close()`.  If anything between open and close raises, the
connection/file-descriptor leaks.

**Fix:** In `_retrieve_var`, wrap every `_open_opendap` call in a `with`
statement.  `xr.Dataset` supports the context-manager protocol, so this is a
drop-in change:

```python
# invariant branch
with _open_opendap(_INVARIANT_PATH) as ds:
    subset = _sel_bbox(ds, x0, y0, x1, y1)
    z = subset["Z"].isel(time=0, drop=True).load()
    return _to_xy(z / 9.80665)

# an.sfc branch
with _open_opendap(path) as ds:
    subset = _sel_bbox(ds, x0, y0, x1, y1)
    da = _to_xy(subset[ncar_var].load())
    return da

# fc.sfc.accumu branch
for url in urls:
    with _open_opendap(url) as ds:
        subset = _sel_bbox(ds, x0, y0, x1, y1)
        parts.append(_fc_to_hourly(subset, ncar_var))
```

Remove all bare `ds.close()` calls.

### F2. Remove redundant sorts in forecast path (low priority, do alongside F1)

**Problem:** `_fc_to_hourly` sorts by time (`.sortby("time")`), then
`_retrieve_var` sorts the concatenated result again, then `np.unique`
implicitly sorts a third time.

**Fix:**
- Remove `.sortby("time")` from the end of `_fc_to_hourly`.
- Remove `.sortby("time")` from the `xr.concat` line in `_retrieve_var`.
- The `np.unique` call already produces sorted indices; that alone is
  sufficient.

### F3. Add tolerance to `method="nearest"` time selection (medium priority)

**Problem:** In `_fetch_vars`, `da.sel(time=mt, method="nearest")` silently
snaps to the closest available timestamp if there's a mismatch.  At month
boundaries this could mask off-by-one-hour bugs.

**Fix:** Add `tolerance="30min"` to the `.sel()` call:

```python
sel = da.sel(time=mt, method="nearest", tolerance="30min")
```

This still handles float-precision rounding in timestamps but raises
`KeyError` if a requested time is genuinely missing (> 30 min gap), making
boundary bugs visible immediately.

### F4. Reduce MAX_WORKERS and move retry to `_open_opendap` (medium priority)

**Problem:** `MAX_WORKERS = 16` can overload THREDDS.  The `@retry` decorator
is on `_retrieve_var`, so a failure on the 3rd forecast URL retries all 3 URLs
from scratch.

**Fix (two parts):**

1. **Reduce `MAX_WORKERS` to 8** (matches the original plan value).

2. **Move `@retry` from `_retrieve_var` to `_open_opendap`**, so each
   individual OPeNDAP open is retried independently:

```python
@retry(
    wait=wait_random_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _open_opendap(path):
    ...
```

Remove the `@retry` from `_retrieve_var`.  This means a transient 500 on one
forecast half-file retries just that file, not the entire variable+month.

### F5. Parallelize forecast half-file fetches (low priority)

**Problem:** Each forecast `_retrieve_var` call opens 3 URLs serially
(prev-half + 2 halves).  The thread pool parallelizes across `(var, month)`
combinations, but within one combination there are 3 serial round-trips.

**Fix:** Restructure so each forecast half-file is its own task submitted to
the pool.  In `_fetch_vars`, expand the task list:

```python
for sn in short_names:
    product_dir, param_code, ncar_var = VAR_MAP[sn]
    if product_dir == "e5.oper.invariant":
        tasks.append(("invariant", sn, None, None, None))
    elif product_dir == "e5.oper.an.sfc":
        for year, month in months:
            tasks.append(("an_sfc", sn, year, month, None))
    else:
        for year, month in months:
            urls = [
                _fc_prev_half_url(product_dir, param_code, year, month),
                *_fc_half_urls(product_dir, param_code, year, month),
            ]
            for url in urls:
                tasks.append(("fc_half", sn, year, month, url))
```

Add a thin `_retrieve_fc_half(url, ncar_var, x0, y0, x1, y1)` that opens one
URL and returns the hourly DataArray.  The assembly step in `_fetch_vars` then
concatenates + deduplicates per `(sn, year, month)` as before.

This replaces `_retrieve_var`'s forecast branch — the function becomes simpler
(only handles invariant and an.sfc) or is split into three small helpers.

### F6. Batch interpolation per feature (low priority)

**Problem:** `_interp` is called once per variable.  Since all variables for a
feature share the same source and target grids, xarray recomputes grid weights
each time.

**Fix:** In `_fetch_vars`, after assembling all per-variable DataArrays into a
dict, combine them into a single `xr.Dataset` and call `.interp()` once:

```python
# Replace the per-variable _interp calls with:
ds_raw = xr.Dataset({sn: da for sn, da in assembled_raw.items()})
ds_interp = ds_raw.interp(x=coords["x"].values, y=coords["y"].values, method="linear")
return {sn: ds_interp[sn] for sn in short_names}
```

Invariant (height) vars have no time dim, so handle them separately — either
interpolate them first, or split the dataset into time-varying and static
groups.

### F7. Lazy loading with dask instead of eager `.load()` (high priority)

**Problem:** Every variable is `.load()`ed into memory at native 0.25°
resolution inside `_retrieve_var`.  For a full-year, continent-scale cutout
with all features, this can exhaust RAM.  The era5 module uses dask lazy
arrays and chunked disk I/O.

**Fix:** This is the largest change and should be done last since it touches
the data flow throughout the module.

1. **Remove `.load()` calls** in `_retrieve_var`.  Let data remain lazy
   (pydap-backed).

2. **Problem:** pydap datasets cannot be used across threads (the HTTP session
   is not thread-safe for lazy access).  Two options:
   - **(a) Keep threads, load to disk:** Each thread downloads its chunk to a
     temporary NetCDF file (using `da.to_netcdf(tmpfile)`), then re-opens it
     lazily with `xr.open_dataarray(tmpfile, chunks={})`.  This mirrors the
     era5 module's pattern (download → tmp file → lazy open).
   - **(b) Sequential lazy open:** Drop threading, open all pydap URLs
     sequentially (metadata only), let xarray/dask handle the parallel
     compute when data is actually needed.  Simpler but loses the download
     parallelism.

   Option (a) is recommended — it preserves the existing download parallelism
   while keeping memory bounded.

3. **Accept `tmpdir` in `get_data`:** The `tmpdir` parameter (already in the
   function signature but currently unused) becomes the location for
   intermediate files.  If `None`, fall back to `tempfile.mkdtemp()`.

4. **Wire up dask chunks:** After re-opening from disk, ensure DataArrays
   have chunks (e.g. `chunks={"time": 24}`) so downstream operations remain
   lazy.

5. **Test with large cutout:** Verify memory stays bounded for a full-year
   European cutout.  Compare numerical output against the eager implementation
   to ensure no regressions.

**Note:** This can be deferred if the module is only used for small/medium
cutouts in the near term.  The other fixes (F1–F6) are all independent of
this one.

### F8. Subset prev-half forecast files before loading (high priority)

**Problem:** Each forecast variable-month fetches the *full* previous
half-month file (~15 days of data at ~65 MB for Europe) just to use ~6 hours
that spill into the target month.  `_fc_to_hourly` calls `.load()` on the
entire spatially-subsetted file without first subsetting on
`forecast_initial_time`.

For a full-year European cutout this adds ~3.8 GB of unnecessary downloads
(5 fc vars × 12 months × ~63 MB wasted per prev-half) and 60 OPeNDAP
requests that each transfer ~25× more data than needed.

**Background:** Forecast files have dimensions
`(forecast_initial_time, forecast_hour, latitude, longitude)`.  Init times
are every 6 hours (00, 06, 12, 18 UTC) with 12 forecast hours each.  Only
the last 1–2 init times of the prev-half file produce hours that fall in the
target month (e.g. Dec 31 18:00 → forecast hours 7–12 cover Jan 1 00:00–
05:00).

**Fix:** In the forecast branch of `_retrieve_var` (or the new
`_retrieve_fc_half` from F5), subset `forecast_initial_time` on the prev-half
dataset before loading.  Only the init times whose forecast hours can reach
into the target month need to be kept:

```python
def _subset_fc_for_month(ds, year, month, is_prev_half):
    """Subset forecast_initial_time to only inits relevant to (year, month).

    For the prev-half file, keep only the last 2 init times (the ones whose
    forecast hours spill into the target month).  For the target month's own
    halves, keep everything (all inits produce target-month hours).
    """
    if not is_prev_half:
        return ds
    # Last 2 inits (12:00 and 18:00 of last day) is conservative;
    # only the 18:00 init is strictly needed for the 00–05 spill,
    # but keeping 2 is cheap insurance.
    init_times = ds["forecast_initial_time"].values
    return ds.isel(forecast_initial_time=slice(-2, None))
```

Call this between `_sel_bbox` and `_fc_to_hourly`:

```python
for i, url in enumerate(urls):
    with _open_opendap(url) as ds:
        subset = _sel_bbox(ds, x0, y0, x1, y1)
        subset = _subset_fc_for_month(subset, year, month, is_prev_half=(i == 0))
        parts.append(_fc_to_hourly(subset, ncar_var))
```

The `.isel()` is applied before `.load()` inside `_fc_to_hourly`, so OPeNDAP
only transfers the selected init times — server-side subsetting.

**Impact:** Reduces prev-half downloads from ~65 MB to ~2–4 MB each.  Total
bandwidth savings: ~3.8 GB for a full-year European cutout.  Also reduces
memory pressure (relevant to F7).

### Implementation order

```
F1 (context managers) + F2 (remove sorts)   — small, independent, do first
F3 (time tolerance)                          — small, independent
F4 (retry + workers)                         — small, independent
F8 (subset prev-half)                        — small, independent, big bandwidth win
F5 (parallelize fc halves)                   — medium, refactors _retrieve_var
F6 (batch interp)                            — small, after F5 stabilizes
F7 (lazy/dask loading)                       — large, do last
```

F1–F4 and F8 can be done in any order (or in parallel).  F8 is a small,
localised change with the largest bandwidth impact — it should be prioritised
alongside F1.  F5 restructures `_retrieve_var`, so F1/F2/F8 should land first
to avoid merge conflicts.  F6 is a small change to `_fetch_vars` that's
independent of F5 but benefits from a stable API.  F7 is a larger
architectural change that should come after everything else is solid.
