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
_retrieve_analysis(short_name, year, month, x0, y0, x1, y1)
_retrieve_forecast_month(...)          Fetches prev + both halves; deduplicates
_retrieve_height(x0, y0, x1, y1)
_collect_analysis(short_name, coords, x0, y0, x1, y1)   Loops over months
_collect_forecast(short_name, coords, x0, y0, x1, y1, divide_by_3600)
_interp(da, coords)                    xr.interp to target x/y grid
get_data_wind(coords)
get_data_influx(coords)
get_data_temperature(coords)
get_data_runoff(coords)
get_data_height(coords)
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

The current implementation is sequential: one OPeNDAP request at a time.  For a
full year × all features, this makes ~276 requests.  Each request takes 5–30 s
depending on server load.  Total wall time: 20–90 minutes for a year.

Potential speedups (not yet implemented):
- **Concurrent requests**: use `concurrent.futures.ThreadPoolExecutor` to
  parallelise across variables or months.  The `monthly_requests` +
  `concurrent_requests` pattern from `era5.get_data()` would be a natural fit.
- **Dask integration**: open multiple OPeNDAP datasets lazily and let dask
  schedule the downloads.  Requires care to stay within the 500 MB/request limit.

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
