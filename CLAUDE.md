# Atlite — Developer Guide

Atlite is a Python library that converts weather data (ERA5, SARAH, NCEP) into
power-systems inputs: capacity factors for wind/solar, heat demand, runoff, etc.
This file covers everything an agent needs to work effectively in this repo.

---

## Setup

```bash
uv sync --extra dev   # installs atlite + dev deps (pytest, ruff, etc.)
```

`cfgrib` (needed for ERA5 GRIB decoding in the standard `era5` module) requires
the ecCodes C library.  It is **not** installed system-wide in this environment
but is available at runtime via `ecmwflibs`:

```bash
uv run --with ecmwflibs cfgrib pytest ...   # ERA5/GRIB tests only
```

Do **not** add `ecmwflibs` to `pyproject.toml`.

---

## Running tests

### ERA5-NCAR tests (the module being developed here)

```bash
uv run pytest test/test_preparation_and_conversion.py::TestERA5NCAR -v --cache-path ./tmp
```

- `--cache-path ./tmp` reuses already-prepared cutout `.nc` files and raw
  download cache files across runs.  Without it, every run re-downloads from
  NCAR THREDDS (~15 min for a single day over Scotland).
- `./tmp/` must exist and be populated before tests can run from cache.
  It is **not committed to git** — `.nc` files are too large.  Populate it
  by running the tests once against a live THREDDS connection:
  ```bash
  uv run pytest test/test_preparation_and_conversion.py::TestERA5NCAR -v --cache-path ./tmp
  ```
  This writes `./tmp/cutout_era5_ncar.nc` (the prepared cutout) and
  `./tmp/era5_ncar_*.nc` (raw per-variable downloads).  All subsequent
  runs with `--cache-path ./tmp` reuse these files and complete in ~2 s
  with no network access.
- Tests hit the live NCAR THREDDS server (no mocks).  They are skipped
  automatically if THREDDS is unreachable and no cached cutout exists.
- All 16 tests pass in ~2 s from cache.

### ERA5 (CDS) tests

```bash
uv run pytest test/test_preparation_and_conversion.py::TestERA5 -v --cache-path ./tmp
```

Requires `CDSAPI_URL` env var **or** a cached cutout at `./tmp/cutout_era5.nc`.
`test_compare_with_era5` (in `TestERA5NCAR`) also needs this file.

### Full test suite (non-slow)

```bash
uv run pytest -m "not slow" --cache-path ./tmp
```

SARAH and GEBCO tests are skipped unless the data paths exist.

---

## Scale tests (scripts)

### Germany — full year 2013

```bash
uv run python scripts/profile_germany.py
```

- Writes output to `germany_2013_profile.nc`, cache in `./tmp2/`.
- Expected: ~7 min total (incl. download), 338 MB output.

### Europe — full year 2013

```bash
uv run python scripts/download_europe_2013.py --tmpdir ./tmp_europe
```

- Grid: 281×153, 8760 h.  Output: `europe_2013_ncar.nc` (~10.9 GB on disk).
- Download: ~52 min.  Processing (dask compute + write, excl. download): ~12.4 min.
- **Performance target:** processing must complete in <45 min.

---

## Codebase structure

```
atlite/
  cutout.py          # Cutout class — user-facing entry point
  data.py            # cutout_prepare() + get_features() — orchestration layer
  datasets/
    era5.py          # Standard CDS/CDSAPI ERA5 module
    era5_ncar.py     # NCAR THREDDS/OPeNDAP ERA5 module (this work)
    sarah.py         # SARAH solar irradiance module
    gebco.py         # GEBCO bathymetry module
    ncep.py          # NCEP reanalysis module
  convert.py         # Feature → capacity factor conversions (wind, pv, hydro…)
  wind.py / pv/      # Technology-specific calculations
test/
  conftest.py        # Fixtures for all test cutouts; defines NCAR_TMPDIR=./tmp
  test_preparation_and_conversion.py  # Main test file
scripts/
  profile_germany.py       # Germany-scale timing benchmark
  download_europe_2013.py  # Europe-scale end-to-end test
tmp/                 # Persistent cache: raw downloads + prepared cutouts
tmp2/                # Germany script cache
tmp_europe/          # Europe script cache
```

---

## Data flow through the codebase

```
cutout.prepare(features=[...], tmpdir="./tmp")
  └── data.cutout_prepare()          # decorated with @maybe_remove_tmpdir
        └── data.get_features()
              ├── creates SerializableLock
              ├── for each feature: delayed(get_data)(cutout, feature, lock=lock)
              └── dask.compute(*delayed_calls)   ← runs all features in parallel
                    └── era5_ncar.get_data()
                          └── get_data_wind / _influx / _temperature / _runoff / _height
                                └── _fetch_vars(short_names, coords, tmpdir, lock)
                                      ├── Phase 1: parallel OPeNDAP downloads
                                      │     ThreadPoolExecutor(max_workers=8)
                                      │     → writes era5_ncar_*.nc to tmpdir
                                      └── Phase 2: lazy dask assembly
                                            xr.open_dataset(path, chunks={"time": 720})
                                            concat → time-sel → spatial interp/sel
                                            returns lazy DataArrays (no compute)
        └── ds.to_netcdf(tmp, compute=False).compute()   ← materializes dask graph
```

Key invariants:
- Phase 1 completes fully (all downloads) before Phase 2 starts.
- Phase 2 returns **lazy** arrays; no materialisation until `to_netcdf`.
- Phase 2 `xr.open_dataset` calls are serialised by `_nc4_open_lock` (a
  module-level `threading.Lock` in `era5_ncar.py`).  This is required because
  dask runs one `get_data` thread per feature concurrently; from cache all
  features reach Phase 2 simultaneously, causing concurrent `H5Fopen` calls
  that corrupt HDF5's global state.  `NETCDF4_PYTHON_LOCK` only protects chunk
  reads (via `_getitem`), not the initial file open in `NetCDF4DataStore.__init__`.
  `HDF5_LOCK` cannot be used as the outer wrapper because `xr.open_dataset`
  internally acquires it for coordinate reads, which would deadlock.
- Chunk reads during `to_netcdf` are serialised by xarray's built-in
  `NETCDF4_PYTHON_LOCK = CombinedLock([HDF5_LOCK, NETCDFC_LOCK])`.
- Raw download files are cached by deterministic filename in `tmpdir`; re-runs
  with the same `tmpdir` skip already-downloaded files.

---

## era5_ncar.py specifics

### Adding a new variable

1. Add to `VAR_MAP`: `"shortname": ("product_dir", "param_code", "NCAR_VARNAME")`.
2. Add to `_FEATURE_VARS[feature]`.
3. If it's a forecast radiation var (J/m²), add to `_FC_DIVIDE_BY_3600`.
4. Use the variable in the appropriate `get_data_*` function.

### Coordinate conventions

- NCAR files: latitude N→S (90 → −90), longitude 0→360.
- Internal convention after `_to_xy`: renamed to `y`/`x`, sorted S→N.
- Bboxes crossing 0° meridian need two `.sel()` calls — never `.roll()` on
  lazy pydap datasets (triggers a full global download, hits THREDDS 500 MB limit).

### Forecast files

- Product: `e5.oper.fc.sfc.accumu`.  Three half-month files per month:
  previous-month second half + two halves of the current month.
- Despite the "accumu" name, values are **per-forecast-hour** (not running totals).
  No differencing needed.  Divide by 3600 for J/m² → W/m² (radiation only).
- `_fc_to_hourly()` flattens `(n_init, n_hour, lat, lon)` → `(time, lat, lon)`.

### Chunking

Phase 2 opens temp files with `chunks={"time": 720}`.  Monthly an.sfc files
contain 720–744 h, so this gives roughly one dask chunk per source file.
For forecast variables, multiple half-month files are concatenated before
chunking, so each month ends up as 1–2 chunks.

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `HTTP 403` from THREDDS | `.roll()` called on lazy pydap dataset → full global download → 500 MB limit | Never call `.roll()`; use two `.sel()` calls for cross-meridian bboxes |
| HDF5 heap corruption / `corrupted size vs. prev_size` on cache rebuild | Concurrent `H5Fopen` calls in Phase 2: dask runs one `get_data` thread per feature; from cache all reach Phase 2 simultaneously | Fixed by `_nc4_open_lock` in `era5_ncar.py` — serialises `xr.open_dataset` calls across threads |
| HDF5 chunk-read crashes | Concurrent HDF5 reads without serialisation | Ensured by xarray's `NETCDF4_PYTHON_LOCK` — check it isn't being bypassed |
| `test_compare_with_era5` skipped/fails | `./tmp/cutout_era5.nc` missing | Run `TestERA5` first, or use `--cache-path ./tmp` with a pre-populated cache |
| Stale `.nc` assertion failures | Old files in `tmp/` for a different bbox | Delete bbox-specific files matching the variable + hash |

---

## Performance notes

- **Chunk size**: `{"time": 720}` balances dask graph size vs. memory.
  Larger chunks reduce graph overhead at Europe scale but require more RAM.
- **complevel**: atlite's default is 9 (small file, slow write); the scripts
  use `complevel=1` (fast write, ~20% larger file).  For production cutouts
  the default is fine; for iterative development or benchmarking use 1.
- **MAX_WORKERS = 8**: caps concurrent THREDDS connections.  More workers
  increase download parallelism but risk 429/503 responses from the server.
