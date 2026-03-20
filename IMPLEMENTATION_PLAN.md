<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: MIT
-->

# ARCO-ERA5 Dataset Module for Atlite — Implementation Plan

## Context

ERA5 data downloads via CDS (Copernicus Climate Data Store) can take >4 hours due to queue times. The same ERA5 dataset is available on Google Cloud as **ARCO-ERA5** (`gs://gcp-public-data-arco-era5/`), accessible instantly via `xarray.open_zarr()` with anonymous access. We'll create a new atlite dataset module that fetches ERA5 data from a cloud source instead of CDS.

## Data Source Investigation (TODO — resolve before Phase B)

### Problem: global chunking makes spatial subsetting impossible

All investigated cloud ERA5 sources chunk as **one full global field per timestep** — `(1, 721, 1440)` for surface variables (~4 MB/chunk). You cannot read a sub-chunk. For a single-country cutout needing a year of 12 variables, this means **~210 GB downloaded** when only ~1 GB is needed. Zarr v3 sharding (sub-chunk byte-range reads) would fix this, but no existing ERA5 store uses it, and we can't retroactively add it to someone else's dataset.

### Sources investigated

| Source | Chunking | Spatial subsetting? | Status |
|--------|----------|-------------------|--------|
| **ARCO-ERA5 `ar/` Zarr** (GCS) | `(1, 721, 1440)` surface, `(1, 721, 1440, 37)` pressure | No — full globe per chunk | Active, anonymous |
| **ARCO-ERA5 `co/` Zarr** (GCS) | `(1, 542080)` flattened Gaussian grid | No — plus needs regridding | Active, anonymous |
| **NSF NCAR ERA5 NetCDF-4** (`s3://nsf-ncar-era5`) | `(1, 721, 1440)` HDF5 chunks | No — same full-globe chunks | Active, anonymous, no egress costs |
| **Old `era5-pds`** (`s3://era5-pds`, us-east-1) | `(24, 100, 100)` — **spatially tiled!** | **Yes** — ~960 KB/chunk, ~8 lat × 15 lon tiles | Deprecated (no new data), may still be accessible |
| **Earthmover Zarr** (Arraylake) | Temporal group: `(8736, 12, 12)` | **Yes** — tiny spatial tiles | Commercial product |

### Candidates to investigate further

**1. `era5-pds` bucket (best if still alive)**
- Only free source with spatial chunking `(24, 100, 100)` — ideal for bounding-box reads
- "Deprecated" may just mean no new data added, not deleted
- **Must verify:** Is it still accessible? What time range? Does it have all atlite variables (u100, v100, fsr, ssrd, tisr, fdir, runoff, stl4, geopotential)?
- If it covers our time range and has the variables, this is the best option by far

**2. NCAR OPeNDAP/THREDDS server-side subsetting**
- NCAR typically serves datasets via THREDDS, which does server-side spatial extraction — chunk layout becomes irrelevant
- **Must verify:** Does NCAR expose d633000 via OPeNDAP? What's the endpoint? Performance for large time ranges?

**3. NCAR S3 NetCDF-4 with kerchunk**
- Kerchunk creates a JSON sidecar mapping HDF5 chunk byte-offsets, enabling xarray to fetch individual chunks via HTTP range requests
- Won't reduce bandwidth (chunks are still full-globe), but avoids downloading entire monthly files
- Useful if combined with option 2 as fallback

**4. Accept ~210 GB from ARCO-ERA5 or NCAR S3**
- Still much faster than 4+ hour CDS queue (~35 min at 100 MB/s)
- Viable for users with good bandwidth, problematic on metered connections
- All 12 atlite variables confirmed available in both sources

### Variable availability (confirmed for NCAR and ARCO)

All atlite-required variables are present in both NCAR S3 and ARCO-ERA5:
- **Analysis surface** (`e5.oper.an.sfc`): u10, v10, t2m, d2m, stl4
- **Forecast instantaneous** (`e5.oper.fc.sfc.instagg`): u100, v100, fsr
- **Forecast accumulated** (`e5.oper.fc.sfc.accumu`): ssr, ssrd, tisr, fdir, runoff
- **Invariant**: geopotential

## How ERA5 Data Download Currently Works

1. `cutout.prepare()` → `cutout_prepare()` in `data.py:132`
2. For each feature, calls `era5.get_data(cutout, feature, ...)` via `data.py:45`
3. `era5.get_data()` (`era5.py:520`) dispatches to `get_data_wind()`, `get_data_influx()`, etc.
4. Each `get_data_*()` calls `retrieve_data()` (`era5.py:432`) → CDS API → GRIB → xarray
5. After retrieval, applies processing: coord rename, feature-specific math, sanitization
6. Results concatenated across time, merged across features, written to NetCDF

**Caching**: All-or-nothing at the `.nc` file level. `prepared_features` attr tracks what's done. No intermediate download caching.

## ARCO-ERA5 Data

**Target store**: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- 0.25° regular lat/lon grid (same as what CDS returns with `grid: "0.25/0.25"`)
- Contains all surface + 37 pressure-level variables
- Coordinates: `latitude`, `longitude`, `time` (+ `level` for pressure vars)
- Longitude: **0–360** (needs conversion to -180–180)
- Chunking: `{'time': 1, 'latitude': 721, 'longitude': 1440, 'level': 37}`
- Anonymous access: `storage_options=dict(token='anon')`
- Variable names: **needs runtime verification** (README shows both long and short names)

The `co/` tier uses short GRIB names but a **reduced Gaussian grid** (542k flattened `values` — unusable without regridding). The `ar/` tier has the regular 0.25° grid we need.

Note: CDS does server-side regridding from ERA5's native Gaussian grid. The `ar/` tier is pre-regridded to 0.25° — equivalent to what CDS returns at default resolution.

## Setup
You'll need the `ecmwflibs` python package to run grib conversion files. install it in system python, do not edit pyproject.toml.

## Implementation Sequence

Incremental approach: standalone script first, then integrate into atlite.

### Phase A: Reference data

**Step 1**: Provide a cached CDS cutout from running existing tests with `--cache-path`.
This gives us `cutout_era5.nc` (BOUNDS=(-4, 56, 1.5, 62), TIME="2013-01-01") as ground truth.
Also provide `cutout_era5_coarse.nc` (dx=0.5, dy=0.7) for regridding comparison.

### Phase B: Explore ARCO-ERA5 in a standalone script

**Step 2**: Write `scripts/arco_explore.py` — open the ARCO zarr store, print variable names, coordinates, dimensions. Confirm:
- Variable naming convention (short vs long)
- Latitude ordering (N→S or S→N)
- Longitude range (0-360)
- Whether surface vars have a `level` dimension
- Data types (float32/float64)

**Step 3**: Extend script — subset to the same BOUNDS and TIME as the test cutout. Confirm we can extract the right spatial/temporal region. Handle longitude 0-360 → -180-180 conversion.

**Step 4**: Extend script — select the wind variables (u10, v10, u100, v100, fsr or equivalent), rename to match cfgrib short names if needed, rename coords to x/y. Print and visually compare a few values against `cutout_era5.nc`.

### Phase C: Feature-by-feature processing in the script

**Step 5**: Implement wind processing in the script (magnitude, shear, azimuth). Load the cached ERA5 cutout, compare `wnd100m`, `roughness` etc. with `xr.testing.assert_allclose`. This tells us the tolerance we can expect.

**Step 6**: Implement influx processing (albedo, diffuse, J→W conversion, SolarPosition). Compare against ERA5 cutout.

**Step 7**: Implement temperature, runoff, height. Compare. At this point we know the full variable mapping and any quirks (accumulation conventions, missing values, etc.).

### Phase D: Regridding

**Step 8**: Test `xr.interp()` for non-0.25° resolution. Compare against the cached `cutout_era5_coarse.nc` (dx=0.5, dy=0.7) to check that our interpolation matches CDS regridding within tolerance.

### Phase E: Integrate into atlite

**Step 9**: Create `atlite/datasets/arco_era5.py` — port the working script into the module interface (`get_data()`, `get_data_wind()`, etc.). Register in `__init__.py`. Add deps to `pyproject.toml`.

**Step 10**: Add `cutout_arco_era5` fixture to `conftest.py`. Add `TestARCOERA5` class to `test_preparation_and_conversion.py` — reuse existing test helpers + add comparison test.

**Step 11**: Run the full test suite (`pytest --cache-path=...`) to verify everything passes.

---

## Design Details

### Module structure: `atlite/datasets/arco_era5.py` (~200 lines)

**Leave `era5.py` completely untouched.**

**Module-level constants** (same as era5.py):
```python
crs = 4326
features = {
    "height": ["height"],
    "wind": ["wnd100m", "wnd_shear_exp", "wnd_azimuth", "roughness"],
    "influx": ["influx_toa", "influx_direct", "influx_diffuse", "albedo",
               "solar_altitude", "solar_azimuth"],
    "temperature": ["temperature", "soil temperature", "dewpoint temperature"],
    "runoff": ["runoff"],
}
static_features = {"height"}
ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
```

**Imports from era5.py** (no modification needed):
```python
from atlite.datasets.era5 import (
    _add_height,        # geopotential → height conversion
    sanitize_wind,      # roughness clipping
    sanitize_influx,    # flux clipping
    sanitize_runoff,    # runoff clipping
)
from atlite.pv.solar_position import SolarPosition
```

**Core retrieval** — replaces `era5.retrieve_data()`:
```python
def retrieve_data(coords, variables, chunks=None, dx=0.25, dy=0.25):
    """Open ARCO-ERA5 zarr, subset spatially/temporally, select variables."""
    ds = xr.open_zarr(ZARR_URL, chunks=chunks, storage_options=dict(token="anon"))

    # Variable name mapping (cfgrib short name → ARCO name if different)
    arco_vars = [VAR_MAPPING.get(v, v) for v in variables]
    ds = ds[arco_vars]

    # Longitude 0-360 → -180-180
    ds = _convert_longitude(ds)

    # Spatial subset
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    ds = ds.sel(latitude=slice(y0, y1), longitude=slice(x0, x1))

    # Temporal subset
    ds = ds.sel(time=coords["time"])

    # Regrid if cutout resolution != 0.25°
    if not (np.isclose(dx, 0.25) and np.isclose(dy, 0.25)):
        ds = ds.interp(
            latitude=coords["y"].values,
            longitude=coords["x"].values,
            method="linear",
        )

    return ds
```

**Coordinate handling**:
```python
def _convert_longitude(ds):
    """Convert 0-360 longitude to -180-180."""
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
    return ds.sortby("longitude")

def _rename_and_clean_coords(ds):
    """Rename latitude/longitude to x/y. ARCO uses 'time' not 'valid_time'."""
    ds = ds.rename({"longitude": "x", "latitude": "y"})
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5),
        y=np.round(ds.y.astype(float), 5),
    )
    ds = maybe_swap_spatial_dims(ds)
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds
```

**Feature-specific functions** — duplicate ~50 lines of processing from era5.py:
- `get_data_wind()`: wind magnitude from u/v, shear exponent, azimuth (~15 lines)
- `get_data_influx()`: albedo, diffuse, J/m²→W/m², SolarPosition (~25 lines)
- `get_data_temperature()`: variable renames (~5 lines)
- `get_data_runoff()`: variable rename (~3 lines)
- `get_data_height()`: calls imported `_add_height()` (~3 lines)

See `era5.py:104-256` for the exact processing logic to duplicate.

**`get_data()` entry point** — same interface as `era5.get_data()`:
```python
def get_data(cutout, feature, tmpdir=None, lock=None, **creation_parameters):
    """Same interface as era5.get_data() for compatibility with data.py."""
    coords = cutout.coords
    sanitize = creation_parameters.get("sanitize", True)

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    ds = func(coords, cutout.chunks, cutout.dx, cutout.dy)
    if sanitize and sanitize_func is not None:
        ds = sanitize_func(ds)

    if feature in static_features:
        return ds.squeeze()
    return ds.sel(time=coords["time"])
```

Key differences from `era5.get_data()`:
- No time chunking needed (zarr handles partial reads natively)
- Accepts `tmpdir`, `lock`, `data_format`, `monthly_requests`, `concurrent_requests` for interface compat but ignores them
- Handles regridding via `xr.interp()` when `dx/dy != 0.25`

### Register the module

**File**: `atlite/datasets/__init__.py`
```python
from atlite.datasets import arco_era5, era5, gebco, sarah
modules = {"era5": era5, "arco_era5": arco_era5, "sarah": sarah, "gebco": gebco}
```

### Add dependencies

**File**: `pyproject.toml`

Add `zarr` and `gcsfs` as optional dependencies:
```toml
[project.optional-dependencies]
arco = ["zarr", "gcsfs"]
```

## Testing

### Existing test harness

- **`conftest.py`**: Session-scoped fixtures creating cutouts. Uses `--cache-path` to persist `.nc` files between runs. Skips CDS tests if `CDSAPI_URL` not set AND no cached file (`conftest.py:44`).
- **`test_creation.py`**: Pure unit tests for Cutout creation — no downloads, no `prepare()`.
- **`test_preparation_and_conversion.py`**: Requires prepared data. `TestERA5` class calls standalone helper functions (`pv_test`, `wind_test`, `runoff_test`, etc.) with `cutout_era5` fixture.
- Test helpers are reusable standalone functions: `pv_test(cutout)`, `wind_test(cutout)`, `runoff_test(cutout)`, `hydro_test(cutout)`, `heat_demand_test(cutout)`, `solar_thermal_test(cutout)`, `line_rating_test(cutout)`, `csp_test(cutout)`, `pv_tracking_test(cutout)`, `coefficient_of_performance_test(cutout)`.

### New fixtures (`conftest.py`)

```python
GCS_AVAILABLE = ...  # check connectivity to storage.googleapis.com

def _prepare_arco_era5_cutout(path, prepare_kwargs=None, **kwargs):
    cutout = Cutout(path=path, module="arco_era5", bounds=BOUNDS, **kwargs)
    if not path.exists() and not GCS_AVAILABLE:
        pytest.skip("GCS not available and no cached cutout")
    cutout.prepare(**(prepare_kwargs or {}))
    return cutout

@pytest.fixture(scope="session")
def cutout_arco_era5(cutouts_path):
    tmp_path = cutouts_path / "cutout_arco_era5.nc"
    return _prepare_arco_era5_cutout(tmp_path, time=TIME)
```

### New test class (`test_preparation_and_conversion.py`)

```python
class TestARCOERA5:
    """Run the same conversion tests against ARCO-ERA5 data."""

    @staticmethod
    def test_all_non_na(cutout_arco_era5):
        assert np.isfinite(cutout_arco_era5.data).all()

    @staticmethod
    def test_dx_dy_preservation(cutout_arco_era5):
        assert np.allclose(np.diff(cutout_arco_era5.data.x), 0.25)
        assert np.allclose(np.diff(cutout_arco_era5.data.y), 0.25)

    @staticmethod
    def test_prepared_features(cutout_arco_era5):
        return prepared_features_test(cutout_arco_era5)

    @staticmethod
    def test_pv(cutout_arco_era5):
        return pv_test(cutout_arco_era5)

    @staticmethod
    def test_wind(cutout_arco_era5):
        return wind_test(cutout_arco_era5)

    @staticmethod
    def test_runoff(cutout_arco_era5):
        return runoff_test(cutout_arco_era5)

    # ... all other conversion tests ...

    @staticmethod
    def test_compare_with_era5(cutout_era5, cutout_arco_era5):
        """Verify ARCO data matches CDS data within tolerance."""
        for var in cutout_era5.data.data_vars:
            xr.testing.assert_allclose(
                cutout_era5.data[var],
                cutout_arco_era5.data[var],
                rtol=1e-5, atol=1e-5,
            )
```

### Test run workflow

1. Provide cached `cutout_era5.nc` (and coarse variant) from a prior CDS run
2. Run `pytest --cache-path=<cache-dir>` with GCS access — ARCO tests download from GCS, compare against cached CDS data
3. Subsequent runs with `--cache-path` use both caches — no network needed

## Files to Create/Modify

| File | Action |
|------|--------|
| `atlite/datasets/arco_era5.py` | **Create**: new dataset module (~200 lines) |
| `atlite/datasets/__init__.py` | Add import + register in `modules` dict |
| `pyproject.toml` | Add `zarr`/`gcsfs` optional deps |
| `test/conftest.py` | Add `cutout_arco_era5` fixture |
| `test/test_preparation_and_conversion.py` | Add `TestARCOERA5` class |
| `atlite/datasets/era5.py` | **No changes** |

## Open Questions (to resolve in Phase B)

1. **Variable names in `ar/` zarr**: Short (`u10`) or long (`10m_u_component_of_wind`)? Build `VAR_MAPPING` accordingly.
2. **Surface vars and `level` dimension**: Do surface-only variables (t2m, u10, etc.) have a `level` coord that needs dropping?
3. **Latitude ordering**: N→S (like CDS) or S→N? Affects `slice()` direction in `.sel()`.
4. **Accumulation conventions**: ERA5 influx vars are accumulated J/m². Verify ARCO stores them identically.
5. **Tolerance**: What `rtol`/`atol` is needed for `assert_allclose`? The `ar/` tier was regridded independently from CDS — expect small numerical differences.

## Usage

```python
cutout = atlite.Cutout(
    path="spain-2024.nc",
    module="arco_era5",    # <-- only change from "era5"
    x=slice(-10, 5),
    y=slice(35, 44),
    time="2024-01",
)
cutout.prepare()  # Reads from Google Cloud — no CDS queue!
```
