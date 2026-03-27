# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and processing data from ECMWFs ERA5 dataset (via NSF NCAR)

NSF NCAR hosts a mirror of ERA5 on their Research Data Archive (RDA, dataset d633000).
However, unlike the original source of ERA5 (the Copernicus Data Store), they do not
require authentication and do not have a download queue.

This module mirrors processing from atlite/datasets/era5.py, and should deliver identical results.
To use it, or to replace era5.py in old code, simply replace `module="era5"` with
`module="era5-ncar"` in `atlite.Cutout` like this:

```
cutout = atlite.Cutout(
    path="my_cutout.nc",
    module="era5-ncar",
    x=slice(-10, 5),
    y=slice(35, 44),
    time="2024",
)
cutout.prepare()
```

Caveats:
- the data has some delay vs. ERA5, ERA5T, so it is not a best choice for very recent data
"""
