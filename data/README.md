
## Data source

Download hourly ERA5 reanalysis data for Zhongshan Station, Antarctica
(`69.37°S, 76.38°E`) from January 2020 through August 2025.

ERA5 catalogue:
https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5

Prepare the following 12 meteorological variables:

1. `wind_u_10m`
2. `wind_v_10m`
3. `wind_u_100m`
4. `wind_v_100m`
5. `temp_2m`
6. `surface_pressure`
7. `skin_temp`
8. `dewpoint_2m`
9. `boundary_layer_height`
10. `relative_humidity`
11. `wind_speed_10m`
12. `wind_speed_100m`

Compute wind speed from the zonal and meridional components:
wind_speed = sqrt(wind_u^2 + wind_v^2)

## CEEMDAN components

Create one CSV file for each prediction target. Apply CEEMDAN offline to the
complete target wind-speed series and retain nine additive components: eight
intrinsic mode functions and one residual term.

Name the component columns:

```text
IMF_1, IMF_2, ..., IMF_9
```

`IMF_1` is the highest-frequency component. `IMF_9` is the lowest-frequency
component or residual term. The sum of the nine columns should reconstruct the
selected target series, subject to numerical precision.

Suggested filenames:

```text
data/zhongshan_10m.csv
data/zhongshan_100m.csv
```

The component columns in `zhongshan_10m.csv` must be derived from
`wind_speed_10m`. The component columns in `zhongshan_100m.csv` must be derived
from `wind_speed_100m`.

## CSV schema

Example header:

```csv
date,wind_u_10m,wind_v_10m,wind_u_100m,wind_v_100m,temp_2m,surface_pressure,skin_temp,dewpoint_2m,boundary_layer_height,relative_humidity,wind_speed_10m,wind_speed_100m,IMF_1,IMF_2,IMF_3,IMF_4,IMF_5,IMF_6,IMF_7,IMF_8,IMF_9
```

The loader automatically moves the selected target to the last position among
the 12 meteorological variables, as required by the NoisePredictor.
