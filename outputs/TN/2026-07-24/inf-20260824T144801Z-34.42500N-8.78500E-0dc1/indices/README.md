# Spectral Indices

Generated: 2026-08-24T14:55:46+00:00 UTC

Computed from the S2SR super-resolved `MS.tif` product (ten Sentinel-2 bands, 1 m) with reflectance DNs divided by 10,000.

Every file is a single-band `float32` GeoTIFF, `COMPRESS=NONE`, `NoData = NaN`, sharing the CRS, extent, and 1 m grid of `MS.tif`. These are raw mathematical rasters; the colorized three-band preview `NDVI.tif` at the inference root is a visualization and is not part of this catalog.

| Category | File | Formula | Bands | Typical range | Size (MB) |
| --- | --- | --- | --- | --- | ---: |
| burn | `burn/nbr.tiff` | `(B08 - B12) / (B08 + B12)` | B08, B12 | -1 .. 1 | 67.6 |
| burn | `burn/nbr2.tiff` | `(B11 - B12) / (B11 + B12)` | B11, B12 | -1 .. 1 | 67.6 |
| soil_urban | `soil_urban/bsi.tiff` | `((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))` | B11, B04, B08, B02 | -1 .. 1 | 67.6 |
| soil_urban | `soil_urban/ndbi.tiff` | `(B11 - B08) / (B11 + B08)` | B11, B08 | -1 .. 1 | 67.6 |
| soil_urban | `soil_urban/ndti.tiff` | `(B11 - B12) / (B11 + B12)` | B11, B12 | -1 .. 1 | 67.6 |
| vegetation | `vegetation/evi2.tiff` | `2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1)` | B08, B04 | approx -1 .. 1, can exceed | 67.6 |
| vegetation | `vegetation/gndvi.tiff` | `(B08 - B03) / (B08 + B03)` | B08, B03 | -1 .. 1 | 67.6 |
| vegetation | `vegetation/mtci.tiff` | `(B06 - B05) / (B05 - B04)` | B06, B05, B04 | positive for healthy vegetation | 67.6 |
| vegetation | `vegetation/ndre.tiff` | `(B08 - B05) / (B08 + B05)` | B08, B05 | -1 .. 1 | 67.6 |
| vegetation | `vegetation/ndvi.tiff` | `(B08 - B04) / (B08 + B04)` | B08, B04 | -1 .. 1 | 67.6 |
| vegetation | `vegetation/savi.tiff` | `1.5 * (B08 - B04) / (B08 + B04 + 0.5)` | B08, B04 | -1 .. 1 | 67.6 |
| water | `water/awei.tiff` | `4 * (B03 - B11) - (0.25 * B08 + 2.75 * B12)` | B03, B08, B11, B12 | unbounded, water typically > 0 | 67.6 |
| water | `water/mndwi.tiff` | `(B03 - B11) / (B03 + B11)` | B03, B11 | -1 .. 1 | 67.6 |
| water | `water/ndmi.tiff` | `(B08 - B11) / (B08 + B11)` | B08, B11 | -1 .. 1 | 67.6 |
| water | `water/ndwi.tiff` | `(B03 - B08) / (B03 + B08)` | B03, B08 | -1 .. 1 | 67.6 |
