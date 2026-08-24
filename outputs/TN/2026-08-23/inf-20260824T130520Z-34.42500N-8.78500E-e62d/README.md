# S2SR Inference Metadata

Generated: 2026-08-24T13:15:26+00:00 UTC

## Run

- inference_id: inf-20260824T130520Z-34.42500N-8.78500E-e62d
- country_code: TN
- longitude: 8.785
- latitude: 34.425
- request_date: 2026-08-23
- target_date_compact: 20260823
- started_at: 2026-08-24T13:05:22+00:00
- finished_at: 2026-08-24T13:15:25+00:00
- command: /home/nextav/miniconda3/envs/s2sr-inference/bin/python scripts/run_location.py --lon 8.7850 --lat 34.4250 --date 2026-08-23

## Model

- model_id: S2SR-GL-20241022.1
- checkpoint: /home/nextav/Workspace/neuralq-s2sr-api/models/S2SR-GL-20241022.1.pt
- checkpoint_sha256: 1ac3d52cac3737842538ed09f329b0023b43cd3d5f509ccce36a0951cb2dd520
- device_setting: auto
- tile_size_pixels: 128

## Data contract

- input_shape: (50, H, W); five dates x ten bands, date-major
- source_band_order: B02, B03, B04, B08, B05, B06, B07, B11, B12, B8A
- normalization: reflectance DN / 10000
- super_resolution_factor: 10x
- output_compression: NONE

## Products

- files:
  - IRP.tif: 4100x4120, 3 bands uint8, 1 m, compression=NONE, 50,726,330 bytes, sha256=fc1f198caebaa649ff6e31f3d55c7ac594209e959e4dafee15c3ba390ec6c426
  - MS.tif: 4100x4120, 10 bands uint16, 1 m, compression=NONE, 337,907,642 bytes, sha256=1ccf9cf6072d4f15035f1b5a3895475eea85ce36819518f9602b63f7c61a654c
  - NDVI.tif: 4100x4120, 3 bands uint8, 1 m, compression=NONE, 50,726,330 bytes, sha256=5d5c16c2185e5f5c28d3c8c14fb1d264c57dac90e68c219c64bd15a1c4fecc33
  - TCI.tif: 4100x4120, 3 bands uint8, 1 m, compression=NONE, 50,726,330 bytes, sha256=8b98f17b6cdb684312e7582b02d09834a2d9018482e590375141d84af332f8e0
- selected: MS, TCI, NDVI, IRP
- sources:
  - IRP.tif <- TN/T32SMD/T32SMD-00e1e140e/S2SR_T32SMD-00e1e140e-20260823_IRP.tif
  - MS.tif <- TN/T32SMD/T32SMD-00e1e140e/S2SR_T32SMD-00e1e140e-20260823_MS.tif
  - NDVI.tif <- TN/T32SMD/T32SMD-00e1e140e/S2SR_T32SMD-00e1e140e-20260823_NDVI.tif
  - TCI.tif <- TN/T32SMD/T32SMD-00e1e140e/S2SR_T32SMD-00e1e140e-20260823_TCI.tif
- excluded:
  - none

## Indices

- catalog: /home/nextav/Workspace/neuralq-s2sr-api/outputs/TN/2026-08-23/inf-20260824T130520Z-34.42500N-8.78500E-e62d/indices/README.md
- categories: burn, soil_urban, vegetation, water
- count: 15
- files:
  - indices/vegetation/ndvi.tiff: 67,593,092 bytes
  - indices/vegetation/gndvi.tiff: 67,593,092 bytes
  - indices/vegetation/ndre.tiff: 67,593,092 bytes
  - indices/vegetation/evi2.tiff: 67,593,092 bytes
  - indices/vegetation/savi.tiff: 67,593,092 bytes
  - indices/vegetation/mtci.tiff: 67,593,092 bytes
  - indices/water/ndwi.tiff: 67,593,092 bytes
  - indices/water/mndwi.tiff: 67,593,092 bytes
  - indices/water/ndmi.tiff: 67,593,092 bytes
  - indices/water/awei.tiff: 67,593,092 bytes
  - indices/burn/nbr.tiff: 67,593,092 bytes
  - indices/burn/nbr2.tiff: 67,593,092 bytes
  - indices/soil_urban/bsi.tiff: 67,593,092 bytes
  - indices/soil_urban/ndbi.tiff: 67,593,092 bytes
  - indices/soil_urban/ndti.tiff: 67,593,092 bytes
- note: single-band float32, compression NONE, nodata NaN

## Engine log

- note: not retained
