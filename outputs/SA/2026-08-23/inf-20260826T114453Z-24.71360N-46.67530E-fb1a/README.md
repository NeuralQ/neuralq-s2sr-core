# S2SR Inference Metadata

Generated: 2026-08-26T11:55:12+00:00 UTC

## Run

- inference_id: inf-20260826T114453Z-24.71360N-46.67530E-fb1a
- country_code: SA
- longitude: 46.6753
- latitude: 24.7136
- request_date: 2026-08-23
- target_date_compact: 20260823
- started_at: 2026-08-26T11:44:53+00:00
- finished_at: 2026-08-26T11:55:11+00:00
- command: /home/nextav/miniconda3/envs/neuralq-s2sr-core/bin/python scripts/run_location.py --lon 46.6753 --lat 24.7136 --date 2026-08-23 --country-code SA

## Model

- model_id: S2SR-GL-20241022.1
- checkpoint: /home/nextav/Workspace/neuralq-s2sr-core/models/S2SR-GL-20241022.1.pt
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
  - IRP.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=d392e72408f9f0c1030b4ac2c95b4a8f9e0bc14fe618245c2009362e01d09ce8
  - MS.tif: 4120x4120, 10 bands uint16, 1 m, compression=NONE, 339,555,584 bytes, sha256=a2b029c024dcee03a5a0e9e669f2164629e61131d56dd672a39e1a405b596b39
  - NDVI.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=3c48cdd28cf2e11a944b4d47f99429db9a7b3893e8aca58a46973cbe3baaaa45
  - TCI.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=ee75e80d9a80390c0dd1b05ac150cbe32f914a0bd9b560c41350c7df41c7e757
- selected: MS, TCI, NDVI, IRP
- sources:
  - flat
- excluded:
  - none

## Indices

- catalog: /home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-08-23/inf-20260826T114453Z-24.71360N-46.67530E-fb1a/indices/README.md
- categories: burn, soil_urban, vegetation, water
- count: 15
- files:
  - indices/vegetation/ndvi.tiff: 67,922,736 bytes
  - indices/vegetation/gndvi.tiff: 67,922,736 bytes
  - indices/vegetation/ndre.tiff: 67,922,736 bytes
  - indices/vegetation/evi2.tiff: 67,922,736 bytes
  - indices/vegetation/savi.tiff: 67,922,736 bytes
  - indices/vegetation/mtci.tiff: 67,922,736 bytes
  - indices/water/ndwi.tiff: 67,922,736 bytes
  - indices/water/mndwi.tiff: 67,922,736 bytes
  - indices/water/ndmi.tiff: 67,922,736 bytes
  - indices/water/awei.tiff: 67,922,736 bytes
  - indices/burn/nbr.tiff: 67,922,736 bytes
  - indices/burn/nbr2.tiff: 67,922,736 bytes
  - indices/soil_urban/bsi.tiff: 67,922,736 bytes
  - indices/soil_urban/ndbi.tiff: 67,922,736 bytes
  - indices/soil_urban/ndti.tiff: 67,922,736 bytes
- note: single-band float32, compression NONE, nodata NaN

## Engine log

- S2SRlog_38RPN-7a7fdcbde-20260823_s89dc80afd76a-4231-11f1_20260826T115508.json: {'PID': '38RPN-7a7fdcbde-20260823', 'mode': 'aoi', 'date': '20260823', 'MGRS': '38RPN', 'ISO': 'SA', 'dimentions': '412.0,412.0', 'bbox': '46.654699029798103,24.695213897106747,46.695914088821155,24.731952637706286', 'bands': 'B02,B03,B04,B08,B05,B06,B07,B11,B12,B8A', 'stack_dates': '2026-05-14,2026-05-24,2026-07-03,2026-07-13,2026-08-22', 'save_path_MS': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-08-23/inf-20260826T114453Z-24.71360N-46.67530E-fb1a/MS.tif', 'save_path_TCI': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-08-23/inf-20260826T114453Z-24.71360N-46.67530E-fb1a/TCI.tif', 'save_path_NDVI': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-08-23/inf-20260826T114453Z-24.71360N-46.67530E-fb1a/NDVI.tif', 'save_path_IRP': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-08-23/inf-20260826T114453Z-24.71360N-46.67530E-fb1a/IRP.tif', 'aoi_overlap': '1.0', 'job_id': 's89dc80afd76a-4231-11f1'}
