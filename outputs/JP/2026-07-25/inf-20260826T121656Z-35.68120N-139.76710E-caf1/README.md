# S2SR Inference Metadata

Generated: 2026-08-26T12:28:52+00:00 UTC

## Run

- inference_id: inf-20260826T121656Z-35.68120N-139.76710E-caf1
- country_code: JP
- longitude: 139.7671
- latitude: 35.6812
- request_date: 2026-07-25
- target_date_compact: 20260725
- started_at: 2026-08-26T12:16:56+00:00
- finished_at: 2026-08-26T12:28:51+00:00
- command: /home/nextav/miniconda3/envs/neuralq-s2sr-core/bin/python scripts/run_location.py --lon 139.7671 --lat 35.6812 --date 2026-07-25 --country-code JP

## Model

- model_id: s2sr-v3.0.0
- checkpoint: /home/nextav/Workspace/neuralq-s2sr-core/models/s2sr-v3.0.0.pt
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
  - IRP.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=fc065b14db201e8ed8387323ad598b92f79365449b6028909423d157e41eb883
  - MS.tif: 4120x4120, 10 bands uint16, 1 m, compression=NONE, 339,555,584 bytes, sha256=9c0a0fa9a9950ae20dd13e372ef5f781c60be5b42efd7706741d6eeacda31d0d
  - NDVI.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=4ce3310ece1a97b79ad998df26f88381ba0884d39448fd724626b16338f26fcb
  - TCI.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=9e77dcbfeddeb74e5c8042be24b0704bc1879993ba509038d526c4bb7dff6fe2
- selected: MS, TCI, NDVI, IRP
- sources:
  - flat
- excluded:
  - none

## Indices

- catalog: /home/nextav/Workspace/neuralq-s2sr-core/outputs/JP/2026-07-25/inf-20260826T121656Z-35.68120N-139.76710E-caf1/indices/README.md
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

- S2SRlog_54SUE-d47752734-20260725_s8484bc07abbc-66ac-11f1_20260826T122848.json: {'PID': '54SUE-d47752734-20260725', 'mode': 'aoi', 'date': '20260725', 'MGRS': '54SUE', 'ISO': 'JP', 'dimentions': '412.0,412.0', 'bbox': '139.744593779827824,35.662394846717930,139.789542196853347,35.700000101917929', 'bands': 'B02,B03,B04,B08,B05,B06,B07,B11,B12,B8A', 'stack_dates': '2026-04-11,2026-04-16,2026-05-16,2026-07-20,2026-07-25', 'save_path_MS': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/JP/2026-07-25/inf-20260826T121656Z-35.68120N-139.76710E-caf1/MS.tif', 'save_path_TCI': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/JP/2026-07-25/inf-20260826T121656Z-35.68120N-139.76710E-caf1/TCI.tif', 'save_path_NDVI': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/JP/2026-07-25/inf-20260826T121656Z-35.68120N-139.76710E-caf1/NDVI.tif', 'save_path_IRP': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/JP/2026-07-25/inf-20260826T121656Z-35.68120N-139.76710E-caf1/IRP.tif', 'aoi_overlap': '1.0', 'job_id': 's8484bc07abbc-66ac-11f1'}
