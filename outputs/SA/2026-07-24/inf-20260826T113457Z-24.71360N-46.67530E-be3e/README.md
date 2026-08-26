# S2SR Inference Metadata

Generated: 2026-08-26T11:44:22+00:00 UTC

## Run

- inference_id: inf-20260826T113457Z-24.71360N-46.67530E-be3e
- country_code: SA
- longitude: 46.6753
- latitude: 24.7136
- request_date: 2026-07-24
- target_date_compact: 20260724
- started_at: 2026-08-26T11:34:57+00:00
- finished_at: 2026-08-26T11:44:21+00:00
- command: /home/nextav/miniconda3/envs/neuralq-s2sr-core/bin/python scripts/run_location.py --lon 46.6753 --lat 24.7136 --date 2026-07-24 --country-code SA

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
  - IRP.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=74793787c83b373cebfc3d29e5cf064fecb00d0774f4dc90e591d6337461f304
  - MS.tif: 4120x4120, 10 bands uint16, 1 m, compression=NONE, 339,555,584 bytes, sha256=86842b55dd84e85ac2a8ca514e0fd6c64adfb76bde48b3576d212b9dbd79925b
  - NDVI.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=3f7ef9b0089d2555b023823c0ca04ec6f90e20e1d7ba747aa01596f051afb2ef
  - TCI.tif: 4120x4120, 3 bands uint8, 1 m, compression=NONE, 50,948,336 bytes, sha256=53c7099331de6e76028e59dd199a8bffc933136409dde7760a65850d2cb09315
- selected: MS, TCI, NDVI, IRP
- sources:
  - flat
- excluded:
  - none

## Indices

- catalog: /home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-07-24/inf-20260826T113457Z-24.71360N-46.67530E-be3e/indices/README.md
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

- S2SRlog_38RPN-1bf56ef7c-20260724_sdf9a6123e4bf-ad30-11f1_20260826T114418.json: {'PID': '38RPN-1bf56ef7c-20260724', 'mode': 'aoi', 'date': '20260724', 'MGRS': '38RPN', 'ISO': 'SA', 'dimentions': '412.0,412.0', 'bbox': '46.654699029798103,24.695213897106747,46.695914088821155,24.731952637706286', 'bands': 'B02,B03,B04,B08,B05,B06,B07,B11,B12,B8A', 'stack_dates': '2026-05-14,2026-05-24,2026-07-03,2026-07-13,2026-07-23', 'save_path_MS': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-07-24/inf-20260826T113457Z-24.71360N-46.67530E-be3e/MS.tif', 'save_path_TCI': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-07-24/inf-20260826T113457Z-24.71360N-46.67530E-be3e/TCI.tif', 'save_path_NDVI': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-07-24/inf-20260826T113457Z-24.71360N-46.67530E-be3e/NDVI.tif', 'save_path_IRP': '/home/nextav/Workspace/neuralq-s2sr-core/outputs/SA/2026-07-24/inf-20260826T113457Z-24.71360N-46.67530E-be3e/IRP.tif', 'aoi_overlap': '1.0', 'job_id': 'sdf9a6123e4bf-ad30-11f1'}
