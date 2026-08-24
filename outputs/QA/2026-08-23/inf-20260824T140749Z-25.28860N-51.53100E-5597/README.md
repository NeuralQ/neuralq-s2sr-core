# S2SR Inference Metadata

Generated: 2026-08-24T14:12:14+00:00 UTC

## Run

- inference_id: inf-20260824T140749Z-25.28860N-51.53100E-5597
- country_code: QA
- longitude: 51.531
- latitude: 25.2886
- request_date: 2026-08-23
- target_date_compact: 20260823
- started_at: 2026-08-24T14:07:51+00:00
- finished_at: 2026-08-24T14:12:13+00:00
- command: /home/nextav/miniconda3/envs/s2sr-inference/bin/python scripts/run_location.py --lon 51.5310 --lat 25.2886 --date 2026-08-23

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
  - IRP.tif: 4120x4100, 3 bands uint8, 1 m, compression=NONE, 50,726,090 bytes, sha256=fc4358f10a2adf0252d2d07955f1f48dca5cf4c4e4c6c91d516cda55270c8cc1
  - MS.tif: 4120x4100, 10 bands uint16, 1 m, compression=NONE, 337,907,322 bytes, sha256=8a02a1fdc45198f1fbae18b15a0605fa4a944401dbc0439b3ab4496f24893e7d
  - NDVI.tif: 4120x4100, 3 bands uint8, 1 m, compression=NONE, 50,726,090 bytes, sha256=fdd4eaf7df0ba28b95dfee1911ccb16065c11f8faa7ab046b4987377767a49b3
  - TCI.tif: 4120x4100, 3 bands uint8, 1 m, compression=NONE, 50,726,090 bytes, sha256=eeaeae38c58e54f88e6ab77f6825bf3c4e5f81b6cb8094a5b981d31c054ef393
- selected: MS, TCI, NDVI, IRP
- sources:
  - IRP.tif <- QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260823_IRP.tif
  - MS.tif <- QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260823_MS.tif
  - NDVI.tif <- QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260823_NDVI.tif
  - TCI.tif <- QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260823_TCI.tif
- excluded:
  - none

## Indices

- catalog: /home/nextav/Workspace/neuralq-s2sr-api/outputs/QA/2026-08-23/inf-20260824T140749Z-25.28860N-51.53100E-5597/indices/README.md
- categories: burn, soil_urban, vegetation, water
- count: 15
- files:
  - indices/vegetation/ndvi.tiff: 67,592,972 bytes
  - indices/vegetation/gndvi.tiff: 67,592,972 bytes
  - indices/vegetation/ndre.tiff: 67,592,972 bytes
  - indices/vegetation/evi2.tiff: 67,592,972 bytes
  - indices/vegetation/savi.tiff: 67,592,972 bytes
  - indices/vegetation/mtci.tiff: 67,592,972 bytes
  - indices/water/ndwi.tiff: 67,592,972 bytes
  - indices/water/mndwi.tiff: 67,592,972 bytes
  - indices/water/ndmi.tiff: 67,592,972 bytes
  - indices/water/awei.tiff: 67,592,972 bytes
  - indices/burn/nbr.tiff: 67,592,972 bytes
  - indices/burn/nbr2.tiff: 67,592,972 bytes
  - indices/soil_urban/bsi.tiff: 67,592,972 bytes
  - indices/soil_urban/ndbi.tiff: 67,592,972 bytes
  - indices/soil_urban/ndti.tiff: 67,592,972 bytes
- note: single-band float32, compression NONE, nodata NaN

## Engine log

- S2SRlog_T39RWH-04742819d-20260823_s4437430-9fc5-11f1_20260824T151153.json: {'PID': 'T39RWH-04742819d-20260823', 'mode': 'aoi', 'date': '20260816', 'MGRS': '39RWH', 'ISO': 'QA', 'dimentions': '412.0,410.0', 'bbox': '51.51114159183148,25.27054575464071,51.550858408168516,25.306654245359287', 'bands': 'B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12', 'save_path_MS': '/home/nextav/Workspace/neuralq-s2sr-api/outputs/QA/2026-08-23/inf-20260824T140749Z-25.28860N-51.53100E-5597/QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260823_MS.tif', 'save_path_TCI': '/home/nextav/Workspace/neuralq-s2sr-api/outputs/QA/2026-08-23/inf-20260824T140749Z-25.28860N-51.53100E-5597/QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260823_TCI.tif', 'pid': 'T39RWH-04742819d-20260823', 'aoi_overlap': '1.0', 'job_id': 's4437430-9fc5-11f1'}
