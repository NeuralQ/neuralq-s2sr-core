# S2SR Inference Metadata

Generated: 2026-08-24T14:25:08+00:00 UTC

## Run

- inference_id: inf-20260824T141904Z-25.28860N-51.53100E-058b
- country_code: QA
- longitude: 51.531
- latitude: 25.2886
- request_date: 2026-07-24
- target_date_compact: 20260724
- started_at: 2026-08-24T14:19:07+00:00
- finished_at: 2026-08-24T14:25:07+00:00
- command: /home/nextav/miniconda3/envs/s2sr-inference/bin/python scripts/run_location.py --lon 51.5310 --lat 25.2886 --date 2026-07-24

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
  - IRP.tif: 4120x4100, 3 bands uint8, 1 m, compression=NONE, 50,726,090 bytes, sha256=c4a890bd14fbc366576c108fd315327f7b13016297963b9b5fa84e0b25bf9aa0
  - MS.tif: 4120x4100, 10 bands uint16, 1 m, compression=NONE, 337,907,322 bytes, sha256=c8bb20ede5d95db4ef902135879ad97183f2ec3c36097579638908dc8033eddb
  - NDVI.tif: 4120x4100, 3 bands uint8, 1 m, compression=NONE, 50,726,090 bytes, sha256=6dfaa7a47dd93b25fc062674a9ed73a249dace1e12b6c612285bae6bd6c6fb76
  - TCI.tif: 4120x4100, 3 bands uint8, 1 m, compression=NONE, 50,726,090 bytes, sha256=e2f175f50064ba41f86b471d65e1ec95b20a1134b32f3607c91372bf26f28c08
- selected: MS, TCI, NDVI, IRP
- sources:
  - IRP.tif <- QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260724_IRP.tif
  - MS.tif <- QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260724_MS.tif
  - NDVI.tif <- QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260724_NDVI.tif
  - TCI.tif <- QA/T39RWH/T39RWH-04742819d/S2SR_T39RWH-04742819d-20260724_TCI.tif
- excluded:
  - none

## Indices

- catalog: /home/nextav/Workspace/neuralq-s2sr-api/outputs/QA/2026-07-24/inf-20260824T141904Z-25.28860N-51.53100E-058b/indices/README.md
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

- S2SRlog_T39RWH-04742819d-20260724_s4999dd2-9fc7-11f1_20260824T152422.json: {'PID': 'T39RWH-04742819d-20260724', 'mode': 'aoi', 'date': '20260727', 'MGRS': '39RWH', 'ISO': 'QA', 'dimentions': '412.0,410.0', 'bbox': '51.51114159183148,25.27054575464071,51.550858408168516,25.306654245359287', 'bands': 'B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12', 'save_path_MS': '/home/nextav/Workspace/neuralq-s2sr-api/outputs/QA/2026-07-24/inf-20260824T141904Z-25.28860N-51.53100E-058b/QA/T39RWH/T39RWH-04742819d/S2SRT39RWH-04742819d-20260724_MS.tif', 'save_path_TCI': '/home/nextav/Workspace/neuralq-s2sr-api/outputs/QA/2026-07-24/inf-20260824T141904Z-25.28860N-51.53100E-058b/QA/T39RWH/T39RWH-04742819d/S2SRT39RWH-04742819d-20260724_TCI.tif', 'pid': 'T39RWH-04742819d-20260724', 'aoi_overlap': '1.0', 'job_id': 's4999dd2-9fc7-11f1'}
