# S2SR Inference Metadata

Generated: 2026-08-24T14:55:47+00:00 UTC

## Run

- inference_id: inf-20260824T144801Z-34.42500N-8.78500E-0dc1
- country_code: TN
- longitude: 8.785
- latitude: 34.425
- request_date: 2026-07-24
- target_date_compact: 20260724
- started_at: 2026-08-24T14:48:03+00:00
- finished_at: 2026-08-24T14:55:46+00:00
- command: /home/nextav/miniconda3/envs/s2sr-inference/bin/python scripts/run_location.py --lon 8.7850 --lat 34.4250 --date 2026-07-24

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
  - IRP.tif: 4100x4120, 3 bands uint8, 1 m, compression=NONE, 50,726,330 bytes, sha256=ba4464bed173b4adde0154a91f24e31cb487ae6a61eeb44fb5ef2be825279a98
  - MS.tif: 4100x4120, 10 bands uint16, 1 m, compression=NONE, 337,907,642 bytes, sha256=7c6ae6dd150557819ae5b7cb128ba85be9f62513d57bcc823f57189529e39eff
  - NDVI.tif: 4100x4120, 3 bands uint8, 1 m, compression=NONE, 50,726,330 bytes, sha256=933361a6046c32f78f0d1e596b057292bac8814ccd8c4c83887e296d6d37e98f
  - TCI.tif: 4100x4120, 3 bands uint8, 1 m, compression=NONE, 50,726,330 bytes, sha256=9fca77d61cb24a521588778bc25c209d62adbf474a3f997fb56fa04fd54e5f28
- selected: MS, TCI, NDVI, IRP
- sources:
  - IRP.tif <- TN/T32SMD/T32SMD-00e1e140e/S2SR_T32SMD-00e1e140e-20260724_IRP.tif
  - MS.tif <- TN/T32SMD/T32SMD-00e1e140e/S2SR_T32SMD-00e1e140e-20260724_MS.tif
  - NDVI.tif <- TN/T32SMD/T32SMD-00e1e140e/S2SR_T32SMD-00e1e140e-20260724_NDVI.tif
  - TCI.tif <- TN/T32SMD/T32SMD-00e1e140e/S2SR_T32SMD-00e1e140e-20260724_TCI.tif
- excluded:
  - none

## Indices

- catalog: /home/nextav/Workspace/neuralq-s2sr-api/outputs/TN/2026-07-24/inf-20260824T144801Z-34.42500N-8.78500E-0dc1/indices/README.md
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

- S2SRlog_T32SMD-00e1e140e-20260724_s4e7e36a-9fcb-11f1_20260824T155539.json: {'PID': 'T32SMD-00e1e140e-20260724', 'mode': 'aoi', 'date': '20260726', 'MGRS': '32SMD', 'ISO': 'TN', 'dimentions': '410.0,412.0', 'bbox': '8.763242458462642,34.40697063933099,8.806757541537358,34.443029360669', 'bands': 'B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12', 'save_path_MS': '/home/nextav/Workspace/neuralq-s2sr-api/outputs/TN/2026-07-24/inf-20260824T144801Z-34.42500N-8.78500E-0dc1/TN/T32SMD/T32SMD-00e1e140e/S2SRT32SMD-00e1e140e-20260724_MS.tif', 'save_path_TCI': '/home/nextav/Workspace/neuralq-s2sr-api/outputs/TN/2026-07-24/inf-20260824T144801Z-34.42500N-8.78500E-0dc1/TN/T32SMD/T32SMD-00e1e140e/S2SRT32SMD-00e1e140e-20260724_TCI.tif', 'pid': 'T32SMD-00e1e140e-20260724', 'aoi_overlap': '1.0', 'job_id': 's4e7e36a-9fcb-11f1'}
