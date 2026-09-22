"""The local NeuralQ S2SR inference engine.

Implements the ``datautils`` / ``inferutils`` surface consumed by
``scripts/run_location.py`` using only open components: Earth Search STAC,
Sentinel-2 L2A Cloud-Optimized GeoTIFFs on AWS, rasterio, and the local
S2SR torch model. Scene selection mirrors the original engine contract:
five usable, co-registered acquisitions stacked date-major (ten Sentinel-2
bands each), tiled GPU inference, and branded product rasters written
uncompressed.
"""
