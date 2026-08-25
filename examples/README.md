# Examples

Location-specific presets built on the generic scripts in `../scripts/`.
Copy one of these files and change the constants to support a new city.

## Doha

```bash
# single location: central Doha, default date 2026-08-14
python examples/run_location_doha.py            # --search-only for scene probe

# weekly time series of boundary mosaics over the Doha municipality
python examples/run_mosaique_doha.py --plan-dates \
  --start-date 2020-01-01 --end-date 2026-08-21 --frequency weekly
python examples/run_mosaique_doha.py
```

Every option of the underlying generic script remains available; the example
only injects defaults (coordinates, dates, boundary query).

## New city checklist

1. Copy `run_location_doha.py`, set your longitude/latitude/date constants.
2. For mosaics or time series, no code is needed at all - pass
   `--boundary-query "<City>, <Country>"` (+ optional `--osm-id`) to
   `scripts/run_mosaic.py`; the UTM zone and product naming follow the
   boundary automatically.
