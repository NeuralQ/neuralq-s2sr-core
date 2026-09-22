"""Unit tests for the NeuralQ S2SR pipeline - standard library only.

Run either way::

    python3 tests/test_pipeline_units.py
    python3 -m pytest tests/test_pipeline_units.py

Third-party dependencies of the scripts (requests, shapely, pyproj,
rasterio) are stubbed so the modules import on any machine. ``s2sr/hub.py``
is loaded directly from its file location so the tests never trigger
``s2sr/__init__.py`` (which needs numpy/torch).
"""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

for name in ("requests", "shapely", "shapely.geometry", "pyproj", "rasterio"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyproj"].Transformer = type("Transformer", (), {})
for symbol in ("MultiPolygon", "box", "mapping", "shape"):
    setattr(sys.modules["shapely.geometry"], symbol, lambda *a, **k: {})
_ops = types.ModuleType("shapely.ops")
_ops.transform = lambda *a, **k: None
_ops.unary_union = lambda *a, **k: None
sys.modules["shapely.ops"] = _ops

sys.path.insert(0, str(SCRIPTS))

import output_layout  # noqa: E402
import run_location  # noqa: E402
import run_mosaic  # noqa: E402
import upstream  # noqa: E402
import lst  # noqa: E402


def _load_hub():
    spec = importlib.util.spec_from_file_location(
        "s2sr_hub_under_test", REPO / "s2sr" / "hub.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hub = _load_hub()


def eq(got, want, label):
    assert got == want, f"{label}: got {got!r}, want {want!r}"


def ok(condition, label):
    assert condition, label


def test_utm_crs_for_matches_validated_zones():
    eq(run_mosaic.utm_crs_for(51.5310, 25.2886), "EPSG:32639", "Doha zone")
    eq(run_mosaic.utm_crs_for(8.7850, 34.4250), "EPSG:32632", "Gafsa zone")
    eq(run_mosaic.utm_crs_for(18.4241, -33.9249), "EPSG:32734", "Cape Town zone")
    eq(run_mosaic.utm_crs_for(-51.0, -12.0), "EPSG:32722", "south-west zone")


def test_upstream_sanitize_brands_all_tokens():
    samples = {
        "S2L3Ax10_T36RXU-abc_MS.tif": "S2SR_T36RXU-abc_MS.tif",
        "log-S2DR4-run.json": "log-S2SR-run.json",
        "s2dr4.inferutils": "S2SR.inferutils",
        "path/S2DR3/cache": "path/S2SR/cache",
        "S2L2Ax10_x": "S2SRx",
    }
    for raw, branded in samples.items():
        eq(upstream.sanitize(raw), branded, f"sanitize({raw!r})")


def test_plan_signature_is_stable_and_order_insensitive():
    a = output_layout.plan_signature(date="2026-08-14", step=4000)
    b = output_layout.plan_signature(step=4000, date="2026-08-14")
    eq(a, b, "signature order independence")
    ok(len(a) == 8, "signature is an 8-char slug")


def test_coordinate_tags_hemispheres():
    eq(output_layout.coordinate_tags(25.2886, 51.5310), "25.28860N-51.53100E", "NE")
    eq(output_layout.coordinate_tags(-33.9249, 18.4241), "33.92490S-18.42410E", "SE")
    eq(output_layout.coordinate_tags(34.4250, -8.7850), "34.42500N-8.78500W", "NW")


def test_boundary_cache_path_is_deterministic():
    p1 = run_mosaic.boundary_cache_path("Doha, Qatar", 27332, 20)
    p2 = run_mosaic.boundary_cache_path("Doha, Qatar", 27332, 20)
    p3 = run_mosaic.boundary_cache_path("Lyon, France", 12345, 20)
    eq(p1, p2, "same plan -> same cache path")
    ok(p1 != p3, "different plan -> different cache path")
    ok(p1.parent.name == ".boundaries", "cache lives under outputs/.boundaries")


def test_band_order_shared_constant():
    eq(
        output_layout.BAND_ORDER,
        ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"),
        "canonical band order",
    )
    eq(len(output_layout.BAND_ORDER), 10, "ten bands")


def test_new_inference_id_shape():
    inference_id = output_layout.new_inference_id(25.2886, 51.5310)
    parts = inference_id.split("-")
    eq(parts[0], "inf", "prefix")
    eq(len(parts[-1]), 4, "random token is 4 hex chars")
    ok(
        "25.28860N-51.53100E" in inference_id,
        "coordinate tag embedded in the id",
    )


def test_place_slug():
    eq(run_mosaic.place_slug("Lyon, France"), "Lyon", "first comma-part")
    eq(run_mosaic.place_slug("Doha, Qatar"), "Doha", "default city")
    ok(
        run_mosaic.place_slug("Frankfurt am Main, Germany").startswith("Frankfurt"),
        "multi-word kept",
    )


def test_mosaic_id_distinguishes_boundaries():
    common = ("2026-08-14", 4000.0, 4120.0, ["MS"], 20)
    doha = run_mosaic.mosaic_inference_id(*common, query="Doha, Qatar", osm_id=27332)
    lyon = run_mosaic.mosaic_inference_id(*common, query="Lyon, France", osm_id=35238)
    ok(doha != lyon, "different cities never share a deterministic folder id")
    eq(
        run_mosaic.mosaic_inference_id(*common, query="Doha, Qatar", osm_id=27332),
        doha,
        "same plan stays stable",
    )


def test_resolve_checkpoint_finds_bundled():
    path = hub.resolve_checkpoint()
    eq(path.name, hub.MODEL_FILENAME, "bundled filename")
    ok(path.is_file(), f"bundled weights exist: {path}")


def test_resolve_checkpoint_missing():
    try:
        hub.resolve_checkpoint("does-not-exist.pt")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing checkpoint must raise FileNotFoundError")
    eq(hub.cached_checkpoint_path("does-not-exist.pt"), None, "cached miss -> None")


def test_resolve_checkpoint_explicit_path():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="hub-test-") as scratch:
        custom = Path(scratch) / "custom.pt"
        custom.write_bytes(b"weights")
        eq(hub.resolve_checkpoint(custom), custom.resolve(), "explicit path wins")
        # Custom files without a sibling sidecar pass verification untouched.
        eq(hub.verify_checkpoint(custom), custom, "no sidecar -> passthrough")


def test_verify_checkpoint_sidecar_lifecycle():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="hub-test-") as scratch:
        candidate = Path(scratch) / "w.pt"
        candidate.write_bytes(b"abc")
        sidecar = Path(f"{candidate}.sha256")
        sidecar.write_text(hashlib.sha256(b"abc").hexdigest() + "  w.pt\n")
        eq(hub.verify_checkpoint(candidate), candidate, "matching sidecar passes")
        candidate.write_bytes(b"abcd")
        try:
            hub.verify_checkpoint(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt file must raise ValueError")


def test_verify_bundled_checkpoint():
    # Hashes ~800 MB; slow but this is the pin that guards every run.
    eq(hub.verify_checkpoint(hub.resolve_checkpoint()), hub.resolve_checkpoint(), "bundled verifies")


def test_model_id_consistent():
    eq(hub.MODEL_ID, run_location.MODEL_ID, "hub and runner agree on MODEL_ID")
    eq(hub.MODEL_FILENAME, f"{run_location.MODEL_ID}.pt", "filename derived from id")


def test_sidecar_format():
    lines = (REPO / "models" / "s2sr-v3.0.0.pt.sha256").read_text().split()
    eq(len(lines), 2, "coreutils two-field sidecar")
    ok(len(lines[0]) == 64 and all(c in "0123456789abcdef" for c in lines[0]), "64 hex chars")
    eq(lines[1], "s2sr-v3.0.0.pt", "sidecar names the weights file")


def test_no_hf_integration_in_sources():
    forbidden = ("hf_hub_download", "HF_REPO_ID", "HF_FILENAME", "HUGGING_FACE_HUB_TOKEN")
    for relative in ("s2sr/hub.py", "s2sr/__init__.py", "scripts/run_location.py", "scripts/run_mosaic.py"):
        text = (REPO / relative).read_text(encoding="utf-8")
        for token in forbidden:
            ok(token not in text, f"{relative} must not reference {token}")


def test_docker_bakes_bundled_model():
    dockerignore = (REPO / ".dockerignore").read_text(encoding="utf-8")
    ok(
        not any(line.strip() == "models/" for line in dockerignore.splitlines()),
        ".dockerignore must not exclude models/",
    )
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    ok("models/s2sr-v3.0.0.pt" in dockerfile, "Dockerfile references bundled weights")
    ok("sha256sum -c" in dockerfile, "Dockerfile verifies the weights checksum")
    compose = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    ok("HF_TOKEN:?" not in compose, "compose must not require HF_TOKEN")


def test_qa_clear_bits():
    eq(lst.qa_clear(0), True, "no bits set -> clear")
    eq(lst.qa_clear(1), False, "fill bit set -> unusable")
    eq(lst.qa_clear(64), True, "clear bit only -> clear")
    eq(lst.qa_clear(128), True, "water bit only -> usable")
    for bit, label in ((1, "dilated"), (2, "cirrus"), (3, "cloud"), (4, "shadow"), (5, "snow")):
        eq(lst.qa_clear(1 << bit), False, f"{label} bit -> not clear")
    eq(lst.qa_clear(64 | 8), False, "clear+cloud -> not clear")


def test_fit_lst_ndvi_line():
    pairs = [(x / 10.0, 300.0 + 20.0 * (x / 10.0)) for x in range(-5, 11)]
    a, b, rmse, n = lst.fit_lst_ndvi(pairs)
    ok(abs(a - 300.0) < 1e-9, "intercept recovered")
    ok(abs(b - 20.0) < 1e-9, "slope recovered")
    ok(rmse < 1e-9, "perfect fit has zero rmse")
    eq(n, len(pairs), "sample count")
    try:
        lst.fit_lst_ndvi([(0.5, 300.0)])
    except lst.LSTUnavailable as error:
        eq(error.reason, "too-few-samples", "single sample rejected")
    else:
        raise AssertionError("single sample must raise LSTUnavailable")


def test_select_scene_prefers_date_then_clouds():
    items = [
        {"id": "far-clear", "info": {"date": "2026-08-01", "clouds": 0.0}},
        {"id": "near-cloudy", "info": {"date": "2026-08-13", "clouds": 40.0}},
        {"id": "near-clear", "info": {"date": "2026-08-13", "clouds": 2.0}},
    ]
    eq(lst.select_scene(items, "2026-08-14")["id"], "near-clear", "date first, clouds second")
    try:
        lst.select_scene([], "2026-08-14")
    except lst.LSTUnavailable as error:
        eq(error.reason, "no-scenes", "empty search rejected")
    else:
        raise AssertionError("empty items must raise LSTUnavailable")


def test_lst_window_and_grid():
    eq(lst.window_range("2026-08-14"), ("2026-07-29", "2026-08-15"), "±16d window")
    eq(lst.coarse_shape(4120, 4120), (138, 138), "30 m cells cover the tile")


def test_plausible_kelvin_band():
    eq(lst.plausible_kelvin(300.0), True, "room desert noon is fine")
    eq(lst.plausible_kelvin(230.0), True, "lower edge inclusive")
    eq(lst.plausible_kelvin(360.0), True, "upper edge inclusive")
    eq(lst.plausible_kelvin(491.2), False, "unphysical heat rejected")
    eq(lst.plausible_kelvin(100.0), False, "unphysical cold rejected")
    eq(lst.plausible_kelvin(float("nan")), False, "NaN rejected")


def test_st_scale_offset_usgs():
    # USGS C2 ST: Kelvin = DN * 0.00341802 + 149.0 (NOT the C1 0.1 factor).
    mult, add = lst.st_scale_offset(0.00341802, 149.0)
    eq(round(mult, 8), round(0.00341802, 8), "matching tags trusted")
    eq(add, 149.0, "matching offset trusted")
    eq(lst.st_scale_offset(None, None), (lst.ST_MULT_USGS, lst.ST_ADD_USGS), "missing tags -> USGS")
    eq(lst.st_scale_offset(0.01, 0.0), (lst.ST_MULT_USGS, lst.ST_ADD_USGS), "C1-style tags rejected")
    eq(lst.st_scale_offset(1.0, 0.0), (lst.ST_MULT_USGS, lst.ST_ADD_USGS), "unset tags rejected")
    ok(abs(50000 * mult + add - 319.9) < 0.1, "Doha noon DN ~50000 -> ~320 K")


def test_mosaic_lst_is_opt_in_float():
    eq(run_mosaic.PRODUCTS["LST"], 1, "LST is single-band")
    eq(run_mosaic.PRODUCT_DTYPES["LST"], "float32", "LST dtype")
    ok("LST" not in run_mosaic.DEFAULT_PRODUCTS, "mosaic default excludes LST")
    ok("LST" in run_mosaic.PRODUCTS, "LST selectable via --products")


def test_oil_fixture_threshold_regression():
    import json

    for name in ("wakashio_oil.geojson", "sousse_harbour.geojson"):
        fixture = REPO / "tests" / "fixtures" / name
        ok(fixture.is_file(), f"oil fixture {name} exists")
        collection = json.loads(fixture.read_text(encoding="utf-8"))
        ok(collection["type"] == "FeatureCollection", f"{name} is FeatureCollection")
        labels = {f["properties"]["label"] for f in collection["features"]}
        ok(bool(labels & {"oil", "oil_candidate"}) and "clean_water" in labels, f"{name} has oil + clean_water")
    # Smoke the OSI formula on synthetic oil vs water spectra — threshold drift guard
    import sys as _sys

    sys.path.insert(0, str(REPO / "scripts"))
    import spectral_indices as si

    # Simulate reflectance: clean water (B04 low, B11/B12 ~0) vs oil (B11/B12 bright)
    oil_bands = {"B04": 0.06, "B08": 0.02, "B11": 0.08, "B12": 0.09, "B03": 0.06}
    water_bands = {"B04": 0.02, "B08": 0.01, "B11": 0.01, "B12": 0.01, "B03": 0.08}
    # OSI = (B11+B12-B08-B04)/(sum)
    osi_oil = si._formulas()["osi"](oil_bands)
    osi_water = si._formulas()["osi"](water_bands)
    ok(float(osi_oil) > 0.15, f"oil OSI {float(osi_oil):.2f} must exceed sea candidate 0.15")
    ok(float(osi_water) < 0.05, f"clean water OSI {float(osi_water):.2f} must stay near zero")
    # Also ensure the six-index oil suite is complete
    ok(set(si.CATEGORIES["oil"]) == {"osi", "hi", "foi", "ndoi", "sr", "rg"}, "oil suite has 6 indices")


def main() -> int:
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    print(f"\n{'ALL TESTS PASSED' if not failures else f'{failures} TEST(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
