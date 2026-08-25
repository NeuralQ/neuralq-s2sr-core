"""Unit tests for the NeuralQ S2SR pipeline - standard library only.

Run either way::

    python3 tests/test_pipeline_units.py
    python3 -m pytest tests/test_pipeline_units.py

Third-party dependencies of the scripts (requests, shapely, pyproj,
rasterio) are stubbed so the modules import on any machine.
"""
import ast
import os
import sys
import tempfile
import types
from argparse import Namespace
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
import run_mosaique_doha  # noqa: E402
import upstream  # noqa: E402


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


def test_period_key_weekly_monthly():
    from datetime import date

    key, midpoint = run_mosaique_doha.period_key(date(2026, 8, 23), "weekly")
    eq(key, "2026-W34", "iso week key")
    eq(midpoint, date(2026, 8, 20), "week midpoint is Thursday")
    key, midpoint = run_mosaique_doha.period_key(date(2026, 8, 1), "monthly")
    eq(key, "2026-08", "month key")
    eq(midpoint, date(2026, 8, 15), "month midpoint")


def test_boundary_cache_path_is_deterministic():
    p1 = run_mosaic.boundary_cache_path("Doha, Qatar", 27332, 20)
    p2 = run_mosaic.boundary_cache_path("Doha, Qatar", 27332, 20)
    p3 = run_mosaic.boundary_cache_path("Lyon, France", 12345, 20)
    eq(p1, p2, "same plan -> same cache path")
    ok(p1 != p3, "different plan -> different cache path")
    ok(p1.parent.name == ".boundaries", "cache lives under outputs/.boundaries")


def test_build_command_only_emits_accepted_run_mosaic_flags():
    tree = ast.parse((SCRIPTS / "run_mosaic.py").read_text())
    accepted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                accepted.add(node.args[0].value)
    args = Namespace(
        products=["MS"],
        tile_size=128,
        grid_step=None,
        tile_footprint=None,
        retries=None,
        prune_unselected=False,
        skip_mosaic=False,
        max_component_distance_km=20,
        boundary_query="Doha, Qatar",
        osm_id=27332,
        tile_timeout=1800,
    )
    command = run_mosaique_doha.build_command(args, "2026-08-14", Path("/out"), "mid")
    unknown = [p for p in command[2:] if p.startswith("--") and p not in accepted]
    eq(unknown, [], "every forwarded flag exists on run_mosaic's parser")
    body = command[2:]
    for index, part in enumerate(body):
        if part.startswith("--") and index + 1 < len(body):
            ok(not body[index + 1].startswith("--"), f"flag {part} carries a value")


def test_workspace_busy_portable_liveness():
    original = run_mosaique_doha._pid_command
    scratch = Path(tempfile.mkdtemp(prefix="ws-test-"))
    try:
        ok(not run_mosaique_doha.workspace_busy(scratch / "missing"), "no pid file -> free")

        garbage = scratch / "garbage"
        garbage.mkdir()
        (garbage / "runner.pid").write_text("not-a-pid\n")
        ok(not run_mosaique_doha.workspace_busy(garbage), "garbage pid -> free")

        dead = scratch / "dead"
        dead.mkdir()
        (dead / "runner.pid").write_text("999999999\n")
        ok(
            not run_mosaique_doha.workspace_busy(dead),
            "dead pid -> free (real ps call)",
        )

        live = scratch / "live"
        live.mkdir()
        (live / "runner.pid").write_text(f"{os.getpid()}\n")
        ok(
            not run_mosaique_doha.workspace_busy(live),
            "live non-run_mosaic process -> free",
        )
        run_mosaique_doha._pid_command = (
            lambda pid: "python scripts/run_mosaic.py --date 2026-08-14"
        )
        ok(run_mosaique_doha.workspace_busy(live), "run_mosaic owner -> busy")

        def explode(pid):
            raise FileNotFoundError("no ps available")

        run_mosaique_doha._pid_command = explode
        ok(
            run_mosaique_doha.workspace_busy(live),
            "uninspectable system -> conservatively busy",
        )
    finally:
        run_mosaique_doha._pid_command = original


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
