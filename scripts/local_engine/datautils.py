"""datautils shim: AOI construction and STAC scene search/ranking."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import requests

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
MAX_CLOUDS = 80.0
SEARCH_WINDOW_DAYS = 120


def aoi_from_xy(lonlat, km: float = 2) -> dict:
    lon, lat = float(lonlat[0]), float(lonlat[1])
    half = km * 1000 / 2
    delta_lon = half / (111_320.0 * max(np.cos(np.radians(lat)), 0.01))
    delta_lat = half / 110_540.0
    bbox = [
        round(lon - delta_lon, 7),
        round(lat - delta_lat, 7),
        round(lon + delta_lon, 7),
        round(lat + delta_lat, 7),
    ]
    return {"lon": lon, "lat": lat, "km": km, "bbox": bbox}


def _stac_search(bbox: list[float], start: str, end: str) -> list[dict]:
    payload = {
        "collections": [COLLECTION],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "bbox": bbox,
        "limit": 100,
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
    }
    features: list[dict] = []
    url, method, body = STAC_URL, "POST", payload
    while True:
        kwargs = {"json" if method == "POST" else "params": body}
        response = requests.request(method, url, timeout=120, **kwargs)
        response.raise_for_status()
        page = response.json()
        features.extend(page.get("features", []))
        link = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if link is None:
            return features
        url, method = link["href"], link.get("method", "POST").upper()
        body = {**payload, **(link.get("body") or {})}


def search_collection(args) -> None:
    """Populate ``args.items`` with candidate scenes around ``args.aoi``."""
    target = datetime.strptime(str(args.date), "%Y%m%d").date()
    start = (target - timedelta(days=SEARCH_WINDOW_DAYS)).isoformat()
    end = (target + timedelta(days=1)).isoformat()
    features = _stac_search(args.aoi["bbox"], start, end)
    items = []
    for feature in features:
        properties = feature["properties"]
        clouds = properties.get("eo:cloud_cover", 9999)
        if clouds > MAX_CLOUDS:
            continue
        items.append(
            {
                "id": feature["id"],
                "mgrs_tile": feature["id"].split("_")[1] if "_" in feature["id"] else None,
                "geometry": feature.get("geometry"),
                "assets": feature.get("assets", {}),
                "info": {
                    "date": properties["datetime"][:10],
                    "clouds": clouds,
                    "mgrs_tile": feature["id"].split("_")[1] if "_" in feature["id"] else None,
                },
            }
        )
    args.items = items


def sort_items_by_recency_and_clouds(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (item["info"]["date"], item["info"]["clouds"]),
        reverse=True,
    )


def select_stack_dates(items: list[dict], target_iso: str, count: int = 5) -> list[dict]:
    """Pick the anchor scene closest to the target plus the clearest others."""
    if not items:
        raise RuntimeError("no usable scenes found for this location/date")
    target = date.fromisoformat(target_iso)

    def distance(item: dict) -> int:
        return abs((date.fromisoformat(item["info"]["date"]) - target).days)

    anchor = min(items, key=lambda item: (distance(item), item["info"]["clouds"]))
    chosen = [anchor]
    tile = anchor.get("mgrs_tile")
    same_tile = [i for i in items if i is not anchor and i.get("mgrs_tile") == tile]
    pool = same_tile if len(same_tile) >= count - 1 else sorted(
        same_tile + [i for i in items if i not in same_tile],
        key=lambda i: (i.get("mgrs_tile") != tile, i["info"]["clouds"]),
    )
    for item in sorted(pool, key=lambda i: (i["info"]["clouds"], distance(i))):
        if len(chosen) >= count:
            break
        if any(existing["info"]["date"] == item["info"]["date"] for existing in chosen):
            continue
        chosen.append(item)
    if len(chosen) < count:
        raise RuntimeError(f"only {len(chosen)} usable scenes; need {count}")
    return sorted(chosen, key=lambda i: i["info"]["date"])
