"""One-way streets from OpenStreetMap (© OpenStreetMap contributors, ODbL 1.0), fetched once per area from the
public Overpass API and cached under ``data/raw/osm``.

Only the driving direction is taken from OSM: the street network itself stays the one rebuilt from IBGE block
faces. A way is one-way when tagged ``oneway=yes|1|true`` (direction of its nodes), ``oneway=-1|reverse``
(opposite direction), or implied by OSM conventions: ``junction=roundabout|circular`` and ``highway=motorway``
(unless ``oneway=no``). Areas are split into tiles so each request stays small (Overpass usage policy: sequential
requests, an identifying User-Agent, a pause between requests).

    python -m street_routing.sources.osm --area 3550308      # prefetch one area (all areas without --area)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("street_routing.osm")
OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "street-routing research (UTFPR; github.com/TiagoRodrigues-GitH/georeferencing-street-routing-algorithms)"
ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"
TILE_DEG = 0.2
PAUSE_S = 2.0

QUERY = """[out:json][timeout:240];
(
  way["highway"]["oneway"~"^(yes|1|true|-1|reverse)$"]({s},{w},{n},{e});
  way["highway"]["junction"~"^(roundabout|circular)$"]["oneway"!="no"]({s},{w},{n},{e});
  way["highway"~"^motorway$"]["oneway"!="no"]({s},{w},{n},{e});
);
out tags geom;"""


@dataclass(frozen=True)
class OnewayWay:
    """A one-way way as (lon, lat) points in its driving direction."""

    id: int
    coords: tuple[tuple[float, float], ...]
    highway: str


def direction(tags: dict) -> int:
    """+1: drive in node order; -1: against node order; 0: not one-way."""
    oneway = str(tags.get("oneway", "")).lower()
    if oneway in {"yes", "1", "true"}:
        return 1
    if oneway in {"-1", "reverse"}:
        return -1
    if oneway == "no":
        return 0
    if tags.get("junction") in {"roundabout", "circular"} or tags.get("highway") == "motorway":
        return 1
    return 0


def parse(payload: dict) -> list[OnewayWay]:
    out = []
    for el in payload.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        d = direction(el.get("tags", {}))
        if d == 0:
            continue
        pts = tuple((p["lon"], p["lat"]) for p in el["geometry"])
        out.append(OnewayWay(el["id"], pts if d > 0 else pts[::-1], el.get("tags", {}).get("highway", "")))
    return out


def tiles(bbox: tuple[float, float, float, float], size: float = TILE_DEG) -> list[tuple[float, float, float, float]]:
    w, s, e, n = bbox
    nx, ny = max(1, math.ceil((e - w) / size)), max(1, math.ceil((n - s) / size))
    dx, dy = (e - w) / nx, (n - s) / ny
    return [(w + i * dx, s + j * dy, w + (i + 1) * dx, s + (j + 1) * dy) for i in range(nx) for j in range(ny)]


def _fetch(bbox: tuple[float, float, float, float], path: Path, retries: int = 4) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    w, s, e, n = bbox
    data = urllib.parse.urlencode({"data": QUERY.format(s=s, w=w, n=n, e=e)}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OVERPASS, data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.loads(r.read().decode("utf-8"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(PAUSE_S)
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:  # 429/504 when busy: wait, retry
            wait = 30 * (attempt + 1)
            LOGGER.warning("Overpass %s (attempt %d), retrying in %d s", err, attempt + 1, wait)
            time.sleep(wait)
    raise RuntimeError(f"Overpass failed for {bbox}")


def oneway_ways(bbox: tuple[float, float, float, float], cache: Path, key: str) -> list[OnewayWay]:
    """One-way ways intersecting ``bbox`` (west, south, east, north); duplicates across tiles removed."""
    seen: dict[int, OnewayWay] = {}
    parts = tiles(bbox)
    for i, t in enumerate(parts):
        payload = _fetch(t, cache / key / f"{i:03d}.json")
        for way in parse(payload):
            seen.setdefault(way.id, way)
        LOGGER.info("%s: tile %d/%d, %d one-way ways so far", key, i + 1, len(parts), len(seen))
    return list(seen.values())


def main() -> None:
    from street_routing.export.areas import AREAS, area

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", action="append")
    ap.add_argument("--places", type=Path, default=Path("outputs/web_assets/places.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw/osm"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    boxes = {r["id"]: tuple(r["bbox"]) for r in json.loads(a.places.read_text(encoding="utf-8"))["routed"]}
    for x in [area(i) for i in a.area] if a.area else AREAS:
        ways = oneway_ways(boxes[x.id], a.cache, x.id)
        LOGGER.info("%s: %d one-way ways", x.name, len(ways))


if __name__ == "__main__":
    main()
