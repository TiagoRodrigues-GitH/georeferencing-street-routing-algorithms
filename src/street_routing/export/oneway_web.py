"""Adds the driving direction matched from OpenStreetMap to the web graphs written by ``export.web``.

    python -m street_routing.export.oneway_web                  # every routed area
    python -m street_routing.export.oneway_web --area 4113700

Each ``graph/<area>.json`` gains ``oneway`` (edges drivable only from u to v; u/v and the polyline of those edges are
put in the driving direction) and ``directionSource`` with the OSM attribution required by the ODbL. Re-running is
safe: edges already in their driving direction match again with the same result.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from street_routing.export.areas import AREAS, area
from street_routing.graph.oneway import assign_oneway
from street_routing.sources.osm import ATTRIBUTION, oneway_ways

LOGGER = logging.getLogger("street_routing.export")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", action="append")
    ap.add_argument("--out", type=Path, default=Path("outputs/web_assets"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw/osm"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    places = json.loads((a.out / "places.json").read_text(encoding="utf-8"))
    boxes = {r["id"]: tuple(r["bbox"]) for r in places["routed"]}
    for x in [area(i) for i in a.area] if a.area else AREAS:
        path = a.out / "graph" / f"{x.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        ways = oneway_ways(boxes[x.id], a.cache, x.id)
        result = assign_oneway(payload, ways, payload["crs"])
        payload["directionSource"] = f"OpenStreetMap one-way tags, matched to the IBGE edges; {ATTRIBUTION}"
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        path.write_text(text, encoding="utf-8")
        for r in places["routed"]:
            if r["id"] == x.id:
                r["bytes"] = len(text)
        LOGGER.info("%s: %d OSM one-way ways; %d of %d edges matched, %d made two-way again by the repair, %d one-way",
                    x.name, len(ways), result.matched, len(payload["edges"]) // 4, result.reverted, len(result.oneway))
    (a.out / "places.json").write_text(json.dumps(places, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
