"""Static files for the web map (no server): boundaries to zoom from Brazil to a street, and the street graphs.

    python -m street_routing.export.web                      # every routed area of export/areas.py
    python -m street_routing.export.web --area 3550308       # one area; the others already written are kept

Writes (coordinates in WGS84, 5 decimals = ~1 m):

* ``places.json`` - search index: every state and municipality with its bounding box, the neighbourhoods of the
  routed areas, and ``routed``: one entry per area (id, name, member municipalities, box of its street network);
* ``states.geojson`` - the 27 states; ``municipalities/<uf_id>.geojson`` - municipalities of one state;
* ``bairros/<area>.geojson`` - neighbourhoods (Censo 2022) of a routed area, where IBGE publishes them;
* ``graph/<area>.json`` - the routable street graph: flat node coordinates, edges as
  ``[u, v, length in decimetres, name index]``, one simplified polyline per edge, street names.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import mapping, shape

from street_routing.export.areas import AREAS, RoutedArea, area
from street_routing.graph.ibge_graph import CorridorParams, build_graph
from street_routing.sources.ibge import IbgeSource

LOGGER = logging.getLogger("street_routing.export")
DECIMALS = 5


def _round(obj):
    if isinstance(obj, float):
        return round(obj, DECIMALS)
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    return obj


def _feature(geom, props: dict) -> dict:
    g = mapping(geom)
    return {"type": "Feature", "properties": props, "geometry": {"type": g["type"], "coordinates": _round(g["coordinates"])}}


def _write(path: Path, data) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text)


def _bbox(geom) -> list[float]:
    return _round(list(geom.bounds))


def export_boundaries(src: IbgeSource, out: Path) -> dict:
    states = {s["id"]: s for s in src.states()}
    feats, places = [], {"states": [], "municipalities": []}
    for f in src.states_geojson()["features"]:
        code = int(f["properties"]["codarea"])
        geom = shape(f["geometry"])
        s = states[code]
        feats.append(_feature(geom, {"code": code, "uf": s["sigla"], "name": s["nome"]}))
        places["states"].append({"code": code, "uf": s["sigla"], "name": s["nome"], "bbox": _bbox(geom)})
        names = {m["id"]: m["nome"] for m in src.municipalities(code)}
        mfeats = []
        for mf in src.municipalities_geojson(code)["features"]:
            mcode = int(mf["properties"]["codarea"])
            mgeom = shape(mf["geometry"])
            mfeats.append(_feature(mgeom, {"code": mcode, "name": names.get(mcode, str(mcode))}))
            places["municipalities"].append({"code": mcode, "uf": s["sigla"], "state": code,
                                             "name": names.get(mcode, str(mcode)), "bbox": _bbox(mgeom)})
        _write(out / "municipalities" / f"{code}.geojson", {"type": "FeatureCollection", "features": mfeats})
    _write(out / "states.geojson", {"type": "FeatureCollection", "features": feats})
    return places


def export_bairros(src: IbgeSource, a: RoutedArea, out: Path) -> list[dict]:
    """Neighbourhoods of every member municipality in one file; none written where IBGE publishes none."""
    feats, entries = [], []
    for municipality in a.municipalities:
        df = src.bairros(a.uf, str(municipality)).to_crs("EPSG:4326")
        for _, row in df.iterrows():
            props = {"code": str(row["CD_BAIRRO"]), "name": row["NM_BAIRRO"]}
            feats.append(_feature(row.geometry, props))
            entries.append({**props, "municipality": municipality, "area": a.id, "bbox": _bbox(row.geometry)})
    if feats:
        _write(out / "bairros" / f"{a.id}.geojson", {"type": "FeatureCollection", "features": feats})
    return entries


def area_faces(src: IbgeSource, a: RoutedArea) -> gpd.GeoDataFrame:
    frames = [src.faces(a.uf, str(m)) for m in a.municipalities]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def graph_payload(g: nx.MultiGraph, a: RoutedArea, params: CorridorParams) -> dict:
    to_wgs = Transformer.from_crs(g.graph["crs"], "EPSG:4326", always_xy=True)
    xs = np.array([g.nodes[n]["x"] for n in range(g.number_of_nodes())])
    ys = np.array([g.nodes[n]["y"] for n in range(g.number_of_nodes())])
    lon, lat = to_wgs.transform(xs, ys)
    names: list[str] = []
    name_index: dict[str, int] = {}
    edges, lines, bridged = [], [], []
    for i, (u, v, d) in enumerate(g.edges(data=True)):
        name = d.get("name") or ""
        if name not in name_index:
            name_index[name] = len(names)
            names.append(name)
        elons, elats = to_wgs.transform(*np.asarray(d["geometry"].coords).T)
        edges += [int(u), int(v), int(round(d["length"] * 10)), name_index[name]]
        lines.append(_round([float(c) for pair in zip(elons, elats) for c in pair]))
        if d.get("bridged"):
            bridged.append(i)
    return {
        "municipality": {"code": a.id, "name": a.name},
        "municipalities": list(a.municipalities),
        "source": "IBGE, Base de Faces de Logradouros, Censo 2022 (street corridors -> skeleton -> graph)",
        "crs": g.graph["crs"],
        "params": params.__dict__,
        "nodes": _round([float(c) for pair in zip(lon, lat) for c in pair]),
        "edges": edges,
        "lines": lines,
        "names": names,
        "bridged": bridged,  # edges that are estimated links between disconnected parts, not IBGE streets
    }


def _payload_bbox(payload: dict) -> list[float]:
    nodes = np.asarray(payload["nodes"]).reshape(-1, 2)
    return _round([float(nodes[:, 0].min()), float(nodes[:, 1].min()), float(nodes[:, 0].max()), float(nodes[:, 1].max())])


def export_area(src: IbgeSource, a: RoutedArea, out: Path, params: CorridorParams) -> dict:
    g = build_graph(area_faces(src, a), params)
    payload = graph_payload(g, a, params)
    size = _write(out / "graph" / f"{a.id}.json", payload)
    links = [d["length"] for *_, d in g.edges(data=True) if d.get("bridged")]
    LOGGER.info("%s: %d nodes, %d edges, %.1f km of streets, %d estimated links (%.2f km), graph file %.1f MB",
                a.name, g.number_of_nodes(), g.number_of_edges(),
                sum(d["length"] for *_, d in g.edges(data=True)) / 1000, len(links), sum(links) / 1000, size / 1e6)
    return {"id": a.id, "name": a.name, "municipalities": list(a.municipalities), "bbox": _payload_bbox(payload),
            "bytes": size}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", action="append", help="routed area id (repeatable; default: all of export/areas.py)")
    ap.add_argument("--cache", type=Path, default=Path("data/raw/ibge"))
    ap.add_argument("--out", type=Path, default=Path("outputs/web_assets"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    src = IbgeSource(a.cache)
    chosen = [area(i) for i in a.area] if a.area else list(AREAS)
    rebuilt = {x.id for x in chosen}
    previous = a.out / "places.json"
    old = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    routed = {r["id"]: r for r in old.get("routed", []) if isinstance(r, dict) and r["id"] not in rebuilt}
    bairros = [b for b in old.get("bairros", []) if b.get("area") not in rebuilt and b.get("area") in routed]
    places = export_boundaries(src, a.out)
    params = CorridorParams()
    for x in chosen:
        bairros += export_bairros(src, x, a.out)
        routed[x.id] = export_area(src, x, a.out, params)
        places["bairros"] = bairros
        places["routed"] = [routed[r.id] for r in AREAS if r.id in routed]
        _write(a.out / "places.json", places)  # after every area, so a long run can be stopped and resumed


if __name__ == "__main__":
    main()
