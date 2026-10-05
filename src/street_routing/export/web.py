"""Static files for the web map (no server): boundaries to zoom from Brazil to a street, and the street graph.

    python -m street_routing.export.web --uf PR --municipality 4113700 --out web_assets

Writes (coordinates in WGS84, 5 decimals = ~1 m):

* ``places.json`` - search index: every state and municipality with its bounding box, the city's neighbourhoods;
* ``states.geojson`` - the 27 states; ``municipalities/<uf_id>.geojson`` - municipalities of one state;
* ``bairros/<municipality>.geojson`` - neighbourhoods (Censo 2022) of the routed city;
* ``graph/<municipality>.json`` - the routable street graph: flat node coordinates, edges as
  ``[u, v, length in decimetres, name index]``, one simplified polyline per edge, street names.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import mapping, shape

from street_routing.graph.ibge_graph import UTM_22S, CorridorParams, build_graph
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


def export_bairros(src: IbgeSource, uf: str, municipality: str, out: Path) -> list[dict]:
    df = src.bairros(uf, municipality).to_crs("EPSG:4326")
    feats, entries = [], []
    for _, row in df.iterrows():
        props = {"code": str(row["CD_BAIRRO"]), "name": row["NM_BAIRRO"]}
        feats.append(_feature(row.geometry, props))
        entries.append({**props, "municipality": int(municipality), "bbox": _bbox(row.geometry)})
    _write(out / "bairros" / f"{municipality}.geojson", {"type": "FeatureCollection", "features": feats})
    return entries


def graph_payload(g: nx.MultiGraph, municipality: dict, params: CorridorParams) -> dict:
    to_wgs = Transformer.from_crs(UTM_22S, "EPSG:4326", always_xy=True)
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
        "municipality": municipality,
        "source": "IBGE, Base de Faces de Logradouros, Censo 2022 (street corridors -> skeleton -> graph)",
        "params": params.__dict__,
        "nodes": _round([float(c) for pair in zip(lon, lat) for c in pair]),
        "edges": edges,
        "lines": lines,
        "names": names,
        "bridged": bridged,  # edges that are estimated links between disconnected parts, not IBGE streets
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uf", default="PR")
    ap.add_argument("--municipality", default="4113700", help="IBGE code (4113700 = Londrina)")
    ap.add_argument("--cache", type=Path, default=Path("data/raw/ibge"))
    ap.add_argument("--out", type=Path, default=Path("outputs/web_assets"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    src = IbgeSource(a.cache)
    places = export_boundaries(src, a.out)
    city = next(m for m in places["municipalities"] if str(m["code"]) == a.municipality)
    places["bairros"] = export_bairros(src, a.uf, a.municipality, a.out)
    places["routed"] = [city["code"]]
    _write(a.out / "places.json", places)
    params = CorridorParams()
    g = build_graph(src.faces(a.uf, a.municipality), params)
    size = _write(a.out / "graph" / f"{a.municipality}.json", graph_payload(g, {"code": city["code"], "name": city["name"]}, params))
    links = [d["length"] for *_, d in g.edges(data=True) if d.get("bridged")]
    LOGGER.info("%s: %d nodes, %d edges, %.1f km of streets, %d estimated links (%.2f km), graph file %.1f MB",
                city["name"], g.number_of_nodes(), g.number_of_edges(),
                sum(d["length"] for *_, d in g.edges(data=True)) / 1000, len(links), sum(links) / 1000, size / 1e6)


if __name__ == "__main__":
    main()
