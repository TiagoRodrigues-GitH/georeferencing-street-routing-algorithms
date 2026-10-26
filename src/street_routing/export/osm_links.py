"""Joins the disconnected parts of a web graph with roads from OpenStreetMap (© OpenStreetMap contributors, ODbL).

    python -m street_routing.export.osm_links --area 5300108      # after export.web and export.oneway_web

IBGE block faces only exist where a street has blocks on its side, so highways and wide avenues through parks
(Brasília's EPIA, Eixo Monumental, EPTG, the roads between the administrative regions) have no faces and the city
falls apart into dozens of parts that straight links of ``max_gap_m`` cannot join. Here the drivable OSM ways of
the area are cut at their junctions; a piece that already runs along an IBGE edge is dropped, and every other piece
becomes an edge of the graph, its ends joined to the nearest IBGE junction (short connectors, marked ``bridged``).
OSM one-way tags are kept on the new edges. At the end only the largest strongly connected part is kept, so every
origin reaches every destination; the payload gains ``linkSource`` with the attribution.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from street_routing.export.areas import area
from street_routing.sources.osm import ATTRIBUTION, _fetch, direction, tiles

LOGGER = logging.getLogger("street_routing.export")

ROADS = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|" \
        "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
QUERY = """[out:json][timeout:240];
way["highway"~"^({roads})$"]["area"!="yes"]["access"!~"^(private|no)$"]({{s}},{{w}},{{n}},{{e}});
out body geom;""".format(roads=ROADS)
MAJOR = set(ROADS.split("|")) - {"residential", "living_street"}

COVER_M = 15.0      # a sample of an OSM piece this close to an IBGE edge is "already in the graph"
COVERED_SHARE = 0.8  # a piece with this share of its samples covered is dropped
SNAP_M = 25.0       # an OSM end this close to an IBGE junction becomes that junction
CONNECT_M = 60.0    # otherwise a straight connector up to this length joins it to the nearest junction
STEP_M = 5.0        # spacing of the samples


def _round(xs):
    return [round(x, 5) for x in xs]


def _densify(xy: np.ndarray, step: float) -> np.ndarray:
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        k = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        t = np.arange(1, k + 1)[:, None] / k
        out.append(a + t * (b - a))
    return np.vstack(out)


def osm_roads(bbox, cache: Path, key: str) -> list[dict]:
    """Drivable ways intersecting ``bbox``, de-duplicated across tiles, as dicts with nodes, coords and tags."""
    seen: dict[int, dict] = {}
    parts = tiles(bbox)
    for i, t in enumerate(parts):
        for el in _fetch(t, cache / key / "roads" / f"{i:03d}.json", query=QUERY).get("elements", []):
            if el.get("type") == "way" and "geometry" in el and len(el["nodes"]) == len(el["geometry"]):
                seen.setdefault(el["id"], el)
        LOGGER.info("%s: roads tile %d/%d, %d ways so far", key, i + 1, len(parts), len(seen))
    return list(seen.values())


def pieces(ways: list[dict]) -> list[tuple[int, int, list[tuple[float, float]], str, int]]:
    """Ways cut at the nodes they share (and at their ends): (osm node a, osm node b, lon/lat points, name, dir)."""
    use = Counter(n for w in ways for n in w["nodes"])
    out = []
    for w in ways:
        tags = w.get("tags", {})
        name = tags.get("name") or tags.get("ref") or ""
        d = direction(tags)
        pts = [(p["lon"], p["lat"]) for p in w["geometry"]]
        start = 0
        for i in range(1, len(pts)):
            if i == len(pts) - 1 or use[w["nodes"][i]] > 1:
                seg = pts[start:i + 1]
                a, b = w["nodes"][start], w["nodes"][i]
                if d < 0:
                    seg, a, b = seg[::-1], b, a
                out.append((a, b, seg, name, d))
                start = i
    return out


def connect(payload: dict, ways: list[dict]) -> dict:
    to_xy = Transformer.from_crs("EPSG:4326", payload["crs"], always_xy=True)
    nodes = np.asarray(payload["nodes"], dtype=float).reshape(-1, 2)
    nxy = np.column_stack(to_xy.transform(nodes[:, 0], nodes[:, 1]))
    edges = np.asarray(payload["edges"], dtype=np.int64).reshape(-1, 4)
    lines, names = payload["lines"], payload["names"]
    bridged, oneway = set(payload.get("bridged", [])), set(payload.get("oneway", []))
    n0, m0 = len(nodes), len(edges)

    # samples along the IBGE edges: what is already in the graph
    samples = []
    for line in lines:
        ll = np.asarray(line).reshape(-1, 2)
        samples.append(_densify(np.column_stack(to_xy.transform(ll[:, 0], ll[:, 1])), STEP_M))
    ibge = cKDTree(np.vstack(samples))
    junctions = cKDTree(nxy)

    name_index = {s: i for i, s in enumerate(names)}
    new_nodes: list[tuple[float, float]] = []   # lon, lat of nodes added for OSM ends
    node_of: dict[int, int] = {}                 # OSM node -> graph node
    new_edges: list[tuple[int, int, int, int]] = []
    new_lines: list[list[float]] = []
    new_bridged, new_oneway = [], []

    def name_id(s: str) -> int:
        if s not in name_index:
            name_index[s] = len(names)
            names.append(s)
        return name_index[s]

    def add_edge(u, v, line_ll, length, name, is_link=False, is_oneway=False):
        e = m0 + len(new_edges)
        new_edges.append((u, v, int(round(length * 10)), name_id(name)))
        new_lines.append(_round([c for p in line_ll for c in p]))
        if is_link:
            new_bridged.append(e)
        if is_oneway:
            new_oneway.append(e)

    def end_node(osm_id: int, lon: float, lat: float, x: float, y: float) -> int:
        if osm_id in node_of:
            return node_of[osm_id]
        dist, j = junctions.query((x, y))
        if dist <= SNAP_M:
            node_of[osm_id] = int(j)
            return int(j)
        v = n0 + len(new_nodes)
        new_nodes.append((lon, lat))
        node_of[osm_id] = v
        if dist <= CONNECT_M:  # a short straight connector to the IBGE network
            add_edge(v, int(j), [(lon, lat), tuple(nodes[j])], float(dist), "", is_link=True)
        return v

    kept = dropped = 0
    # residential streets are the IBGE's own; the gaps are the roads between them
    for a, b, seg, name, d in pieces([w for w in ways if w.get("tags", {}).get("highway") in MAJOR]):
        ll = np.asarray(seg)
        xy = np.column_stack(to_xy.transform(ll[:, 0], ll[:, 1]))
        length = float(np.hypot(*np.diff(xy, axis=0).T).sum())
        if length == 0:
            continue
        dist, _ = ibge.query(_densify(xy, STEP_M), distance_upper_bound=COVER_M)
        if np.mean(np.isfinite(dist)) >= COVERED_SHARE:
            dropped += 1
            continue
        u = end_node(a, *seg[0], *xy[0])
        v = end_node(b, *seg[-1], *xy[-1])
        if u == v:
            continue
        add_edge(u, v, seg, length, name, is_oneway=d != 0)
        kept += 1
    LOGGER.info("OSM pieces: %d added, %d already along IBGE edges", kept, dropped)

    all_nodes = np.vstack([nodes, np.asarray(new_nodes).reshape(-1, 2)])
    all_edges = np.vstack([edges, np.asarray(new_edges, dtype=np.int64).reshape(-1, 4)])
    all_lines = lines + new_lines
    bridged |= set(new_bridged)
    oneway |= set(new_oneway)
    alive = _prune_spurs(len(all_nodes), all_edges, n0)
    keep_nodes = _strong_part(len(all_nodes), all_edges[alive], {i for i, e in enumerate(alive) if e in oneway})

    # renumber: only the strongly connected part, edges in their original order
    new_id = -np.ones(len(all_nodes), dtype=np.int64)
    new_id[keep_nodes] = np.arange(len(keep_nodes))
    keep_edges = [e for e in range(len(all_edges)) if new_id[all_edges[e, 0]] >= 0 and new_id[all_edges[e, 1]] >= 0]
    edge_id = {e: i for i, e in enumerate(keep_edges)}
    out = dict(payload)
    out["nodes"] = _round(all_nodes[keep_nodes].ravel().tolist())
    out["edges"] = [int(x) for e in keep_edges for x in (new_id[all_edges[e, 0]], new_id[all_edges[e, 1]],
                                                          all_edges[e, 2], all_edges[e, 3])]
    out["lines"] = [all_lines[e] for e in keep_edges]
    out["names"] = names
    out["bridged"] = sorted(edge_id[e] for e in bridged if e in edge_id)
    out["oneway"] = sorted(edge_id[e] for e in oneway if e in edge_id)
    out["linkSource"] = f"OpenStreetMap roads joining the parts the IBGE faces leave apart; {ATTRIBUTION}"
    km = lambda es: sum(int(all_edges[e, 2]) for e in es) / 10000  # noqa: E731
    LOGGER.info("IBGE streets kept: %.1f of %.1f km; total %d nodes, %d edges, %.1f km; %d OSM edges, %d connectors",
                km([e for e in keep_edges if e < m0]), km(range(m0)), len(keep_nodes), len(keep_edges),
                km(keep_edges), sum(1 for e in keep_edges if e >= m0 and e not in bridged),
                sum(1 for e in keep_edges if e >= m0 and e in bridged))
    return out


def _prune_spurs(n: int, edges: np.ndarray, first_new: int) -> list[int]:
    """Edges left after removing, again and again, the OSM roads that end at an added node with no other road
    (rural roads leading out of the area add nothing to the routes between IBGE streets)."""
    deg = np.bincount(edges[:, :2].ravel(), minlength=n)
    touching: list[list[int]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(edges[:, :2]):
        touching[u].append(e)
        touching[v].append(e)
    dead = np.zeros(len(edges), dtype=bool)
    stack = [v for v in range(first_new, n) if deg[v] <= 1]
    while stack:
        v = stack.pop()
        for e in touching[v]:
            if dead[e]:
                continue
            dead[e] = True
            for w in edges[e, :2]:
                deg[w] -= 1
                if w >= first_new and deg[w] == 1:
                    stack.append(int(w))
    LOGGER.info("pruned %d OSM edges that lead nowhere", int(dead.sum()))
    return [e for e in range(len(edges)) if not dead[e]]


def _strong_part(n: int, edges: np.ndarray, oneway: set[int]) -> np.ndarray:
    """Nodes of the largest strongly connected part (two-way edges count both ways)."""
    u, v = edges[:, 0], edges[:, 1]
    two = np.array([e not in oneway for e in range(len(edges))])
    src = np.concatenate([u, v[two]])
    dst = np.concatenate([v, u[two]])
    k, lab = connected_components(coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)), connection="strong")
    big = np.bincount(lab).argmax()
    LOGGER.info("%d strongly connected parts; the largest has %d of %d nodes", k, int((lab == big).sum()), n)
    return np.flatnonzero(lab == big)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", action="append", required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/web_assets"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw/osm"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    places = json.loads((a.out / "places.json").read_text(encoding="utf-8"))
    for x in [area(i) for i in a.area]:
        path = a.out / "graph" / f"{x.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        r = next(r for r in places["routed"] if r["id"] == x.id)
        payload = connect(payload, osm_roads(tuple(r["bbox"]), a.cache, x.id))
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        path.write_text(text, encoding="utf-8")
        ll = np.asarray(payload["nodes"]).reshape(-1, 2)
        r["bbox"] = _round([float(ll[:, 0].min()), float(ll[:, 1].min()), float(ll[:, 0].max()), float(ll[:, 1].max())])
        r["bytes"] = len(text)
        LOGGER.info("%s: graph file %.1f MB", x.name, len(text) / 1e6)
    (a.out / "places.json").write_text(json.dumps(places, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
