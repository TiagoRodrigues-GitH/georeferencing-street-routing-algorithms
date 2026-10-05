"""Routable street graph of a municipality from the IBGE *Base de Faces de Logradouros* (Censo 2022).

IBGE publishes block faces, not street centre lines: each street appears as the two block sides that face it,
a few metres apart, and the faces stop at the block corners, so noding the lines would give two disconnected
parallel networks. The street space between facing blocks is recovered instead:

1. every face is buffered by ``half_width_m`` (the corridor of facing faces merges into one street);
2. the buffers are rasterised at ``resolution_m`` (2 m = the CBERS-4A pan-fused resolution used in milestone 3);
3. small gaps are closed, the holes the faces leave at every crossing are filled, and the corridor is thinned to
   a one-pixel skeleton;
4. the skeleton becomes a graph (``skeleton.skeleton_to_graph``); spurs shorter than ``min_spur_m`` (buffer ends,
   corners) are removed, junctions split by the thinning (linked by less than 1.5 x ``half_width_m``) are merged,
   edge geometry is simplified and every edge gets the name of the nearest face;
5. parts of the network cut off by roads that have no facing blocks (avenues along parks and lakes, bridges,
   highways) are joined to the rest by straight links across gaps up to ``max_gap_m``, marked ``bridged``
   (Londrina: 17 links, 0.56 km, raise the connected share from 70% to 97.6% of the street length), and the
   largest connected component is kept (routing needs one network).

All lengths are metres in SIRGAS 2000 / UTM zone 22S (EPSG:31982); the IBGE data have no speeds or one-way
streets, so routes minimise distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import LineString
from skimage.morphology import remove_small_holes, skeletonize

from street_routing.graph.skeleton import skeleton_to_graph

UTM_22S = "EPSG:31982"


@dataclass(frozen=True)
class CorridorParams:
    half_width_m: float = 8.0     # facing faces closer than 2x this merge into one street
    resolution_m: float = 2.0
    close_iterations: int = 1     # binary closing (3x3) to join faces broken at corners
    min_spur_m: float = 20.0
    max_gap_m: float = 100.0      # parts of the network closer than this are joined by an estimated link
    simplify_m: float = 2.0


def street_name(row) -> str:
    parts = [row.get("NM_TIP_LOG", ""), row.get("NM_TIT_LOG", ""), row.get("NM_LOG", "")]
    return " ".join(p.strip().title() for p in parts if isinstance(p, str) and p.strip())


def corridor_mask(faces: gpd.GeoDataFrame, p: CorridorParams) -> tuple[np.ndarray, object]:
    minx, miny, maxx, maxy = faces.total_bounds
    pad = 4 * p.half_width_m
    width = int(np.ceil((maxx - minx + 2 * pad) / p.resolution_m))
    height = int(np.ceil((maxy - miny + 2 * pad) / p.resolution_m))
    transform = from_origin(minx - pad, maxy + pad, p.resolution_m, p.resolution_m)
    shapes = ((geom, 1) for geom in faces.geometry.buffer(p.half_width_m, cap_style="flat"))
    mask = features.rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
    mask = ndimage.binary_closing(mask.astype(bool), structure=np.ones((3, 3), bool), iterations=p.close_iterations)
    # Faces stop at the block corners, so every crossing is an empty square inside the corridor; left open, the
    # skeleton would circle it. Holes smaller than (3 x half width)^2 are filled; city blocks are far larger.
    max_hole_px = int((3 * p.half_width_m / p.resolution_m) ** 2)
    mask = remove_small_holes(mask, max_size=max_hole_px)
    return mask, transform


def _pixel_to_xy(path: np.ndarray, transform) -> np.ndarray:
    rows, cols = path[:, 0] + 0.5, path[:, 1] + 0.5
    xs = transform.c + cols * transform.a
    ys = transform.f + rows * transform.e
    return np.column_stack([xs, ys])


def prune_spurs(g: nx.MultiGraph, min_len: float) -> None:
    """Remove dead-end edges shorter than ``min_len`` until none is left."""
    while True:
        spurs = [(u, v, k) for u, v, k, d in g.edges(keys=True, data=True)
                 if d["length"] < min_len and (g.degree(u) == 1 or g.degree(v) == 1) and u != v]
        if not spurs:
            return
        g.remove_edges_from(spurs)
        g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])


def collapse_junction_links(g: nx.MultiGraph, max_len: float) -> nx.MultiGraph:
    """Thinning splits a crossing into two or more junctions joined by links a few metres long; contract every
    link shorter than ``max_len`` between junctions (degree >= 3) into one node at the members' mean position."""
    parent = {n: n for n in g.nodes}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v, d in g.edges(data=True):
        if u != v and d["length"] < max_len and g.degree(u) >= 3 and g.degree(v) >= 3:
            parent[find(u)] = find(v)
    members: dict[int, list[int]] = {}
    for n in g.nodes:
        members.setdefault(find(n), []).append(n)
    out = nx.MultiGraph()
    for rep, group in members.items():
        out.add_node(rep, x=float(np.mean([g.nodes[n]["x"] for n in group])),
                     y=float(np.mean([g.nodes[n]["y"] for n in group])))
    for u, v, d in g.edges(data=True):
        ru, rv = find(u), find(v)
        if ru != rv:  # links inside a merged crossing disappear
            out.add_edge(ru, rv, **d)
    return out


def bridge_components(g: nx.MultiGraph, max_gap_m: float) -> int:
    """Join disconnected parts of the network by straight links between their closest junctions, shortest first
    (Boruvka rounds), when the gap is at most ``max_gap_m``. Roads without facing blocks (avenues along parks and
    lakes, bridges, highways) have no IBGE faces, so whole districts would otherwise be cut off. Links are marked
    ``bridged=True`` so they can be shown and counted separately. Returns the number of links added."""
    from scipy.spatial import cKDTree

    nodes = list(g.nodes)
    xy = np.array([(g.nodes[n]["x"], g.nodes[n]["y"]) for n in nodes])
    tree = cKDTree(xy)
    added = 0
    while True:
        comp_of = {}
        for i, comp in enumerate(nx.connected_components(g)):
            for n in comp:
                comp_of[n] = i
        if max(comp_of.values()) == 0:
            return added
        best: dict[int, tuple[float, int, int]] = {}  # component -> (gap, u, v)
        k = min(len(nodes), 32)
        dists, idx = tree.query(xy, k=k, distance_upper_bound=max_gap_m)
        for a, (drow, irow) in enumerate(zip(dists, idx)):
            ca = comp_of[nodes[a]]
            for dist, b in zip(drow, irow):
                if not np.isfinite(dist):
                    break
                if comp_of[nodes[b]] != ca:
                    if ca not in best or dist < best[ca][0]:
                        best[ca] = (float(dist), nodes[a], nodes[b])
                    break
        merged = 0
        for gap, u, v in sorted(best.values()):
            if not nx.has_path(g, u, v):
                line = LineString([(g.nodes[u]["x"], g.nodes[u]["y"]), (g.nodes[v]["x"], g.nodes[v]["y"])])
                g.add_edge(u, v, length=gap, geometry=line, bridged=True)
                merged += 1
        added += merged
        if merged == 0:
            return added


def _oriented(line: LineString, start_xy: tuple[float, float]) -> list[tuple[float, float]]:
    """Coordinates of ``line`` starting at the end closer to ``start_xy``."""
    coords = list(line.coords)
    first, last = np.hypot(*np.subtract(coords[0], start_xy)), np.hypot(*np.subtract(coords[-1], start_xy))
    return coords if first <= last else coords[::-1]


def merge_degree_two(g: nx.MultiGraph) -> None:
    """Join the two edges of every node that is not a junction or an end (left behind by spur removal)."""
    for n in list(g.nodes):
        if n not in g or g.degree(n) != 2:
            continue
        (_, a, ka, da), (_, b, kb, db) = list(g.edges(n, keys=True, data=True))
        if a == n or b == n or a == b:  # self-loop, or two edges closing a loop: keep the node
            continue
        if bool(da.get("bridged")) != bool(db.get("bridged")):  # keep estimated links apart from real streets
            continue
        a_xy, n_xy = (g.nodes[a]["x"], g.nodes[a]["y"]), (g.nodes[n]["x"], g.nodes[n]["y"])
        coords = _oriented(da["geometry"], a_xy) + _oriented(db["geometry"], n_xy)[1:]
        g.remove_node(n)
        g.add_edge(a, b, length=da["length"] + db["length"], geometry=LineString(coords),
                   bridged=bool(da.get("bridged") or db.get("bridged")))


def build_graph(faces: gpd.GeoDataFrame, p: CorridorParams = CorridorParams()) -> nx.MultiGraph:
    """Faces (any CRS) -> undirected multigraph in EPSG:31982 with node ``x, y`` and edge ``length, geometry,
    name``."""
    faces = faces.to_crs(UTM_22S)
    mask, transform = corridor_mask(faces, p)
    pg = skeleton_to_graph(skeletonize(mask))
    g = nx.MultiGraph()
    node_xy = _pixel_to_xy(pg.nodes, transform)
    for i, (x, y) in enumerate(node_xy):
        g.add_node(i, x=float(x), y=float(y))
    for u, v, path in pg.edges:
        line = LineString(_pixel_to_xy(path, transform))
        if line.length == 0:
            continue
        g.add_edge(u, v, length=float(line.length), geometry=line)
    prune_spurs(g, p.min_spur_m)
    g.remove_edges_from([(u, v, k) for u, v, k in nx.selfloop_edges(g, keys=True)])
    g = collapse_junction_links(g, 1.5 * p.half_width_m)
    bridge_components(g, p.max_gap_m)
    merge_degree_two(g)
    largest = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest).copy()
    names = faces.apply(street_name, axis=1).to_numpy()
    tree = faces.sindex
    for _, _, d in g.edges(data=True):
        d["geometry"] = d["geometry"].simplify(p.simplify_m, preserve_topology=False)
        if d.get("bridged"):
            d["name"] = ""  # an estimated link, not a street of the IBGE data
            continue
        idx = tree.nearest(d["geometry"].interpolate(0.5, normalized=True), return_all=False)[1][0]
        d["name"] = names[idx]
    return nx.convert_node_labels_to_integers(g)
