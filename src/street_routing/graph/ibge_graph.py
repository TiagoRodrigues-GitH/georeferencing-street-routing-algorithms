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
   (Londrina: 17 links, 0.56 km, raise the connected share from 70% to 97.6% of the street length);
6. junctions closer than ``detour_gap_m`` whose road distance is over ``max_detour`` times their gap are joined
   the same way, when the link crosses no block face and no street: the inner sides of an avenue's carriageways
   face no block, so the openings of a wide central median are missing and the two carriageways only meet far
   away (Londrina: node pairs 150-600 m apart with a detour over 5x drop from 13.9% to 3.8%);
7. connected parts with at least ``min_component_km`` of streets are kept, and smaller fragments dropped. A city
   can stay in several parts when the roads between them have no facing blocks over more than ``max_gap_m``
   (the island and the mainland of Florianopolis, joined by bridges; the administrative regions of Brasilia,
   joined by highways): no link is invented there, so a route between two parts does not exist in these data.

All lengths are metres in SIRGAS 2000 / UTM, in the zone of the city (``utm_crs``: 22S for Londrina, Curitiba and
Florianopolis, 23S for Sao Paulo and Brasilia); the IBGE data have no speeds or one-way streets, so routes minimise
distance.

Large cities are rasterised in tiles (``tile_m``) with an overlap of ``tile_margin_m``: closing, hole filling and
thinning only look a few tens of metres around a pixel, so the skeleton of each tile's core is the one the whole
image would give (``tests/test_graph.py`` checks it), and only the skeleton pixels are kept (``skeleton``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
from rasterio import features
from rasterio.transform import Affine, from_origin
from scipy import ndimage
from shapely.geometry import LineString, box
from shapely.geometry.base import BaseGeometry
from skimage.morphology import remove_small_holes, skeletonize

from street_routing.graph.skeleton import sparse_skeleton_to_graph

LOGGER = logging.getLogger("street_routing.graph")
UTM_22S = "EPSG:31982"
_SIRGAS_UTM_SOUTH_BASE = 31960  # EPSG code of SIRGAS 2000 / UTM zone NS = 31960 + N (18S..25S cover Brazil)


@dataclass(frozen=True)
class CorridorParams:
    half_width_m: float = 8.0     # facing faces closer than 2x this merge into one street
    resolution_m: float = 2.0
    close_iterations: int = 1     # binary closing (3x3) to join faces broken at corners
    min_spur_m: float = 20.0
    max_gap_m: float = 100.0      # parts of the network closer than this are joined by an estimated link
    detour_gap_m: float = 100.0   # junctions closer than this (carriageways of a wide median: 80-90 m) ...
    max_detour: float = 10.0      # ... with a road distance over this many times their gap get an estimated link
    simplify_m: float = 2.0
    min_component_km: float = 20.0  # connected parts with less street length than this are dropped
    tile_m: float = 8000.0        # rasterised in tiles of this size ...
    tile_margin_m: float = 400.0  # ... each drawn with this overlap so that its core is exact


def street_name(row) -> str:
    parts = [row.get("NM_TIP_LOG", ""), row.get("NM_TIT_LOG", ""), row.get("NM_LOG", "")]
    return " ".join(p.strip().title() for p in parts if isinstance(p, str) and p.strip())


def utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """SIRGAS 2000 / UTM south zone of the centre of ``gdf`` (Brazil lies south of the equator but for a strip in
    the north; distances there are still metric, only slightly more distorted)."""
    minx, miny, maxx, maxy = gpd.GeoSeries([box(*gdf.total_bounds)], crs=gdf.crs).to_crs("EPSG:4326").total_bounds
    zone = int(((minx + maxx) / 2 + 180) // 6) + 1
    return f"EPSG:{_SIRGAS_UTM_SOUTH_BASE + zone}"


def clean_corridor(mask: np.ndarray, p: CorridorParams) -> np.ndarray:
    mask = ndimage.binary_closing(mask.astype(bool), structure=np.ones((3, 3), bool), iterations=p.close_iterations)
    # Faces stop at the block corners, so every crossing is an empty square inside the corridor; left open, the
    # skeleton would circle it. Holes smaller than (3 x half width)^2 are filled; city blocks are far larger.
    max_hole_px = int((3 * p.half_width_m / p.resolution_m) ** 2)
    return remove_small_holes(mask, max_size=max_hole_px)


@dataclass(frozen=True)
class Skeleton:
    """Skeleton pixels as sorted keys ``row * width + col`` of a raster with the given ``transform``."""

    keys: np.ndarray
    width: int
    transform: Affine


def skeleton(faces: gpd.GeoDataFrame, p: CorridorParams) -> Skeleton:
    """Street centre lines of the corridor of ``faces`` (projected, metres), drawn tile by tile."""
    minx, miny, maxx, maxy = faces.total_bounds
    res = p.resolution_m
    pad = 4 * p.half_width_m
    width = int(np.ceil((maxx - minx + 2 * pad) / res))
    height = int(np.ceil((maxy - miny + 2 * pad) / res))
    transform = from_origin(minx - pad, maxy + pad, res, res)
    buffers = faces.geometry.buffer(p.half_width_m, cap_style="flat").reset_index(drop=True)
    tree = buffers.sindex
    tile, margin = max(1, int(p.tile_m / res)), int(np.ceil(p.tile_margin_m / res))
    parts: list[np.ndarray] = [np.zeros(0, np.int64)]
    for r0 in range(0, height, tile):
        for c0 in range(0, width, tile):
            r1, c1 = min(r0 + tile, height), min(c0 + tile, width)
            wr0, wc0 = max(0, r0 - margin), max(0, c0 - margin)
            wr1, wc1 = min(height, r1 + margin), min(width, c1 + margin)
            x0, top = transform.c + wc0 * res, transform.f - wr0 * res
            x1, bottom = transform.c + wc1 * res, transform.f - wr1 * res
            idx = tree.query(box(x0, bottom, x1, top))
            if len(idx) == 0:
                continue
            mask = features.rasterize(((buffers.iloc[i], 1) for i in idx), out_shape=(wr1 - wr0, wc1 - wc0),
                                      transform=from_origin(x0, top, res, res), fill=0, dtype="uint8")
            core = skeletonize(clean_corridor(mask, p))[r0 - wr0:r1 - wr0, c0 - wc0:c1 - wc0]
            rr, cc = np.nonzero(core)
            parts.append((rr.astype(np.int64) + r0) * width + (cc + c0))
    return Skeleton(np.unique(np.concatenate(parts)), width, transform)


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


def _detour_limit(gap: float, max_detour: float) -> float:
    return max_detour * max(gap, 10.0)


def _detour_candidates(g: nx.MultiGraph, pairs: list[tuple[float, int, int]], max_detour: float
                       ) -> list[tuple[float, int, int]]:
    """Pairs whose road distance on ``g`` is over their detour limit, in the order given. Links only shorten road
    distances, so a pair that is short enough now stays so; one bounded Dijkstra per source node replaces one per
    pair (Sao Paulo: ~10^6 pairs)."""
    by_source: dict[int, list[tuple[float, int, int]]] = {}
    for pair in pairs:
        by_source.setdefault(pair[1], []).append(pair)
    keep: set[tuple[float, int, int]] = set()
    for u, group in by_source.items():
        reach = nx.single_source_dijkstra_path_length(
            g, u, cutoff=max(_detour_limit(gap, max_detour) for gap, *_ in group), weight="length")
        keep.update(pair for pair in group if reach.get(pair[2], np.inf) > _detour_limit(pair[0], max_detour))
    return [pair for pair in pairs if pair in keep]


def bridge_detours(g: nx.MultiGraph, barriers: Iterable[BaseGeometry], max_gap_m: float, max_detour: float) -> int:
    """Join two nodes at most ``max_gap_m`` apart whose road distance is over ``max_detour`` x their gap (at least
    10 m), nearest pairs first, by a straight link marked ``bridged=True``, unless the link crosses one of the
    ``barriers`` (block faces: it would go through a block) or a street (streets meet at junctions). The road
    distance is re-measured after every link, so one link per gap is enough. Returns the number of links added."""
    from scipy.spatial import cKDTree
    from shapely import STRtree

    nodes = list(g.nodes)
    xy = np.array([(g.nodes[n]["x"], g.nodes[n]["y"]) for n in nodes])
    pairs = sorted((float(np.hypot(*(xy[i] - xy[j]))), nodes[i], nodes[j])
                   for i, j in cKDTree(xy).query_pairs(max_gap_m))
    pairs = _detour_candidates(g, pairs, max_detour)
    barrier_tree = STRtree(list(barriers))
    street_tree = STRtree([d["geometry"] for *_, d in g.edges(data=True)])
    links: list[LineString] = []
    for gap, u, v in pairs:
        try:
            nx.single_source_dijkstra(g, u, target=v, cutoff=_detour_limit(gap, max_detour), weight="length")
            continue  # a short enough road already exists
        except nx.NetworkXNoPath:
            pass
        link = LineString([(g.nodes[u]["x"], g.nodes[u]["y"]), (g.nodes[v]["x"], g.nodes[v]["y"])])
        if (len(barrier_tree.query(link, predicate="intersects")) or len(street_tree.query(link, predicate="crosses"))
                or any(link.crosses(other) for other in links)):
            continue
        g.add_edge(u, v, length=gap, geometry=link, bridged=True)
        links.append(link)
    return len(links)


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


def major_components(g: nx.MultiGraph, min_km: float) -> set[int]:
    """Nodes of the connected parts with at least ``min_km`` of streets (always at least the largest part)."""
    parts = [(sum(d["length"] for *_, d in g.subgraph(c).edges(data=True)), c) for c in nx.connected_components(g)]
    if not parts:
        return set()
    keep = [c for km, c in parts if km >= 1000 * min_km] or [max(parts, key=lambda kc: kc[0])[1]]
    return set().union(*keep)


class _Stopwatch:
    """Logs how long each stage of the build took (Sao Paulo runs for tens of minutes)."""

    def __init__(self) -> None:
        self.t = time.perf_counter()

    def lap(self, stage: str, g: nx.MultiGraph | None = None) -> None:
        now = time.perf_counter()
        size = f" ({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)" if g is not None else ""
        LOGGER.info("%-24s %7.1f s%s", stage, now - self.t, size)
        self.t = now


def build_graph(faces: gpd.GeoDataFrame, p: CorridorParams = CorridorParams()) -> nx.MultiGraph:
    """Faces (any CRS) -> undirected multigraph in the city's UTM zone (``graph["crs"]``) with node ``x, y`` and
    edge ``length, geometry, name``."""
    crs = utm_crs(faces)
    faces = faces.to_crs(crs)
    clock = _Stopwatch()
    sk = skeleton(faces, p)
    clock.lap(f"skeleton ({len(sk.keys)} px)")
    pg = sparse_skeleton_to_graph(sk.keys, sk.width)
    g = nx.MultiGraph()
    node_xy = _pixel_to_xy(pg.nodes, sk.transform)
    for i, (x, y) in enumerate(node_xy):
        g.add_node(i, x=float(x), y=float(y))
    for u, v, path in pg.edges:
        line = LineString(_pixel_to_xy(path, sk.transform))
        if line.length == 0:
            continue
        g.add_edge(u, v, length=float(line.length), geometry=line)
    del pg, sk
    clock.lap("pixel graph", g)
    prune_spurs(g, p.min_spur_m)
    g.remove_edges_from([(u, v, k) for u, v, k in nx.selfloop_edges(g, keys=True)])
    g = collapse_junction_links(g, 1.5 * p.half_width_m)
    clock.lap("spurs and junctions", g)
    bridge_components(g, p.max_gap_m)
    merge_degree_two(g)
    clock.lap("component links", g)
    bridge_detours(g, faces.geometry.to_numpy(), p.detour_gap_m, p.max_detour)
    clock.lap("median links", g)
    g = g.subgraph(major_components(g, p.min_component_km)).copy()
    names = faces.apply(street_name, axis=1).to_numpy()
    tree = faces.sindex
    for _, _, d in g.edges(data=True):
        d["geometry"] = d["geometry"].simplify(p.simplify_m, preserve_topology=False)
        if d.get("bridged"):
            d["name"] = ""  # an estimated link, not a street of the IBGE data
            continue
        idx = tree.nearest(d["geometry"].interpolate(0.5, normalized=True), return_all=False)[1][0]
        d["name"] = names[idx]
    clock.lap("names", g)
    g = nx.convert_node_labels_to_integers(g)
    g.graph["crs"] = crs
    return g
