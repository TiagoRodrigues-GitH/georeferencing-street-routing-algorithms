"""Skeleton -> graph and IBGE faces -> routable graph, on small synthetic inputs."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from skimage.morphology import skeletonize

from street_routing.graph.ibge_graph import (
    UTM_22S, CorridorParams, bridge_components, bridge_detours, build_graph, major_components, skeleton, utm_crs,
)
from street_routing.graph.skeleton import skeleton_to_graph, sparse_skeleton_to_graph


def test_cross_has_one_junction_and_four_arms():
    sk = np.zeros((21, 21), bool)
    sk[10, :] = True
    sk[:, 10] = True
    g = skeleton_to_graph(sk)
    degrees = np.bincount(np.array([[u, v] for u, v, _ in g.edges]).ravel(), minlength=len(g.nodes))
    assert len(g.nodes) == 5 and len(g.edges) == 4
    assert sorted(degrees) == [1, 1, 1, 1, 4]


def test_ring_road_is_kept_as_a_closed_cycle():
    yy, xx = np.mgrid[:41, :41]
    radius = np.hypot(yy - 20, xx - 20)
    sk = skeletonize((radius > 12) & (radius < 16))  # a real skeleton of a ring, no junction on it
    g = skeleton_to_graph(sk)
    assert len(g.nodes) >= 1 and len(g.edges) >= 1
    degrees = np.bincount(np.array([[u, v] for u, v, _ in g.edges]).ravel(), minlength=len(g.nodes))
    assert (degrees % 2 == 0).all()                       # a cycle: no dead end
    assert sum(len(p) - 1 for *_, p in g.edges) >= 70     # ~2*pi*14 pixels around


def faces_of_a_crossroad() -> gpd.GeoDataFrame:
    """Two streets 12 m wide crossing at (0, 0): each street is drawn as the block faces on both sides, cut at the
    corners, as IBGE publishes them (no centre line, no shared vertex)."""
    x0, y0, L, half = 500_000.0, 7_400_000.0, 200.0, 6.0
    lines = []
    for sign in (-1, 1):
        lines += [LineString([(x0 - L, y0 + sign * half), (x0 - half, y0 + sign * half)]),
                  LineString([(x0 + half, y0 + sign * half), (x0 + L, y0 + sign * half)]),
                  LineString([(x0 + sign * half, y0 - L), (x0 + sign * half, y0 - half)]),
                  LineString([(x0 + sign * half, y0 + half), (x0 + sign * half, y0 + L)])]
    names = ["RUA"] * len(lines)
    return gpd.GeoDataFrame({"NM_TIP_LOG": names, "NM_TIT_LOG": [""] * len(lines), "NM_LOG": ["A"] * len(lines)},
                            geometry=lines, crs=UTM_22S)


def test_parts_closer_than_the_limit_are_joined_by_one_marked_link():
    g = nx.MultiGraph()
    for n, (x, y) in enumerate([(0, 0), (100, 0), (160, 0), (260, 0), (2000, 0), (2100, 0)]):
        g.add_node(n, x=float(x), y=float(y))
    for u, v in [(0, 1), (2, 3), (4, 5)]:
        g.add_edge(u, v, length=100.0, geometry=LineString([(g.nodes[u]["x"], 0), (g.nodes[v]["x"], 0)]))
    added = bridge_components(g, max_gap_m=100.0)
    assert added == 1 and nx.number_connected_components(g) == 2   # the far part (1.7 km away) stays apart
    (u, v, d), = [(u, v, d) for u, v, d in g.edges(data=True) if d.get("bridged")]
    assert {u, v} == {1, 2} and d["length"] == pytest.approx(60.0)


def test_faces_become_one_connected_crossroad_with_centre_lines():
    g = build_graph(faces_of_a_crossroad(), CorridorParams(half_width_m=8.0, resolution_m=2.0, min_spur_m=10.0))
    assert nx.is_connected(g)
    degrees = sorted(d for _, d in g.degree())
    assert degrees[-1] == 4, degrees            # one junction where the streets cross
    total = sum(d["length"] for *_, d in g.edges(data=True))
    assert 650 < total < 820                    # ~2 x 400 m of centre line, not 4 x 400 m of faces
    assert {d["name"] for *_, d in g.edges(data=True)} == {"Rua A"}


def test_carriageways_meeting_far_away_are_joined_but_never_through_a_block():
    """Two parallel carriageways 40 m apart that only meet 2 km away: joined at both ends, but not in the middle,
    where a block face lies between them."""
    g = nx.MultiGraph()
    for n, (x, y) in enumerate([(0, 0), (500, 0), (1000, 0), (0, 40), (500, 40), (1000, 40)]):
        g.add_node(n, x=float(x), y=float(y))
    for u, v in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        g.add_edge(u, v, length=500.0, geometry=LineString([(g.nodes[u]["x"], g.nodes[u]["y"]),
                                                           (g.nodes[v]["x"], g.nodes[v]["y"])]))
    g.add_edge(2, 5, length=3000.0, geometry=LineString([(1000, 0), (2500, 0), (2500, 40), (1000, 40)]))
    block_face = LineString([(400, 20), (600, 20)])
    added = bridge_detours(g, [block_face], max_gap_m=70.0, max_detour=10.0)
    links = {frozenset((u, v)) for u, v, d in g.edges(data=True) if d.get("bridged")}
    assert added == 2 and links == {frozenset((0, 3)), frozenset((2, 5))}


def test_sparse_and_image_skeletons_give_the_same_graph():
    sk = np.zeros((30, 40), bool)
    sk[5, 2:38] = True
    sk[5:25, 20] = True
    sk[24, 10:30] = True
    a, b = skeleton_to_graph(sk), sparse_skeleton_to_graph(np.flatnonzero(sk), sk.shape[1])
    assert np.allclose(a.nodes, b.nodes) and len(a.edges) == len(b.edges) == 5


def town_of_blocks(n: int = 9, block: float = 90.0, street: float = 12.0) -> gpd.GeoDataFrame:
    """An n x n grid of square blocks; every block side is a face, as IBGE draws them."""
    x0, y0 = 500_000.0, 7_400_000.0
    lines = []
    for i in range(n):
        for j in range(n):
            ax, ay = x0 + i * (block + street), y0 + j * (block + street)
            corners = [(ax, ay), (ax + block, ay), (ax + block, ay + block), (ax, ay + block), (ax, ay)]
            lines += [LineString([corners[k], corners[k + 1]]) for k in range(4)]
    return gpd.GeoDataFrame({"NM_TIP_LOG": ["RUA"] * len(lines), "NM_TIT_LOG": [""] * len(lines),
                             "NM_LOG": ["B"] * len(lines)}, geometry=lines, crs=UTM_22S)


def test_tiles_with_a_margin_draw_the_same_skeleton_as_one_image():
    faces = town_of_blocks()
    whole = skeleton(faces, CorridorParams(tile_m=1e9))
    tiled = skeleton(faces, CorridorParams(tile_m=150.0, tile_margin_m=60.0))   # 75 x 75 px tiles, 30 px overlap
    assert len(whole.keys) > 1000
    assert np.array_equal(whole.keys, tiled.keys)


def test_each_city_is_projected_in_its_own_utm_zone():
    def at(lon: float, lat: float) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry=[LineString([(lon, lat), (lon + 0.01, lat)])], crs="EPSG:4326")
    assert utm_crs(at(-51.16, -23.31)) == "EPSG:31982"   # Londrina, 22S
    assert utm_crs(at(-46.63, -23.55)) == "EPSG:31983"   # Sao Paulo, 23S
    assert utm_crs(at(-47.88, -15.79)) == "EPSG:31983"   # Brasilia, 23S


def test_parts_with_enough_streets_are_all_kept_and_fragments_dropped():
    g = nx.MultiGraph()
    for u, v, km in [(0, 1, 30.0), (2, 3, 25.0), (4, 5, 0.5)]:   # two districts and a fragment, not connected
        g.add_edge(u, v, length=km * 1000)
    assert major_components(g, min_km=20.0) == {0, 1, 2, 3}
    assert major_components(g, min_km=100.0) == {0, 1}          # none is big enough: the largest stays