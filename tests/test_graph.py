"""Skeleton -> graph and IBGE faces -> routable graph, on small synthetic inputs."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from street_routing.graph.ibge_graph import UTM_22S, CorridorParams, bridge_components, build_graph
from skimage.morphology import skeletonize

from street_routing.graph.skeleton import skeleton_to_graph


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
