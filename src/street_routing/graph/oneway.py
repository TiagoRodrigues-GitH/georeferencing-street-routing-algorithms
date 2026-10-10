"""Driving direction of the IBGE street edges, matched from OpenStreetMap one-way ways.

The IBGE network has no direction. Each edge is sampled every ``step_m`` metres; a sample matches an OSM one-way
segment when one lies within ``tolerance_m`` and is parallel (angle under ``max_angle_deg``), and records whether
the OSM direction runs with or against the edge. An edge becomes one-way when at least ``min_coverage`` of its
samples match and at least ``min_agreement`` of the matches agree on the direction. A divided avenue that IBGE
draws as one centre line and OSM as two opposite carriageways therefore stays two-way.

Wrong or partial matches could leave a node that can be entered but not left (or the reverse). ``repair`` keeps
each connected part of the network strongly connected: one-way edges touching nodes outside the largest strongly
connected component of their part are made two-way again, until every part is strongly connected.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import LineString


@dataclass(frozen=True)
class MatchParams:
    step_m: float = 10.0
    tolerance_m: float = 12.0
    max_angle_deg: float = 30.0
    min_coverage: float = 0.6
    min_agreement: float = 0.8


DEFAULT_PARAMS = MatchParams()


@dataclass
class OnewayResult:
    oneway: list[int]          # edges drivable only from u to v (after swapping)
    swapped: list[int]         # edges whose u and v were swapped to put them in the driving direction
    matched: int               # edges matched before repair
    reverted: int              # one-way edges made two-way again by the repair


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _oriented(coords: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Polyline coordinates starting at the end closer to ``start``."""
    return coords if np.hypot(*(coords[0] - start)) <= np.hypot(*(coords[-1] - start)) else coords[::-1]


def match_directions(nodes_xy: np.ndarray, edges: list[tuple[int, int]], lines_xy: list[np.ndarray],
                     ways_xy: list[np.ndarray], skip: set[int], p: MatchParams = DEFAULT_PARAMS) -> dict[int, int]:
    """{edge: +1 (drive u -> v) or -1 (drive v -> u)} for the edges that match OSM one-way ways."""
    segments, directions = [], []
    for way in ways_xy:
        for a, b in pairwise(way):
            if not np.allclose(a, b):
                segments.append(LineString([a, b]))
                directions.append(_unit(*(b - a)))
    if not segments:
        return {}
    tree = STRtree(segments)
    cos_max = math.cos(math.radians(p.max_angle_deg))
    out: dict[int, int] = {}
    for e, (u, _v) in enumerate(edges):
        if e in skip:
            continue
        line = LineString(_oriented(lines_xy[e], nodes_xy[u]))
        if line.length < 1:
            continue
        near = tree.query(line, predicate="dwithin", distance=p.tolerance_m)
        if len(near) == 0:
            continue
        n = max(2, int(line.length // p.step_m))
        signs = []
        for i in range(n):
            d0 = line.length * (i + 0.5) / n
            pt = line.interpolate(d0)
            ahead = line.interpolate(min(line.length, d0 + 1.0))
            behind = line.interpolate(max(0.0, d0 - 1.0))
            ex, ey = _unit(ahead.x - behind.x, ahead.y - behind.y)
            here = set()
            for k in near:
                cos = ex * directions[k][0] + ey * directions[k][1]
                if abs(cos) >= cos_max and segments[k].distance(pt) <= p.tolerance_m:
                    here.add(1 if cos > 0 else -1)
            if here:
                # both directions beside the same point: two opposite carriageways, a conflict (0)
                signs.append(here.pop() if len(here) == 1 else 0)
        if len(signs) >= p.min_coverage * n:
            sign, count = Counter(signs).most_common(1)[0]
            if sign != 0 and count >= p.min_agreement * len(signs):
                out[e] = sign
    return out


def repair(n_nodes: int, edges: list[tuple[int, int]], oneway: set[int], max_rounds: int = 20) -> set[int]:
    """One-way edges to make two-way again so that every connected part is strongly connected."""
    reverted: set[int] = set()
    for _ in range(max_rounds):
        g = nx.MultiDiGraph()
        g.add_nodes_from(range(n_nodes))
        for e, (u, v) in enumerate(edges):
            g.add_edge(u, v, key=e)
            if e not in oneway or e in reverted:
                g.add_edge(v, u, key=e)
        bad: set[int] = set()
        for part in nx.weakly_connected_components(g):
            sub = g.subgraph(part)
            sccs = list(nx.strongly_connected_components(sub))
            if len(sccs) == 1:
                continue
            main = max(sccs, key=len)
            bad |= set(part) - main
        if not bad:
            return reverted
        fix = {e for e in oneway - reverted if edges[e][0] in bad or edges[e][1] in bad}
        if not fix:  # cannot happen with two-way repairs, but never loop forever
            return reverted
        reverted |= fix
    return reverted


def assign_oneway(payload: dict, ways: list, crs: str, p: MatchParams = DEFAULT_PARAMS) -> OnewayResult:
    """Adds ``oneway`` to a web graph payload (lon/lat), swapping u/v (and the polyline) of edges driven v -> u."""
    to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lonlat = np.asarray(payload["nodes"], dtype=float).reshape(-1, 2)
    nodes_xy = np.column_stack(to_xy.transform(lonlat[:, 0], lonlat[:, 1]))
    edges = [(payload["edges"][4 * i], payload["edges"][4 * i + 1]) for i in range(len(payload["edges"]) // 4)]
    lines_xy = []
    for line in payload["lines"]:
        pts = np.asarray(line, dtype=float).reshape(-1, 2)
        lines_xy.append(np.column_stack(to_xy.transform(pts[:, 0], pts[:, 1])))
    ways_xy = []
    for way in ways:
        pts = np.asarray(way.coords, dtype=float)
        ways_xy.append(np.column_stack(to_xy.transform(pts[:, 0], pts[:, 1])))
    directions = match_directions(nodes_xy, edges, lines_xy, ways_xy, set(payload.get("bridged", [])), p)
    driven = [(v, u) if directions.get(e) == -1 else (u, v) for e, (u, v) in enumerate(edges)]
    reverted = repair(len(nodes_xy), driven, set(directions))
    oneway = sorted(set(directions) - reverted)
    swapped = sorted(e for e in oneway if directions[e] == -1)
    for e in swapped:  # store every one-way edge in its driving direction, polyline from u to v
        payload["edges"][4 * e], payload["edges"][4 * e + 1] = payload["edges"][4 * e + 1], payload["edges"][4 * e]
    for e in oneway:
        u = payload["edges"][4 * e]
        pts = np.asarray(payload["lines"][e]).reshape(-1, 2)
        if np.hypot(*(pts[0] - lonlat[u])) > np.hypot(*(pts[-1] - lonlat[u])):
            payload["lines"][e] = [c for pt in pts[::-1] for c in pt.tolist()]
    payload["oneway"] = oneway
    return OnewayResult(oneway, swapped, len(directions), len(reverted & set(directions)))
