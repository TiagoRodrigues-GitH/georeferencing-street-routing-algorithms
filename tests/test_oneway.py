"""Matching OSM one-way ways to graph edges, and the strong-connectivity repair."""

from __future__ import annotations

import numpy as np

from street_routing.graph.oneway import MatchParams, match_directions, repair
from street_routing.sources.osm import direction, parse

NODES = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]          # a square block, 200 m sides
LINES = [NODES[[u, v]] for u, v in EDGES]


def way(*pts: tuple[float, float]) -> np.ndarray:
    return np.array(pts, dtype=float)


def test_osm_tags_give_the_driving_direction():
    assert direction({"oneway": "yes"}) == 1
    assert direction({"oneway": "-1"}) == -1
    assert direction({"junction": "roundabout"}) == 1
    assert direction({"highway": "motorway", "oneway": "no"}) == 0
    assert direction({"highway": "residential"}) == 0
    ways = parse({"elements": [{"type": "way", "id": 7, "tags": {"highway": "primary", "oneway": "-1"},
                                "geometry": [{"lon": 1, "lat": 2}, {"lon": 3, "lat": 4}]}]})
    assert ways[0].coords == ((3, 4), (1, 2))      # stored in the driving direction


def test_parallel_way_within_tolerance_sets_the_direction():
    along = way((-5, 4), (205, 4))                 # 4 m off edge 0, drawn from node 0 to node 1
    against = way((205, 196), (-5, 196))           # beside edge 2 (2 -> 3), same direction as the edge
    reverse = way((204, 205), (204, -5))           # beside edge 1 (1 -> 2) but running 2 -> 1
    found = match_directions(NODES, EDGES, LINES, [along, against, reverse], set())
    assert found == {0: 1, 2: 1, 1: -1}


def test_far_perpendicular_and_two_carriageway_ways_do_not_match():
    far = way((-5, 40), (205, 40))                 # 40 m away
    across = way((100, -50), (100, 50))            # perpendicular
    up, down = way((-5, 3), (205, 3)), way((205, -3), (-5, -3))   # a divided avenue drawn as one IBGE line
    assert match_directions(NODES, EDGES, LINES, [far, across], set()) == {}
    assert match_directions(NODES, EDGES, LINES, [up, down], set()) == {}
    assert match_directions(NODES, EDGES, LINES, [way((-5, 4), (205, 4))], {0}) == {}   # estimated links skipped
    assert MatchParams().tolerance_m == 12.0


def test_repair_keeps_every_part_strongly_connected():
    # a one-way loop around the block is fine; a one-way spur into a dead end traps its end node
    loop = repair(4, EDGES, {0, 1, 2, 3})
    assert loop == set()
    spur_edges = EDGES + [(1, 4)]
    reverted = repair(5, spur_edges, {0, 1, 2, 3, 4})
    assert 4 in reverted                            # the dead-end spur becomes two-way again
