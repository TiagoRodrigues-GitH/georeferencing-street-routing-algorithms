"""Road skeleton (one-pixel-wide mask) -> graph of junctions and street segments.

Used twice in the project: on the corridor mask drawn from IBGE street faces (milestone 1) and on the road mask
predicted from CBERS-4A images (milestone 3), so both graphs are built by the same code and the APLS comparison
measures the images, not two different vectorisers.

Pixels are 8-connected. A pixel with exactly two skeleton neighbours lies inside a segment; any other pixel
(end: one neighbour; junction: three or more) is a node pixel. Touching node pixels form one node (a junction
drawn as a small blob of pixels is one intersection). Every chain of segment pixels between two node pixels
becomes an edge; a closed loop without any node pixel gets one node so that it is not lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_EIGHT = np.ones((3, 3), dtype=bool)
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class PixelGraph:
    """Nodes as (row, col) centroids; edges as (u, v, pixel path from u's pixel to v's pixel)."""

    nodes: np.ndarray                          # (N, 2) float
    edges: list[tuple[int, int, np.ndarray]]   # path (K, 2) int


def neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    sk = skeleton.astype(np.uint8)
    return ndimage.convolve(sk, kernel, mode="constant") * sk


def _node_pixels(sk: np.ndarray) -> np.ndarray:
    """Pixels that are not inside a segment, plus one pixel per closed loop."""
    node_px = sk & (neighbour_count(sk) != 2)
    components, n = ndimage.label(sk, structure=_EIGHT)
    has_node = np.zeros(n + 1, dtype=bool)
    has_node[np.unique(components[node_px])] = True
    for comp in np.flatnonzero(~has_node[1:]) + 1:
        r, c = np.argwhere(components == comp)[0]
        node_px[r, c] = True
    return node_px


def skeleton_to_graph(skeleton: np.ndarray) -> PixelGraph:
    sk = np.asarray(skeleton, dtype=bool)
    node_px = _node_pixels(sk)
    labels, n_nodes = ndimage.label(node_px, structure=_EIGHT)
    centroids = np.array(ndimage.center_of_mass(node_px, labels, range(1, n_nodes + 1)), dtype=float).reshape(-1, 2)
    h, w = sk.shape

    def inside(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and bool(sk[r, c])

    visited = np.zeros_like(sk)  # segment pixels already part of an edge
    edges: list[tuple[int, int, np.ndarray]] = []
    for r0, c0 in np.argwhere(node_px):
        start = labels[r0, c0]
        for dr, dc in _NEIGHBOURS:
            r, c = r0 + dr, c0 + dc
            if not inside(r, c) or node_px[r, c] or visited[r, c]:
                continue
            path, prev, cur = [(r0, c0), (r, c)], (r0, c0), (r, c)
            visited[r, c] = True
            while True:  # follow the segment until a node pixel
                nxt = None
                for er, ec in _NEIGHBOURS:
                    rr, cc = cur[0] + er, cur[1] + ec
                    if (rr, cc) == prev or not inside(rr, cc):
                        continue
                    if node_px[rr, cc]:
                        if labels[rr, cc] == start and len(path) <= 2:
                            continue  # still beside the node we started from
                        nxt = (rr, cc)
                        break
                    if not visited[rr, cc]:
                        nxt = (rr, cc)
                if nxt is None:
                    break  # a dead end inside a segment cannot happen in a skeleton; stop defensively
                path.append(nxt)
                if node_px[nxt]:
                    edges.append((start - 1, labels[nxt] - 1, np.array(path)))
                    break
                visited[nxt] = True
                prev, cur = cur, nxt
    return PixelGraph(centroids, edges)
