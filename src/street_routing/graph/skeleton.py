"""Road skeleton (one-pixel-wide mask) -> graph of junctions and street segments.

Used twice in the project: on the corridor mask drawn from IBGE street faces (milestone 1) and on the road mask
predicted from CBERS-4A images (milestone 3), so both graphs are built by the same code and the APLS comparison
measures the images, not two different vectorisers.

Pixels are 8-connected. A pixel with exactly two skeleton neighbours lies inside a segment; any other pixel
(end: one neighbour; junction: three or more) is a node pixel. Touching node pixels form one node (a junction
drawn as a small blob of pixels is one intersection). Every chain of segment pixels between two node pixels
becomes an edge; a closed loop without any node pixel gets one node so that it is not lost.

The skeleton is handled as a sorted set of pixel keys (``row * width + col``), not as an image: a large city at
2 m is a raster of ~10^9 pixels of which ~1% belong to the skeleton, so the tiled pipeline (``ibge_graph``) never
holds the full image. ``skeleton_to_graph`` (an image) and ``sparse_skeleton_to_graph`` (keys) give the same graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class PixelGraph:
    """Nodes as (row, col) centroids; edges as (u, v, pixel path from u's pixel to v's pixel)."""

    nodes: np.ndarray                          # (N, 2) float
    edges: list[tuple[int, int, np.ndarray]]   # path (K, 2) int


def neighbour_table(keys: np.ndarray, width: int) -> np.ndarray:
    """(N, 8) index of each pixel's 8 neighbours in ``keys`` (sorted, unique), -1 where there is none; neighbours
    in the order of ``_NEIGHBOURS``."""
    rows, cols = np.divmod(keys, width)
    table = np.full((len(keys), 8), -1, dtype=np.int64)
    for j, (dr, dc) in enumerate(_NEIGHBOURS):
        ok = (cols + dc >= 0) & (cols + dc < width) & (rows + dr >= 0)
        target = keys + dr * width + dc
        pos = np.searchsorted(keys, target)
        pos[pos >= len(keys)] = 0
        hit = ok & (keys[pos] == target)
        table[hit, j] = pos[hit]
    return table


def _components(table: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Connected-component label of every pixel among ``members`` (a boolean mask); -1 for the others. Labels are
    numbered in order of each component's first pixel, like ``scipy.ndimage.label`` in raster order."""
    n = len(table)
    src = np.repeat(np.arange(n), 8)
    dst = table.ravel()
    keep = (dst >= 0) & members[src] & members[np.maximum(dst, 0)]
    adjacency = sparse.coo_matrix((np.ones(keep.sum(), bool), (src[keep], dst[keep])), shape=(n, n)).tocsr()
    _, raw = connected_components(adjacency, directed=False)
    labels = np.full(n, -1, dtype=np.int64)
    idx = np.flatnonzero(members)
    if len(idx):
        raw_m = raw[idx]
        unique, first = np.unique(raw_m, return_index=True)     # first pixel (lowest key) of each component
        rank = np.empty(len(unique), dtype=np.int64)
        rank[np.argsort(first)] = np.arange(len(unique))
        labels[idx] = rank[np.searchsorted(unique, raw_m)]
    return labels


def sparse_skeleton_to_graph(keys: np.ndarray, width: int) -> PixelGraph:
    """Graph of a skeleton given as the sorted, unique keys ``row * width + col`` of its pixels."""
    keys = np.asarray(keys, dtype=np.int64)
    if len(keys) == 0:
        return PixelGraph(np.zeros((0, 2)), [])
    table = neighbour_table(keys, width)
    count = (table >= 0).sum(axis=1)
    node = count != 2
    everything = _components(table, np.ones(len(keys), bool))
    has_node = np.zeros(everything.max() + 1, bool)
    has_node[everything[node]] = True
    for comp in np.flatnonzero(~has_node):                      # a closed loop: its first pixel becomes a node
        node[np.flatnonzero(everything == comp)[0]] = True
    labels = _components(table, node)
    rows, cols = np.divmod(keys, width)
    n_nodes = labels.max() + 1
    weight = np.bincount(labels[node], minlength=n_nodes)
    centroids = np.column_stack([np.bincount(labels[node], rows[node], n_nodes),
                                 np.bincount(labels[node], cols[node], n_nodes)]) / weight[:, None]

    visited = [False] * len(keys)        # segment pixels already part of an edge
    edges: list[tuple[int, int, np.ndarray]] = []
    tbl = table.tolist()                 # Python lists: the walk below is a tight loop over single pixels
    is_node = node.tolist()
    lab = labels.tolist()
    for p0 in np.flatnonzero(node).tolist():
        start = lab[p0]
        for p in tbl[p0]:
            if p < 0 or is_node[p] or visited[p]:
                continue
            path, prev, cur = [p0, p], p0, p
            visited[p] = True
            while True:  # follow the segment until a node pixel
                nxt = -1
                for q in tbl[cur]:
                    if q < 0 or q == prev:
                        continue
                    if is_node[q]:
                        if lab[q] == start and len(path) <= 2:
                            continue  # still beside the node we started from
                        nxt = q
                        break
                    if not visited[q]:
                        nxt = q
                if nxt < 0:
                    break  # a dead end inside a segment cannot happen in a skeleton; stop defensively
                path.append(nxt)
                if is_node[nxt]:
                    edges.append((start, lab[nxt], np.column_stack([rows[path], cols[path]])))
                    break
                visited[nxt] = True
                prev, cur = cur, nxt
    return PixelGraph(centroids, edges)


def skeleton_to_graph(skeleton: np.ndarray) -> PixelGraph:
    sk = np.asarray(skeleton, dtype=bool)
    return sparse_skeleton_to_graph(np.flatnonzero(sk), sk.shape[1])
