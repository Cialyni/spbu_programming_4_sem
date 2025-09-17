import time
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, cast
from dataclasses import dataclass
import heapq


@dataclass
class LeafNode:
    points: np.ndarray
    indices: np.ndarray
    type: str = "leaf"


@dataclass
class KDNode:
    axis: int
    point: np.ndarray
    point_idx: int
    left: Union["KDNode", LeafNode]
    right: Union["KDNode", LeafNode]
    type: str = "node"


class KDTree:
    def __init__(self, X: np.ndarray, leaf_size: int = 1) -> None:
        self.leaf_size: int = leaf_size
        self.dim: int = X.shape[1] if len(X.shape) > 1 else 1
        self.X = X
        self.root = self._build_tree(X, np.arange(len(X)))

    def _build_tree(
        self, points: np.ndarray, indices: np.ndarray
    ) -> Union[KDNode, LeafNode]:
        if len(points) <= self.leaf_size:
            return LeafNode(points.copy(), indices.copy())

        axis = self._select_axis_with_max_variance(points)
        sort_idx = np.argsort(points[:, axis])
        sorted_points = points[sort_idx]
        sorted_indices = indices[sort_idx]
        median_idx = len(sorted_points) // 2

        return KDNode(
            axis=axis,
            point=sorted_points[median_idx].copy(),
            point_idx=sorted_indices[median_idx],
            left=self._build_tree(
                sorted_points[:median_idx], sorted_indices[:median_idx]
            ),
            right=self._build_tree(
                sorted_points[median_idx + 1 :], sorted_indices[median_idx + 1 :]
            ),
        )

    def _select_axis_with_max_variance(self, points: np.ndarray) -> int:
        if len(points) == 0:
            return 0
        variances = np.var(points, axis=0)
        return int(np.argmax(variances))

    def query(self, X: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        all_indices = []
        all_distances = []

        for query_point in X:
            heap: List[Tuple[float, int]] = [] 
            self._neighbors_search(self.root, query_point, k, heap)
            sorted_results = sorted(
                [(-dist, idx) for dist, idx in heap], key=lambda x: x[0]
            )
            distances = [dist for dist, idx in sorted_results]
            indices = [idx for dist, idx in sorted_results]

            all_indices.append(indices)
            all_distances.append(distances)

        return np.array(all_indices), np.array(all_distances)

    def _neighbors_search(
        self,
        node: Union[KDNode, LeafNode],
        query_point: np.ndarray,
        k: int,
        heap: List[Tuple[float, int]],
    ) -> None:
        if node.type == "leaf":
            for i, point in enumerate(node.points):
                dist = self._euclidean_distance(query_point, point)
                idx = node.indices[i]
                self._add_to_heap(heap, dist, idx, k)
            return

        axis = node.axis
        dist_to_node = KDTree._euclidean_distance(query_point, node.point)
        self._add_to_heap(heap, dist_to_node, node.point_idx, k)

        if query_point[axis] < node.point[axis]:
            near_node = node.left
            far_node = node.right
        else:
            near_node = node.right
            far_node = node.left

        self._neighbors_search(near_node, query_point, k, heap)
        current_max_dist = -heap[0][0] if len(heap) == k else float("inf")
        potential_dist = abs(query_point[axis] - node.point[axis])

        if potential_dist < current_max_dist or len(heap) < k:
            self._neighbors_search(far_node, query_point, k, heap)

    def _add_to_heap(
        self, heap: List[Tuple[float, int]], dist: float, idx: int, k: int
    ) -> None:
        if len(heap) < k:
            heapq.heappush(heap, (-dist, idx))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, idx))

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.sum((a - b) ** 2)))
