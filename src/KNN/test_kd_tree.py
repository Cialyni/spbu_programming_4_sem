import numpy as np
import pytest
from kd_tree import KDTree


class TestKDTree:
    @pytest.mark.parametrize(
        "X_train_size, X_test_size, dim, k",
        [
            (100, 30, 2, 3),
            (200, 40, 3, 5),
            (300, 50, 5, 2),
        ],
    )
    def test_kdtree_implementation(self, X_train_size, X_test_size, dim, k):
        X_train, X_test = TestKDTree._generate_test_data(X_train_size, X_test_size, dim)
        kdtree = KDTree(X_train, leaf_size=5)
        kdtree_indices, kdtree_distances = kdtree.query(X_test, k=k)
        naive_result = TestKDTree._naive_knn(X_train, X_test, k=k)
        comparison = TestKDTree._compare_results(kdtree_indices, naive_result, X_test)

        assert comparison["exact_match"] == comparison["total_points"]

    @staticmethod
    def _naive_knn(
        X_train: np.ndarray, X_test: np.ndarray, k: int = 1
    ) -> np.ndarray:
        results = []
        for test_point in X_test:
            distances = np.linalg.norm(X_train - test_point, axis=1)
            indices = np.argsort(distances)[:k]
            results.append(indices)
        return np.array(results)
    
    @staticmethod
    def _compare_results(
        kdtree_result: np.ndarray,
        brute_force_result: np.ndarray,
        X_test: np.ndarray,
    ) -> dict:
        comparison = {
            "exact_match": 0,
            "total_points": len(X_test),
        }
        for i, (kdtree_indices, brute_indices) in enumerate(
            zip(kdtree_result, brute_force_result)
        ):
            if np.array_equal(kdtree_indices, brute_indices):
                comparison["exact_match"] += 1
        return comparison
    
    @staticmethod
    def _generate_test_data(
        X_train_size: int = 100,
        X_test_size: int = 30,
        dim: int = 3,
        seed: int = 42,
    ) -> tuple:
        np.random.seed(seed)
        X_train = np.random.rand(X_train_size, dim)
        X_test = np.random.rand(X_test_size, dim)
        return X_train, X_test
