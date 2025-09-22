from collections import Counter
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from kd_tree import KDTree
import pandas as pd


class KNNClassifier:
    def __init__(self, k: int, leaf_size: int) -> None:
        self.k: int = k
        self.leaf_size: int = leaf_size
        self.kd_tree: Optional[KDTree] = None
        self.classes: Optional[NDArray] = None
        self.X_train: Optional[NDArray] = None
        self.y_train: Optional[NDArray] = None

    def fit(self, X_train: NDArray[np.float64], y_train: NDArray) -> None:
        if len(X_train) != len(y_train):
            raise ValueError(
                "Number of training samples does not match number of labels"
            )

        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.array(y_train)
        self.classes = np.unique(y_train)
        self.kd_tree = KDTree(X_train, self.leaf_size)

    def predict_proba(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.kd_tree is None or self.classes is None or self.y_train is None:
            raise ValueError("Model is not fitted yet")

        result = []
        indices, _ = self.kd_tree.query(X_test, self.k)

        for point_indices in indices:
            neighbour_labels = self.y_train[point_indices]
            labels_counter = Counter(neighbour_labels)
            proba_vector = [labels_counter.get(cls, 0) / self.k for cls in self.classes]
            result.append(proba_vector)

        return np.array(result)

    def predict(self, X_test: NDArray[np.float64]) -> NDArray:
        probabilities = self.predict_proba(X_test)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]

    def score(self, X_test: NDArray[np.float64], y_test: NDArray) -> float:
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)