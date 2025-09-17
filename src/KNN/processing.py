import abc
from math import floor
from typing import Optional, Union, Tuple
import numpy as np
from numpy.typing import NDArray


def train_test_split(
    data: np.ndarray, 
    targets: np.ndarray, 
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> tuple:
    if len(data) != len(targets):
        raise ValueError("number of points doesnt equal number of classes")

    if test_size == 1.0:
        return np.array([]), np.array([]), data, targets
    elif test_size == 0.0:
        return data, targets, np.array([]), np.array([])

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(len(data))
        data = data[indices]
        targets = targets[indices]

    ind = floor(len(data) * (1 - test_size))
    
    return data[:ind], targets[:ind], data[ind:], targets[ind:]

class Scaler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, train_data: NDArray) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, transform_data: NDArray) -> NDArray:
        raise NotImplementedError()

    def fit_transform(self, train_data: NDArray) -> NDArray:
        self.fit(train_data)
        return self.transform(train_data)


class MinMaxScaler(Scaler):
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.x_min: Union[NDArray, None] = None
        self.x_max: Union[NDArray, None] = None
        self.feature_range = feature_range

    def fit(self, train_data: NDArray) -> None:
        self.x_min = np.min(train_data, axis=0)
        self.x_max = np.max(train_data, axis=0)

    def transform(self, transform_data: NDArray) -> NDArray:
        transform_data = transform_data.astype(np.float64).copy()
        range_diff = self.x_max - self.x_min
        range_diff[range_diff == 0] = 1
        scaled_data = (transform_data - self.x_min) / range_diff
        min_range, max_range = self.feature_range
        return scaled_data * (max_range - min_range) + min_range


class MaxAbsScaler(Scaler):
    def __init__(self):
        self.x_max_abs: Union[NDArray, None] = None

    def fit(self, train_data: NDArray) -> None:
        self.x_max_abs = np.max(np.abs(train_data), axis=0)

    def transform(self, transform_data: NDArray) -> NDArray:
        transform_data = transform_data.astype(np.float64).copy()
        max_abs_nonzero = self.x_max_abs.copy()
        max_abs_nonzero[max_abs_nonzero == 0] = 1

        return transform_data / max_abs_nonzero


class Metrics:
    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        return np.mean(y_pred == y_true)
    
    @staticmethod
    def f1_score(y_pred: np.ndarray, y_true: np.ndarray):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1