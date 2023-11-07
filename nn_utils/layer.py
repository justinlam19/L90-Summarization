from typing import Any
import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: np.ndarray[Any, np.dtype[np.float64]]):
        pass

    def backward(
        self,
        output_gradient: np.ndarray[Any, np.dtype[np.float64]],
        learning_rate: float,
        momentum: float,
    ):
        pass
