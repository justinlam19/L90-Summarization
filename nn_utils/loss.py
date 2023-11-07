from typing import Any
import numpy as np

epsilon = 1e-9


def binary_cross_entropy(
    y_true: np.ndarray[Any, np.float64], y_pred: np.ndarray[Any, np.float64]
):
    return -np.mean(
        np.multiply(y_true, np.log(y_pred + epsilon))
        + np.multiply(1 - y_true, np.log(1 - y_pred + epsilon))
    )


def binary_cross_entropy_derivative(
    y_true: np.ndarray[Any, np.float64], y_pred: np.ndarray[Any, np.float64]
):
    """
    E = -(1/n)SUM(y_true*log(y + e) + (1 - y_true)*log(1 - y + e)))
    dE/dy = -(1/n)(y_true/(y + e) - (1 - y_true)/(1 - y + e)) = ((1 - y_true)/(1 - y + e) - y_true/(y + e)) / n
    """
    return (
        (1 - y_true) / (1 - y_pred + epsilon) - y_true / (y_pred + epsilon)
    ) / np.size(y_true)
