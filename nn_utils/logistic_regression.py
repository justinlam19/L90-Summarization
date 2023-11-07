from typing import Any
import numpy as np

from nn_utils.layer import Layer


class LogisticRegression(Layer):
    def __init__(self, input_height: int):
        self.weights = np.random.randn(input_height, 1)
        self.bias = np.random.randn()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, y_pred, y_true, n):
        cost = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n
        return np.squeeze(cost)

    def forward(self, input):
        """
        y = sigmoid(wX+b)
        """
        # We store the input for backpropagation.
        return self.sigmoid(np.dot(self.weights.T, input) + self.bias)

    def backward(self, y_pred, y_true, input, learning_rate, momentum):
        """
        dE/dw = (1/n) * X * (y_pred - y_true).T
        dE/db = (1/n) * sum(y_pred - y_true).T
        """
        # update weights
        n = input.shape[1]
        diff = y_pred - y_true
        self.weights -= learning_rate * np.dot(input, diff.T) / n
        self.bias -= learning_rate * np.sum(diff) / n
