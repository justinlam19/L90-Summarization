from typing import Any
import numpy as np

from nn_utils.layer import Layer


class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)

    def forward(self, input):
        """
        Y = WX + B
        dimensions = (output, features) * (features, number) + (output, 1) = (output, number)
        """
        self.input = input    # store input for backpropagation
        return np.dot(self.weights, input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        dE/dw = (dE/dy)(dy/dw) = (dE/dy)(X.T)
        dE/db = (dE/dy)(dy/db) = dE/dy
        dE/dx = (dE/dy)(dy/dx) = (dE/dy)(W.T)
        """
        # update weights and bias
        n = self.input.shape[1]
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights = self.weights - learning_rate * weights_gradient / n
        self.bias = self.bias - learning_rate * np.reshape(np.sum(output_gradient, axis=1), (-1, 1)) / n

        # input gradient, which is passed as the output gradient to the previous layer
        return np.dot(self.weights.T, output_gradient) 
