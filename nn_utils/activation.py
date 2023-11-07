from typing import Any
import numpy as np

from nn_utils.layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        """
        Non linear component
        """
        self.input = input  # store input for backpropagation
        return self.activation(input)

    def backward(self, output_gradient, learning_rate):
        """
        dE/dx = (dE/dy)(dy/dx) = (dE/dy)*f'(x)
        """
        return np.multiply(output_gradient, self.activation_derivative(self.input))


class ReLU(Activation):
    def __init__(self):
        def relu(x: np.ndarray[Any, np.float64]):
            return np.maximum(0, x)

        def relu_derivative(x: np.ndarray[Any, np.float64]):
            return np.where(x <= 0, 0, 1)

        super().__init__(relu, relu_derivative)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x: np.ndarray[Any, np.float64]):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x: np.ndarray[Any, np.float64]):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_derivative)


class Tanh(Activation):
    def __init__(self):
        def tanh(x: np.ndarray[Any, np.float64]):
            return np.tanh(x)

        def tanh_derivative(x: np.ndarray[Any, np.float64]):
            t = tanh(x)
            return 1 - t**2

        super().__init__(tanh, tanh_derivative)
