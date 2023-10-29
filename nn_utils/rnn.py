from typing import Any
import numpy as np

from nn_utils.layer import Layer

class RNN(Layer):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        n = np.sqrt(hidden_dim)
        self.W_hx = np.random.randn(hidden_dim, input_dim) / n
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) / n
        self.W_yh = np.random.randn(output_dim, hidden_dim) / n
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((output_dim, 1))

        self.dW_hx = np.zeros(self.W_hx.shape)
        self.dW_hh = np.zeros(self.W_hh.shape)
        self.dW_yh = np.zeros(self.W_yh.shape)
        self.db_h = np.zeros(self.b_h.shape)
        self.db_y = np.zeros(self.b_y.shape)

    def tanh(self, x: np.ndarray[Any, np.float64], backward: bool=False):
        if backward:
            return 1 - self.tanh(x) ** 2
        return np.tanh(x)
    
    def sigmoid(self, x: np.ndarray[Any, np.float64], backward: bool=False):
        if backward:
            s = self.sigmoid(x)
            return s * (1 - s)
        return 1 / (1 + np.exp(-x))
    
    def cost(self, ys_pred, ys_true, n):
        cost = -np.sum(ys_true * np.log(ys_pred) + (1 - ys_true) * np.log(1 - ys_pred)) / n
        return np.squeeze(cost)

    def forward(self, inputs):
        """
        h_t = tanh(W_hx*X + W_hh*h_(t-1) + b_h)
        y_pred = sigmoid(W_yh*h_t + b_h)
        """
        self.inputs = inputs

        hidden = np.zeros((self.hidden_dim, 1))
        self.hiddens = [hidden]
        self.y_preds = []

        for x in inputs:
            # h_t
            a = self.W_hx @ x
            b = self.W_hh @ hidden
            c = a + b + self.b_h
            hidden = self.tanh(self.W_hx @ x + self.W_hh @ hidden + self.b_h)
            self.hiddens.append(hidden)

            # y
            y_pred = self.W_yh @ hidden + self.b_y
            self.y_preds.append(y_pred)
        
        return self.y_preds

    def backward(self, grads, learning_rate, momentum):
        """
        dE/dW_yh = (dE/dy)(dy/dW_yh) = (dE/dy)(h_t)
        dE/db_y = dE/dy
        dE/dW_hx = (dE/dy)(dy/dh_t)(tanh'(a))(da/dW_hX) = (dE/dy)(W_yh.T)(tanh')(X.T)
        dE/dW_hh = (dE/dy)(dy/dh_t)(tanh'(a))(da/dW_hh) = (dE/dy)(W_yh.T)(tanh')(h_(t-1).T)
        dE/db_h = (dE/dy)(W_yh.T)(tanh')
        """
        dW_yh = np.zeros(self.W_yh.shape)
        db_y = np.zeros(self.b_y.shape)
        dW_hx = np.zeros(self.W_hx.shape)
        dW_hh = np.zeros(self.W_hh.shape)
        db_h = np.zeros(self.b_h.shape)

        for t in reversed(range(len(self.inputs))):
            dy = grads[t]
            hidden = self.hiddens[t + 1]
            dW_yh += dy @ hidden.T
            db_y += dy
            
            dh = self.W_yh.T @ dy
            temp = dh * self.tanh(hidden, backward=True)
            db_h += temp
            dW_hh += temp @ self.hiddens[t].T
            dW_hx += temp @ self.inputs[t].T

        for d in [dW_hx, dW_hh, dW_yh, db_h, db_y]:
            np.clip(d, -1, 1, out=d)

        self.dW_hx = dW_hx + momentum * self.dW_hx
        self.dW_hh = dW_hh + momentum * self.dW_hh
        self.dW_yh = dW_yh + momentum * self.dW_yh
        self.db_h = db_h + momentum * self.db_h
        self.db_y = db_y + momentum * self.db_y
        
        self.W_hx -= learning_rate * self.dW_hx
        self.W_hh -= learning_rate * self.dW_hh
        self.W_yh -= learning_rate * self.dW_yh
        self.b_h -= learning_rate * self.db_h
        self.b_y -= learning_rate * self.db_y 
