import numpy as np
import activation_function


class NN:
    def __init__(self, W1, W2, W3, B1, B2, B3):
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3

    def forward(self, X, last_activation_function="id"):
        A1 = activation_function.sigmoid(np.dot(X, self.W1) + self.B1)
        A2 = activation_function.sigmoid(np.dot(A1, self.W2) + self.B2)
        if last_activation_function == "softmax":
            return activation_function.softmax(np.dot(A2, self.W3) + self.B3)
        else:
            return activation_function.identity_function(np.dot(A2, self.W3) + self.B3)
