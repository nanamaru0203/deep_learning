from activation_function import step_function, sigmoid, ReLU, softmax
import numpy as np
from matplotlib import pyplot as plt
from neural_network import NN

# activation function
x = np.arange(-5, 5, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
y3 = ReLU(x)
plt.plot(x, y1, label="step_function")
plt.plot(x, y2, linestyle="--", label="sigmoid")
plt.plot(x, y3, linestyle="-", label="ReLU")
plt.legend()
plt.show()
print(softmax(np.array([0.3, 2.9, 4.0])))

# neural network
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B1 = np.array([0.1, 0.2, 0.3])
B2 = np.array([0.1, 0.2])
B3 = np.array([0.1, 0.2])

nn = NN(W1, W2, W3, B1, B2, B3)
x = np.array([1.0, 0.5])
print(nn.forward(x))
