from mnist import load_mnist
import pickle
from neural_network import NN
import numpy as np


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

with open("sample_weight.pkl", "rb") as f:
    network = pickle.load(f)

nn = NN(
    network["W1"],
    network["W2"],
    network["W3"],
    network["b1"],
    network["b2"],
    network["b3"],
)
correct = 0
for x, t in zip(x_test, t_test):
    y = nn.forward(x)
    p = np.argmax(y)
    if t == p:
        correct += 1

print("The accuracy is", correct / len(x_test))
