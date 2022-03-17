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
batch_size = 100
correct = 0
for i in range(0, len(x_test), batch_size):
    y = nn.forward(x_test[i : i + batch_size])
    p = np.argmax(y, axis=1)
    check = p == t_test[i : i + batch_size]
    correct += np.sum(check.astype(np.int))

print("The accuracy is", correct / len(x_test))
