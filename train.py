from common.network import MultiLayerNet
from common.optimizers import *
from dataset.mnist import load_mnist


# Load training and test data
# x: image, t: answer
x_train: np.ndarray
t_train: np.ndarray
x_test: np.ndarray
t_test: np.ndarray
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


network = MultiLayerNet(784, [100, 100, 100], 10, Adam())

network.fit(x_train, t_train, x_test, t_test, epochs=15, network_save_name="network2.0.pkl")

