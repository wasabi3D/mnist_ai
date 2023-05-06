import numpy
import pickle
from collections import OrderedDict


class PredictOnlyNet:
    def __init__(self, layers: OrderedDict):
        self.layers = layers

    def predict(self, x: numpy.ndarray):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def build(network):
    layers = OrderedDict()

    for key, layer in network.layers.items():
        tmp = layer
        tmp.to_numpy()
        layers[key] = tmp

    return PredictOnlyNet(layers)
