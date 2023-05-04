import numpy as np


def softmax(x: np.ndarray):
    max_ = np.max(x, axis=-1, keepdims=True)  # ????
    sum_ = np.sum(np.exp(x - max_), axis=-1, keepdims=True)

    out = np.exp(x - max_) / sum_

    return out


# t: one-hot vector!!!
def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:  # if y is a single data (not batch)
        y.reshape(1, y.size)
        t.reshape(1, t.size)

    eps = 1e-7  # prevent -inf
    return -np.sum(t * np.log(y + eps)) / y.shape[0]


class ReLU:
    def __init__(self):
        self.neg_mask = None

    def forward(self, x: np.ndarray):
        self.neg_mask = (x <= 0)
        output = x.copy()
        output[self.neg_mask] = 0
        return output

    def backward(self, dout: np.ndarray):
        dout[self.neg_mask] = 0
        return dout


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x: np.ndarray):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout: np.ndarray):
        return dout * (1.0 - self.y) * self.y


class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W: np.ndarray = W
        self.b: np.ndarray = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout: np.ndarray):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: np.ndarray, t: np.ndarray):  # t-> one-hot-vector
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)

        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
