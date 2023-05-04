from dataset.mnist import load_mnist
from layers import *
from collections import OrderedDict

# Load training and test data
# x: image, t: answer
x_train: np.ndarray
t_train: np.ndarray
x_test: np.ndarray
t_test: np.ndarray
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

# Show sample image
# sample_img, sample_label = x_train[0].reshape(28, 28), t_train[0]
# print(f"Label of the sample image is {sample_label}")
# pil_sample = Image.fromarray(sample_img * 255)
# pil_sample.show()


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size),
                       'b2': np.zeros(output_size)
                       }

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:input, t:answer
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = self.lastLayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {'W1': self.layers['Affine1'].dW,
                 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW,
                 'b2': self.layers['Affine2'].db
                 }

        return grads
