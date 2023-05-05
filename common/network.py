import numpy as np

from common.numpy_import import *
from common.layers import *
from common.optimizers import *
from collections import OrderedDict
from typing import Union
import pickle


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad


class MultiLayerNet:
    def __init__(self,
                 input_size: int,
                 hidden_size_list: list[int],
                 output_size: int,
                 optimizer: Union[Momentum, SGD, Adam, AdaGrad],
                 activation='relu',
                 weight_init_std='relu',
                 weight_decay_lambda=0,
                 train_verbose=True):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params: dict[str, np.ndarray] = {}
        self.optimizer: Union[Momentum, SGD, Adam, AdaGrad] = optimizer
        self.__init_params(weight_init_std)
        self.train_loss_list = []
        self.verbose = train_verbose

        # Generate layers
        self.layers = OrderedDict()
        assert activation.lower() in ('relu', 'sigmoid')
        activation_layer = Sigmoid if activation == 'sigmoid' else ReLU
        for id_ in range(1, self.hidden_layer_num + 1):
            self.layers[f"Affine{id_}"] = Affine(self.params[f"W{id_}"], self.params[f"b{id_}"])
            self.layers[f"Activation{id_}"] = activation_layer()
        last_id = self.hidden_layer_num + 1
        self.layers[f"Affine{last_id}"] = Affine(self.params[f"W{last_id}"], self.params[f"b{last_id}"])
        self.last_layer = SoftmaxWithLoss()

    def __init_params(self, weight_type: str):
        # std or He or Xavier
        weights_num_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for id_ in range(1, len(weights_num_list)):
            scale = 0.01
            last_layer_size = weights_num_list[id_ - 1]
            cur_layer_size = weights_num_list[id_]

            if weight_type.lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / last_layer_size)
            elif weight_type.lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / last_layer_size)

            self.params[f"W{id_}"] = scale * np.random.randn(last_layer_size, cur_layer_size)
            self.params[f"b{id_}"] = np.zeros(cur_layer_size)

    def __train(self, x_batch, t_batch):
        grads = self.gradient(x_batch, t_batch)
        self.optimizer.update(self.params, grads)

        loss = self.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss:" + str(loss))

    def fit(self,
            x_train: np.ndarray,
            t_train: np.ndarray,
            x_test: np.ndarray,
            t_test: np.ndarray,
            epochs=20,
            batch_size=100,
            evaluate_num=None,
            network_save_name=None):
        x_train = np.asarray(x_train)
        t_train = np.asarray(t_train)
        x_test = np.asarray(x_test)
        t_test = np.asarray(t_test)
        train_size = x_train.shape[0]
        iter_per_epoch = int(max(train_size / batch_size, 1))
        total_iter = epochs * iter_per_epoch
        for it in range(total_iter):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            self.__train(x_batch, t_batch)

            # Testing
            if it % iter_per_epoch == 0:
                x_train_sample, t_train_sample = x_train, t_train
                x_test_sample, t_test_sample = x_test, t_test
                if not (evaluate_num is None):
                    t = evaluate_num
                    x_train_sample, t_train_sample = x_train[:t], t_train[:t]
                    x_test_sample, t_test_sample = x_test[:t], t_test[:t]

                train_acc = self.accuracy(x_train_sample, t_train_sample)
                test_acc = self.accuracy(x_test_sample, t_test_sample)

                if self.verbose:
                    print(f"=== epoch: {int(it / iter_per_epoch)}, train acc: {train_acc}, test acc: {test_acc} ===")

        if network_save_name is not None:
            with open(network_save_name, "wb") as f:
                pickle.dump(self, f)

    def predict(self, x: np.ndarray):
        x = np.asarray(x)
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray):
        x = np.asarray(x)
        t = np.asarray(t)
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x: np.ndarray, t: np.ndarray):
        x = np.asarray(x)
        t = np.asarray(t)
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray):
        x = np.asarray(x)
        t = np.asarray(t)
        def loss_W(_):
            return self.loss(x, t)
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray):
        x = np.asarray(x)
        t = np.asarray(t)
        # forward
        self.loss(x, t)

        # backward
        dout = self.last_layer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for id_ in range(1, self.hidden_layer_num + 2):
            grads[f"W{id_}"] = self.layers[f"Affine{id_}"].dW + self.weight_decay_lambda * self.layers[f"Affine{id_}"].W
            grads[f"b{id_}"] = self.layers[f"Affine{id_}"].db

        return grads
