from common.network import MultiLayerNet
from common.optimizers import *
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from dataset.emnist_load import load as load_letters


def data_augmentation(x, t):
    import numpy
    new_x_train = []
    new_t_train = []
    for i, img in enumerate(x):
        pil_img = Image.fromarray(numpy.reshape(img, (28, 28)) * 255)

        # 1: rotate randomly
        rotated = pil_img.rotate(numpy.random.randint(-30, 30))
        # 2: move randomly
        translated = pil_img.rotate(0, translate=(numpy.random.randint(-4, 4), numpy.random.randint(-4, 4)))
        # 3: both
        both = pil_img.rotate(numpy.random.randint(-30, 30), translate=(numpy.random.randint(-4, 4), numpy.random.randint(-4, 4)))
        new_x_train += [img,
                        numpy.array(rotated).flatten() / 255,
                        numpy.array(translated).flatten() / 255,
                        numpy.array(both).flatten() / 255]
        new_t_train += 4 * [t[i]]
    return new_x_train, new_t_train


# Load training and test data
# x: image, t: answer
x_train: np.ndarray
t_train: np.ndarray
x_test: np.ndarray
t_test: np.ndarray
try:
    with open("dataset_letters.pkl", "rb") as f:
        x_train, t_train, x_test, t_test = pickle.load(f)
except FileNotFoundError:
    # (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    (x_train, t_train), (x_test, t_test) = load_letters()

    print(f"Size of x_train: {len(x_train)}")
    x_train, t_train = data_augmentation(x_train, t_train)
    x_test, t_test = data_augmentation(x_test, t_test)
    #
    with open("dataset.pkl", "wb") as f:
        pickle.dump((x_train, t_train, x_test, t_test), f)

print(type(x_train[0][0]))
print(f"Final data size: {len(x_train)}")
network = MultiLayerNet(784, [100, 100, 100], 26, Adam(), train_verbose=False)
network.fit(x_train, t_train, x_test, t_test, epochs=15, network_save_name="network2.0letters.pkl", evaluate_num=100)

