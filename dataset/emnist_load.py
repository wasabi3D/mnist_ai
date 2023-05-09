from emnist import *
from PIL import Image
import numpy as np


def _load_test():
    images, labels = extract_test_samples('letters')
    image = Image.fromarray(images[10000])
    labels = labels.copy() - 1
    images = images / 255

    flatten_images = []
    for im in images:
        flatten_images.append(im.flatten())

    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1

    return np.array(flatten_images), one_hot_labels

def _load_train():
    images, labels = extract_training_samples('letters')
    image = Image.fromarray(images[10000])
    labels = labels.copy() - 1
    images = images / 255

    flatten_images = []
    for im in images:
        flatten_images.append(im.flatten())

    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1

    return np.array(flatten_images), one_hot_labels


def load():
    return _load_train(), _load_test()
