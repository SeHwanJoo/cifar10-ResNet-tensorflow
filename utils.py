import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np


def use_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            print('start with GPU 7')
            tf.config.experimental.set_visible_devices(gpus[7], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[7], True)
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)


def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images


def load_images():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    (train_images, test_images) = normalization(train_images, test_images)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


def build_optimizer(learning_rate=0.1, momentum=0.9):
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [500, 32000, 48000],
        [learning_rate / 10., learning_rate, learning_rate / 10., learning_rate / 100.])

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    return optimizer
