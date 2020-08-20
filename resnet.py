from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


class ResNetModel(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 shortcut_connection=True,
                 weight_decay=2e-4,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 batch_norm_center=True,
                 batch_norm_scale=True):
        super(ResNetModel, self).__init__()

        if num_layers not in (20, 32, 44, 56, 110):
            raise ValueError('num_layers must be one of 20, 32, 44, 56 or 110.')

        self._num_layers = num_layers
        self._shortcut_connection = shortcut_connection
        self._weight_decay = weight_decay
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._batch_norm_center = batch_norm_center
        self._batch_norm_scale = batch_norm_scale

        self._num_units = (num_layers - 2) // 6

        self._kernel_regularizer = regularizers.l2(weight_decay)

        self._init_conv = layers.Conv2D(16, 3, 1, 'same', use_bias=False, kernel_regularizer=self._kernel_regularizer)

        self._block1 = models.Sequential([ResNetUnit(
            16,
            1,
            shortcut_connection,
            True if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)])

        self._block2 = models.Sequential([ResNetUnit(
            32,
            2 if i == 0 else 1,
            shortcut_connection,
            False if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)])
        self._block3 = models.Sequential([ResNetUnit(
            64,
            2 if i == 0 else 1,
            shortcut_connection,
            False if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)])

        self._final_bn = layers.BatchNormalization(
            -1,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale)
        self._final_conv = layers.Conv2D(
            10,
            1,
            1,
            'same',
            use_bias=True,
            kernel_regularizer=self._kernel_regularizer)

        self.softmax = keras.layers.Activation('softmax')


    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)

    def call(self, inputs, training=False):
        net = inputs
        net = self._init_conv(net)

        net = self._block1(net)
        net = self._block2(net)
        net = self._block3(net)

        net = self._final_bn(net)
        net = tf.nn.relu(net)
        net = tf.reduce_mean(net, [1, 2], keepdims=True)
        net = self._final_conv(net)
        net = tf.squeeze(net, axis=[1, 2])
        net = self.softmax(net)

        return net


class ResNetUnit(layers.Layer):
    def __init__(self,
                 depth,
                 stride,
                 shortcut_connection,
                 shortcut_from_preact,
                 weight_decay,
                 batch_norm_momentum,
                 batch_norm_epsilon,
                 batch_norm_center,
                 batch_norm_scale,
                 name):
        super(ResNetUnit, self).__init__(name=name)
        self._depth = depth
        self._stride = stride
        self._shortcut_connection = shortcut_connection
        self._shortcut_from_preact = shortcut_from_preact
        self._weight_decay = weight_decay

        self._kernel_regularizer = regularizers.l2(weight_decay)

        self._bn1 = layers.BatchNormalization(-1,
                                              batch_norm_momentum,
                                              batch_norm_epsilon,
                                              batch_norm_center,
                                              batch_norm_scale,
                                              name='batchnorm_1')
        self._conv1 = layers.Conv2D(depth,
                                    3,
                                    stride,
                                    'same',
                                    use_bias=False,
                                    kernel_regularizer=self._kernel_regularizer,
                                    name='conv1')
        self._bn2 = layers.BatchNormalization(-1,
                                              batch_norm_momentum,
                                              batch_norm_epsilon,
                                              batch_norm_center,
                                              batch_norm_scale,
                                              name='batchnorm_2')
        self._conv2 = layers.Conv2D(depth,
                                    3,
                                    1,
                                    'same',
                                    use_bias=False,
                                    kernel_regularizer=self._kernel_regularizer,
                                    name='conv2')

    def call(self, inputs):
        depth_in = inputs.shape[3]
        depth = self._depth
        preact = tf.nn.relu(self._bn1(inputs))

        shortcut = preact if self._shortcut_from_preact else inputs

        if depth != depth_in:
            shortcut = tf.nn.avg_pool2d(
                shortcut, (2, 2), strides=(1, 2, 2, 1), padding='SAME')
            shortcut = tf.pad(
                shortcut, [[0, 0], [0, 0], [0, 0], [(depth - depth_in) // 2] * 2])

        residual = tf.nn.relu(self._bn2(self._conv1(preact)))
        residual = self._conv2(residual)

        output = residual + shortcut if self._shortcut_connection else residual

        return output


if __name__ == '__main__':
    print(tf.__version__)
    print(keras.__version__)

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

    training_epochs = 250
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    lr_decay = 1e-4
    lr_drop = 20

    tf.random.set_seed(777)


    def lr_scheduler(epoch):
        return learning_rate * (0.1 ** (epoch // lr_drop))


    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    train_images, train_labels, test_images, test_labels = load_images()

    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_images)

    # choose model with parameter 20, 32, 44, 56, 110
    model = ResNetModel(20)
    # model = ResNetModel(32)
    # model = ResNetModel(44)
    # model = ResNetModel(56)
    # model = ResNetModel(110)

    model.build_graph(train_images.shape)
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                        decay=lr_decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit_generator(datagen.flow(train_images, train_labels,
                                     batch_size=batch_size), epochs=training_epochs, verbose=2, callbacks=[reduce_lr],
                        steps_per_epoch=train_images.shape[0] // batch_size, validation_data=(test_images, test_labels))

    model.save_weights('cifar10-resnet.h5')
