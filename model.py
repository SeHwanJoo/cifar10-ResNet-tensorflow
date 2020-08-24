import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import models


class ResNetModel(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 shortcut_connection=True,
                 weight_decay=1e-4,
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

        self._softmax = layers.Activation('softmax')

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
        net = self._softmax(net)

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

