from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import *
import model

if __name__ == '__main__':
    print(tf.__version__)
    print(keras.__version__)

    use_gpu()

    training_epochs = 165
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4,
    batch_norm_momentum = 0.99,
    batch_norm_epsilon = 1e-3,
    batch_norm_center = True,
    batch_norm_scale = True
    tf.random.set_seed(777)

    train_images, train_labels, test_images, test_labels = load_images()

    data_generator = ImageDataGenerator(
        # brightness_range=[.2, .2],
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False  # randomly flip images
    )
    data_generator.fit(train_images)

    # choose model with parameter 20, 32, 44, 56, 110
    model = model.ResNetModel(num_layers=20)
    # model = model.ResNetModel(num_layers=32)
    # model = model.ResNetModel(num_layers=44)
    # model = model.ResNetModel(num_layers=56)
    # model = model.ResNetModel(num_layers=110)


    model.build_graph(train_images.shape)
    model.summary()
    optimizer = build_optimizer(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define the Keras TensorBoard callback.
    logdir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model.fit_generator(data_generator.flow(train_images, train_labels, batch_size=batch_size),
                        epochs=training_epochs,
                        verbose=2,
                        callbacks=[tensorboard_callback],
                        steps_per_epoch=train_images.shape[0] // batch_size,
                        validation_data=(test_images, test_labels))

    model.save_weights('cifar10-resnet.h5')
