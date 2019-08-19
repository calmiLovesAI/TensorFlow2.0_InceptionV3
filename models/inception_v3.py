import tensorflow as tf
from models.inception_modules import InceptionModule_1

class InceptionV3(tf.keras.Model):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.preprocess = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding="valid"),
            tf.keras.layers.Conv2D(filters=80,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=192,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding="valid")
        ])

        self.block_1 = tf.keras.Sequential([
            InceptionModule_1(filter_num=32),
            InceptionModule_1(filter_num=64),
            InceptionModule_1(filter_num=64)
        ])


    def call(self, inputs, training=None, mask=None):
        pass