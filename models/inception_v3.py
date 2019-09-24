import tensorflow as tf
from models.inception_modules import InceptionModule_1, InceptionModule_2, \
    InceptionModule_3, InceptionModule_4, InceptionModule_5, InceptionAux, Preprocess
from collections import namedtuple

_InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])


# class Preprocess(tf.keras.layers.Layer):
#     x = tf.keras.layers.Conv2D(filters=32,
#                                kernel_size=(3, 3),
#                                strides=2,
#                                padding="valid")(inputs)
#     x = tf.keras.layers.BatchNormalization()(x, training=training)
#     x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
#
#     x = tf.keras.layers.Conv2D(filters=32,
#                                kernel_size=(3, 3),
#                                strides=1,
#                                padding="valid")(x)
#     x = tf.keras.layers.BatchNormalization()(x, training=training)
#     x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
#
#     x = tf.keras.layers.Conv2D(filters=64,
#                                kernel_size=(3, 3),
#                                strides=1,
#                                padding="valid")(x)
#     x = tf.keras.layers.BatchNormalization()(x, training=training)
#     x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
#
#     x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
#                                   strides=2,
#                                   padding="valid")(x)
#     x = tf.keras.layers.Conv2D(filters=80,
#                                kernel_size=(1, 1),
#                                strides=1,
#                                padding="valid")(x)
#     x = tf.keras.layers.BatchNormalization()(x, training=training)
#     x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
#
#     x = tf.keras.layers.Conv2D(filters=192,
#                                kernel_size=(3, 3),
#                                strides=1,
#                                padding="valid")(x)
#     x = tf.keras.layers.BatchNormalization()(x, training=training)
#     x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
#
#     x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
#                                   strides=2,
#                                   padding="valid")(x)
#
#     return x

class InceptionV3(tf.keras.Model):
    def __init__(self, num_class, aux_logits=True):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.preprocess = Preprocess()
        # self.preprocess = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(filters=32,
        #                            kernel_size=(3, 3),
        #                            strides=2,
        #                            padding="valid"),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation(tf.keras.activations.relu),
        #     tf.keras.layers.Conv2D(filters=32,
        #                            kernel_size=(3, 3),
        #                            strides=1,
        #                            padding="valid"),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation(tf.keras.activations.relu),
        #     tf.keras.layers.Conv2D(filters=64,
        #                            kernel_size=(3, 3),
        #                            strides=1,
        #                            padding="same"),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation(tf.keras.activations.relu),
        #     tf.keras.layers.MaxPool2D(pool_size=(3, 3),
        #                               strides=2,
        #                               padding="valid"),
        #     tf.keras.layers.Conv2D(filters=80,
        #                            kernel_size=(1, 1),
        #                            strides=1,
        #                            padding="valid"),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation(tf.keras.activations.relu),
        #     tf.keras.layers.Conv2D(filters=192,
        #                            kernel_size=(3, 3),
        #                            strides=1,
        #                            padding="valid"),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation(tf.keras.activations.relu),
        #     tf.keras.layers.MaxPool2D(pool_size=(3, 3),
        #                               strides=2,
        #                               padding="valid")
        # ])

        self.block_1 = tf.keras.Sequential([
            InceptionModule_1(filter_num=32),
            InceptionModule_1(filter_num=64),
            InceptionModule_1(filter_num=64)
        ])

        self.block_2 = tf.keras.Sequential([
            InceptionModule_2(),
            InceptionModule_3(filter_num=128),
            InceptionModule_3(filter_num=160),
            InceptionModule_3(filter_num=160),
            InceptionModule_3(filter_num=192),
        ])

        if self.aux_logits:
            self.AuxLogits = InceptionAux(num_classes=num_class)

        self.block_3 = tf.keras.Sequential([
            InceptionModule_4(),
            InceptionModule_5(),
            InceptionModule_5()
        ])
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=(8, 8),
                                                  strides=1,
                                                  padding="valid")
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=num_class, activation=tf.keras.activations.linear)

    def call(self, inputs, training=None, mask=None, include_aux_logits=True):
        x = self.preprocess(inputs, training=training)
        x = self.block_1(x, training=training)
        x = self.block_2(x, training=training)
        if include_aux_logits and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.block_3(x, training=training)
        x = self.avg_pool(x)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        if include_aux_logits and self.aux_logits:
            return _InceptionOutputs(x, aux)
        return x
