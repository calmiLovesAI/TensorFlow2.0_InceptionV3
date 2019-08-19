import tensorflow as tf
from models.inception_modules import InceptionModule_1, InceptionModule_2, \
    InceptionModule_3, InceptionModule_4, InceptionModule_5

class InceptionV3(tf.keras.Model):
    def __init__(self, num_class):
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

        self.block_2 = tf.keras.Sequential([
            InceptionModule_2(),
            InceptionModule_3(filter_num=128),
            InceptionModule_3(filter_num=160),
            InceptionModule_3(filter_num=160),
            InceptionModule_3(filter_num=192),
        ])

        self.block_3 = tf.keras.Sequential([
            InceptionModule_4(),
            InceptionModule_5(),
            InceptionModule_5()
        ])
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=(8, 8),
                                                  strides=1,
                                                  padding="valid")
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.logits = tf.keras.layers.Conv2D(filters=num_class,
                                             kernel_size=(1, 1),
                                             strides=1,
                                             padding="same",
                                             activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        prep = self.preprocess(inputs)
        b_1 = self.block_1(prep)
        b_2 = self.block_2(b_1)
        b_3 = self.block_3(b_2)
        avgpool = self.avg_pool(b_3)
        dropout_layer = self.dropout(avgpool)
        output = self.logits(dropout_layer)

        return output