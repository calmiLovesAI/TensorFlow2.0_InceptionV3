import tensorflow as tf


class InceptionModule_1(tf.keras.layers.Layer):
    def __init__(self, filter_num):
        super(InceptionModule_1, self).__init__()
        # branch 0
        self.conv_b0_1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b0_2 = tf.keras.layers.BatchNormalization()
        self.relu_b0_3 = tf.keras.layers.ReLU()

        # branch 1
        self.conv_b1_1 = tf.keras.layers.Conv2D(filters=48,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b1_2 = tf.keras.layers.BatchNormalization()
        self.relu_b1_3 = tf.keras.layers.ReLU()
        self.conv_b1_4 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(5, 5),
                                                strides=1,
                                                padding="same")
        self.bn_b1_5 = tf.keras.layers.BatchNormalization()
        self.relu_b1_6 = tf.keras.layers.ReLU()

        # branch 2
        self.conv_b2_1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b2_2 = tf.keras.layers.BatchNormalization()
        self.relu_b2_3 = tf.keras.layers.ReLU()
        self.conv_b2_4 = tf.keras.layers.Conv2D(filters=96,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.bn_b2_5 = tf.keras.layers.BatchNormalization()
        self.relu_b2_6 = tf.keras.layers.ReLU()
        self.conv_b2_7 = tf.keras.layers.Conv2D(filters=96,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.bn_b2_8 = tf.keras.layers.BatchNormalization()
        self.relu_b2_9 = tf.keras.layers.ReLU()

        # branch 3
        self.avgpool_b3_1 = tf.keras.layers.AvgPool2D(pool_size=(3, 3),
                                                      strides=1,
                                                      padding="same")
        self.conv_b3_2 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")

    def call(self, inputs, **kwargs):

        b0 = self.conv_b0_1(inputs)
        b0 = self.bn_b0_2(b0)
        b0 = self.relu_b0_3(b0)

        b1 = self.conv_b1_1(inputs)
        b1 = self.bn_b1_2(b1)
        b1 = self.relu_b1_3(b1)
        b1 = self.conv_b1_4(b1)

        b2 = self.conv_b2_1(inputs)
        b2 = self.bn_b2_2(b2)
        b2 = self.relu_b2_3(b2)
        b2 = self.conv_b2_4(b2)
        b2 = self.bn_b2_5(b2)
        b2 = self.relu_b2_6(b2)
        b2 = self.conv_b2_7(b2)
        b2 = self.bn_b2_8(b2)
        b2 = self.relu_b2_9(b2)

        b3 = self.avgpool_b3_1(inputs)
        b3 = self.conv_b3_2(b3)

        output = tf.keras.layers.concatenate([b0, b1, b2, b3], axis=-1)
        return output
