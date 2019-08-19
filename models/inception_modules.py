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


class InceptionModule_2(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionModule_2, self).__init__()
        # branch 0
        self.conv_b0_1 = tf.keras.layers.Conv2D(filters=384,
                                                kernel_size=(3, 3),
                                                strides=2,
                                                padding="valid")
        self.bn_b0_2 = tf.keras.layers.BatchNormalization()
        self.relu_b0_3 = tf.keras.layers.Activation(tf.keras.activations.relu)

        # branch 1
        self.conv_b1_1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b1_2 = tf.keras.layers.BatchNormalization()
        self.relu_b1_3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b1_4 = tf.keras.layers.Conv2D(filters=96,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.bn_b1_5 = tf.keras.layers.BatchNormalization()
        self.relu_b1_6 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b1_7 = tf.keras.layers.Conv2D(filters=96,
                                                kernel_size=(3, 3),
                                                strides=2,
                                                padding="valid")
        self.bn_b1_8 = tf.keras.layers.BatchNormalization()
        self.relu_b1_9 = tf.keras.layers.Activation(tf.keras.activations.relu)

        # branch 2
        self.maxpool_b2_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                      strides=2,
                                                      padding="valid")

    def call(self, inputs, **kwargs):
        b0 = self.conv_b0_1(inputs)
        b0 = self.bn_b0_2(b0)
        b0 = self.relu_b0_3(b0)

        b1 = self.conv_b1_1(inputs)
        b1 = self.bn_b1_2(b1)
        b1 = self.relu_b1_3(b1)
        b1 = self.conv_b1_4(b1)
        b1 = self.bn_b1_5(b1)
        b1 = self.relu_b1_6(b1)
        b1 = self.conv_b1_7(b1)
        b1 = self.bn_b1_8(b1)
        b1 = self.relu_b1_9(b1)

        b2 = self.maxpool_b2_1(inputs)

        output = tf.keras.layers.concatenate([b0, b1, b2], axis=-1)
        return output


class InceptionModule_3(tf.keras.layers.Layer):
    def __init__(self, filter_num):
        super(InceptionModule_3, self).__init__()
        # branch 0
        self.conv_b0_1 = tf.keras.layers.Conv2D(filters=192,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b0_2 = tf.keras.layers.BatchNormalization()
        self.relu_b0_3 = tf.keras.layers.Activation(tf.keras.activations.relu)

        # branch 1
        self.conv_b1_1 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b1_2 = tf.keras.layers.BatchNormalization()
        self.relu_b1_3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b1_4 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(1, 7),
                                                strides=1,
                                                padding="same")
        self.bn_b1_5 = tf.keras.layers.BatchNormalization()
        self.relu_b1_6 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b1_7 = tf.keras.layers.Conv2D(filters=192,
                                                kernel_size=(7, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b1_8 = tf.keras.layers.BatchNormalization()
        self.relu_b1_9 = tf.keras.layers.Activation(tf.keras.activations.relu)

        # branch 2
        self.conv_b2_1 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b2_2 = tf.keras.layers.BatchNormalization()
        self.relu_b2_3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b2_4 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(7, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b2_5 = tf.keras.layers.BatchNormalization()
        self.relu_b2_6 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b2_7 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(1, 7),
                                                strides=1,
                                                padding="same")
        self.bn_b2_8 = tf.keras.layers.BatchNormalization()
        self.relu_b2_9 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b2_10 = tf.keras.layers.Conv2D(filters=filter_num,
                                                 kernel_size=(7, 1),
                                                 strides=1,
                                                 padding="same")
        self.bn_b2_11 = tf.keras.layers.BatchNormalization()
        self.relu_b2_12 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b2_13 = tf.keras.layers.Conv2D(filters=192,
                                                 kernel_size=(1, 7),
                                                 strides=1,
                                                 padding="same")
        self.bn_b2_14 = tf.keras.layers.BatchNormalization()
        self.relu_b2_15 = tf.keras.layers.Activation(tf.keras.activations.relu)

        # branch 3
        self.avgpool_b3_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                      strides=1,
                                                      padding="same")
        self.conv_b3_2 = tf.keras.layers.Conv2D(filters=192,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b3_3 = tf.keras.layers.BatchNormalization()
        self.relu_b3_4 = tf.keras.layers.Activation(tf.keras.activations.relu)

    def call(self, inputs, **kwargs):
        b0 = self.conv_b0_1(inputs)
        b0 = self.bn_b0_2(b0)
        b0 = self.relu_b0_3(b0)

        b1 = self.conv_b1_1(inputs)
        b1 = self.bn_b1_2(b1)
        b1 = self.relu_b1_3(b1)
        b1 = self.conv_b1_4(b1)
        b1 = self.bn_b1_5(b1)
        b1 = self.relu_b1_6(b1)
        b1 = self.conv_b1_7(b1)
        b1 = self.bn_b1_8(b1)
        b1 = self.relu_b1_9(b1)

        b2 = self.conv_b2_1(inputs)
        b2 = self.bn_b2_2(b2)
        b2 = self.relu_b2_3(b2)
        b2 = self.conv_b2_4(b2)
        b2 = self.bn_b2_5(b2)
        b2 = self.relu_b2_6(b2)
        b2 = self.conv_b2_7(b2)
        b2 = self.bn_b2_8(b2)
        b2 = self.relu_b2_9(b2)
        b2 = self.conv_b2_10(b2)
        b2 = self.bn_b2_11(b2)
        b2 = self.relu_b2_12(b2)
        b2 = self.conv_b2_13(b2)
        b2 = self.bn_b2_14(b2)
        b2 = self.relu_b2_15(b2)

        b3 = self.avgpool_b3_1(inputs)
        b3 = self.conv_b3_2(b3)
        b3 = self.bn_b3_3(b3)
        b3 = self.relu_b3_4(b3)

        output = tf.keras.layers.concatenate([b0, b1, b2, b3], axis=-1)
        return output


class InceptionModule_4(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionModule_4, self).__init__()
        # branch 0
        self.conv_b0_1 = tf.keras.layers.Conv2D(filters=192,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b0_2 = tf.keras.layers.BatchNormalization()
        self.relu_b0_3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b0_4 = tf.keras.layers.Conv2D(filters=320,
                                                kernel_size=(3, 3),
                                                strides=2,
                                                padding="valid")
        self.bn_b0_5 = tf.keras.layers.BatchNormalization()
        self.relu_b0_6 = tf.keras.layers.Activation(tf.keras.activations.relu)

        # branch 1
        self.conv_b1_1 = tf.keras.layers.Conv2D(filters=192,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b1_2 = tf.keras.layers.BatchNormalization()
        self.relu_b1_3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b1_4 = tf.keras.layers.Conv2D(filters=192,
                                                kernel_size=(1, 7),
                                                strides=1,
                                                padding="same")
        self.bn_b1_5 = tf.keras.layers.BatchNormalization()
        self.relu_b1_6 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b1_7 = tf.keras.layers.Conv2D(filters=192,
                                                kernel_size=(7, 1),
                                                strides=1,
                                                padding="same")
        self.bn_b1_8 = tf.keras.layers.BatchNormalization()
        self.relu_b1_9 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv_b1_10 = tf.keras.layers.Conv2D(filters=192,
                                                 kernel_size=(3, 3),
                                                 strides=2,
                                                 padding="valid")
        self.bn_b1_11 = tf.keras.layers.BatchNormalization()
        self.relu_b1_12 = tf.keras.layers.Activation(tf.keras.activations.relu)

        # branch 2
        self.maxpool_b2_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                      strides=2,
                                                      padding="valid")

    def call(self, inputs, **kwargs):
        b0 = self.conv_b0_1(inputs)
        b0 = self.bn_b0_2(b0)
        b0 = self.relu_b0_3(b0)
        b0 = self.conv_b0_4(b0)
        b0 = self.bn_b0_5(b0)
        b0 = self.relu_b0_6(b0)

        b1 = self.conv_b1_1(inputs)
        b1 = self.bn_b1_2(b1)
        b1 = self.relu_b1_3(b1)
        b1 = self.conv_b1_4(b1)
        b1 = self.bn_b1_5(b1)
        b1 = self.relu_b1_6(b1)
        b1 = self.conv_b1_7(b1)
        b1 = self.bn_b1_8(b1)
        b1 = self.relu_b1_9(b1)
        b1 = self.conv_b1_10(b1)
        b1 = self.bn_b1_11(b1)
        b1 = self.relu_b1_12(b1)

        b2 = self.maxpool_b2_1(inputs)

        output = tf.keras.layers.concatenate([b0, b1, b2], axis=-1)


class InceptionModule_5(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionModule_5, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=320,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=384,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=448,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=384,
                                            kernel_size=(1, 3),
                                            strides=1,
                                            padding="same")
        self.conv5 = tf.keras.layers.Conv2D(filters=384,
                                            kernel_size=(3, 1),
                                            strides=1,
                                            padding="same")
        self.conv6 = tf.keras.layers.Conv2D(filters=384,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.conv7 = tf.keras.layers.Conv2D(filters=192,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.avgpool = tf.keras.layers.AvgPool2D(pool_size=(3, 3),
                                                 strides=1,
                                                 padding="same")

    def call(self, inputs, **kwargs):
        b0 = self.conv1(inputs)
        b0 = self.bn(b0)
        b0 = self.relu(b0)

        b1 = self.conv2(inputs)
        b1 = self.bn(b1)
        b1 = self.relu(b1)
        b1_part_a = self.conv4(b1)
        b1_part_a = self.bn(b1_part_a)
        b1_part_a = self.relu(b1_part_a)
        b1_part_b = self.conv5(b1)
        b1_part_b = self.bn(b1_part_b)
        b1_part_b = self.relu(b1_part_b)
        b1 = tf.keras.layers.concatenate([b1_part_a, b1_part_b], axis=-1)

        b2 = self.conv3(inputs)
        b2 = self.bn(b2)
        b2 = self.relu(b2)
        b2 = self.conv6(b2)
        b2 = self.bn(b2)
        b2 = self.relu(b2)
        b2_part_a = self.conv4(b2)
        b2_part_a = self.bn(b2_part_a)
        b2_part_a = self.relu(b2_part_a)
        b2_part_b = self.conv5(b2)
        b2_part_b = self.bn(b2_part_b)
        b2_part_b = self.relu(b2_part_b)
        b2 = tf.keras.layers.concatenate([b2_part_a, b2_part_b], axis=-1)

        b3 = self.avgpool(inputs)
        b3 = self.conv7(b3)
        b3 = self.bn(b3)
        b3 = self.relu(b3)

        output = tf.keras.layers.concatenate([b0, b1, b2, b3], axis=-1)
        return output