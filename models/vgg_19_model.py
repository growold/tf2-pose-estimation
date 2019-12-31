# Copyright 2019 Felix Liu (felix.fly.lw@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================
# -*- coding: utf-8 -*-

# ref: https://github.com/edvardHua/PoseEstimationForMobile/blob/master/training/src/network_mv2_hourglass.py

import tensorflow as tf
from tensorflow.keras import models, layers

from network_base import max_pool, upsample, inverted_bottleneck, is_trainable

N_KPOINTS = 14
STAGE_NUM = 4


def out_channel_ratio(d): return int(d * 1.0)


def up_channel_ratio(d): return int(d * 1.0)


class VGG19_Model():

    def __init__(self):
        # self.build_model()
        print("new HourglassModelBuilder")

    def build_model(self, inputs=None, trainable=True):
        if inputs != None:
            # input must be (256, 256, 3)
            inputs = tf.keras.Input(tensor=inputs)
        else:
            # Returns a placeholder tensor
            inputs = tf.keras.Input(shape=(128, 128, 3))

        predictions = self.build_network(
            inputs, trainable=trainable)

        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

    def build_network(self, input, trainable):
        is_trainable(trainable)

        # input : (batch_size, 128, 128, 3)

        output = layers.Conv2D(
            64,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(input)  # (batch_size, 128, 128, 64)
        output = layers.Conv2D(
            64,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 128, 128, 64)
        output = layers.MaxPool2D(
            pool_size=2,
            strides=2
        )(output)  # (batch_size, 64, 64, 64)

        #
        output = layers.Conv2D(
            128,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 64, 64, 128)
        output = layers.Conv2D(
            128,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 64, 64, 128)
        output = layers.MaxPool2D(
            pool_size=2,
            strides=2
        )(output)  # (batch_size, 32, 32, 128)

        #
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 32, 32, 256)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 32, 32, 256)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 32, 32, 256)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 32, 32, 256)
        output = layers.MaxPool2D(
            pool_size=2,
            strides=2
        )(output)  # (batch_size, 16, 16, 256)

        #
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 16, 16, 512)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 16, 16, 512)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 16, 16, 512)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 16, 16, 512)
        output = layers.MaxPool2D(
            pool_size=2,
            strides=2
        )(output)  # (batch_size, 8, 8, 512)

        #
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 8, 8, 512)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 8, 8, 512)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 8, 8, 512)
        output = layers.Conv2D(
            256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(output)  # (batch_size, 8, 8, 512)
        output = layers.MaxPool2D(
            pool_size=2,
            strides=2
        )(output)  # (batch_size, 4, 4, 512)

        output = layers.Flatten()(output)
        output = layers.Dense(4096)(output)
        output = layers.Dense(4096)(output)
        output = layers.Dense(14336)(output)
        print('shape before reshape:', output.shape)

        output = layers.Reshape((32, 32, 14))(output)

        return output
