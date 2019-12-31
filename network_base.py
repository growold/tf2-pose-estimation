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

import tensorflow as tf
from tensorflow.keras import layers, regularizers, activations

_trainable = True

def is_trainable(trainable=True):
    global _trainable
    _trainable = trainable


def max_pool(inputs, k_h, k_w, s_h, s_w, name, padding="same"):
    return layers.MaxPool2D(pool_size=(k_h, k_w),
                            strides=(s_h, s_w),
                            padding=padding)(inputs)


def upsample(inputs, factor, name):
    return layers.UpSampling2D(size=(factor, factor))(inputs)


def inverted_bottleneck(inputs, up_channel_rate, channels, subsample, k_s=3, scope=""):
    if inputs.shape[-1] == channels:
        origin_inputs = inputs
    else:
        origin_inputs = layers.SeparableConv2D(
            channels,
            kernel_size=1,
            activation='relu',
            padding="same"
        )(inputs)

    tower = layers.Conv2D(filters=channels // 2,
                          kernel_size=(1, 1),
                          activation='relu',
                          padding='same')(inputs)
    tower = layers.BatchNormalization()(tower)

    tower = layers.SeparableConv2D(filters=channels // 2,
                                   kernel_size=k_s,
                                   activation='relu',
                                   padding='same')(tower)
    tower = layers.BatchNormalization()(tower)

    tower = layers.Conv2D(filters=channels,
                          kernel_size=(1, 1),
                          activation='relu',
                          padding='same')(tower)
    tower = layers.BatchNormalization()(tower)

    output = layers.Add()([origin_inputs, output])

    return output