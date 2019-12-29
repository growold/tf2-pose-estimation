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
    # with tf.variable_scope("inverted_bottleneck_%s" % scope):

    stride = 2 if subsample else 1

    tower = layers.Conv2D(filters=up_channel_rate * inputs.shape[-1],
                          kernel_size=[1, 1],
                          kernel_initializer=tf.keras.initializers.glorot_normal(),
                          bias_initializer=tf.keras.initializers.Zeros(),
                          activation=None,
                          padding='same')(inputs)
    tower = layers.BatchNormalization()(tower)
    tower = layers.ReLU()(tower)

    tower = layers.SeparableConv2D(filters=channels // 2,
                                   strides=stride,
                                   depth_multiplier=1,
                                   kernel_size=k_s,
                                   depthwise_initializer=tf.keras.initializers.glorot_normal(),
                                   pointwise_initializer=tf.keras.initializers.glorot_normal(),
                                   bias_initializer=None,
                                   padding='same',
                                   depthwise_regularizer=tf.keras.regularizers.l2(0.00004),
                                   pointwise_regularizer=tf.keras.regularizers.l2(0.00004))(tower)
    tower = layers.BatchNormalization()(tower)

    tower = layers.Conv2D(filters=channels,
                          kernel_size=[1, 1],
                          kernel_initializer=tf.keras.initializers.glorot_normal(),
                          bias_initializer=tf.keras.initializers.Zeros(),
                          activation=None,
                          padding='same')(tower)
    tower = layers.BatchNormalization()(tower)
    output = layers.ReLU()(tower)

    if inputs.shape[-1] == channels:
        output = layers.Add()([inputs, output])

    return output