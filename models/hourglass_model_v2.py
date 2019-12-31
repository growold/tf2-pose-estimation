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

# ref: https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/tree/master/src/net
# for better understanding of this model, I removed those function calls

import tensorflow as tf
from tensorflow.keras import models, layers

from network_base import max_pool, upsample, inverted_bottleneck, is_trainable

N_KPOINTS = 14

def out_channel_ratio(d): return int(d * 1.0)

def up_channel_ratio(d): return int(d * 1.0)

def bottleneck(input, output_channels):
    if input.shape[-1] == output_channels:
        _skip = input
    else:
        _skip = layers.SeparableConv2D(
            output_channels,
            kernel_size=1,
            activation='relu',
            padding='same'
        )(input)

    _x = layers.SeparableConv2D(
        output_channels // 2,
        kernel_size=1,
        activation='relu',
        padding='same'
    )(input)
    _x = layers.BatchNormalization()(_x)

    _x = layers.SeparableConv2D(
        output_channels // 2,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(_x)
    _x = layers.BatchNormalization()(_x)

    _x = layers.SeparableConv2D(
        output_channels,
        kernel_size=1,
        activation='relu',
        padding='same'
    )(_x)
    _x = layers.BatchNormalization()(_x)

    output = layers.Add()([_skip, _x])

    return output


class HourglassModelBuilderV2():

    def __init__(self):
        # self.build_model()
        print("new HourglassModelBuilderV2")

    def build_model(self, inputs=None, trainable=True):
        if inputs != None:
            # input must be (256, 256, 3)
            inputs = tf.keras.Input(tensor=inputs)
        else:
            # Returns a placeholder tensor
            inputs = tf.keras.Input(shape=(128, 128, 3))

        predictions = self.build_network(
            inputs, num_channels=128, trainable=trainable)

        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

    def build_network(self, input, num_channels, trainable):
        is_trainable(trainable)

        # input : (batch_size, 128, 128, 3)
        ##### create front module start ##########
        _x = layers.Conv2D(
            64,
            kernel_size=7,
            strides=2,
            padding='same',
            activation='relu',
        )(input)
        _x = layers.BatchNormalization()(_x)

        _x = bottleneck(_x, num_channels // 2)
        _x = layers.MaxPool2D(pool_size=2, strides=2)(_x)

        _x = bottleneck(_x, num_channels // 2)
        _x = bottleneck(_x, num_channels)

        front_x = _x
        ##### create front module end ##########

        output = []
        ##### hourglass module start #####
       
        ##### create left half block start #####
        lf1 = bottleneck(_x, num_channels)
        _x = layers.MaxPool2D(pool_size=2, strides=2)(lf1)

        lf2  = bottleneck(_x, num_channels)
        _x = layers.MaxPool2D(pool_size=2, strides=2)(lf2)

        lf4 = bottleneck(_x, num_channels)
        _x = layers.MaxPool2D(pool_size=2, strides=2)(lf4)

        lf8 = bottleneck(_x, num_channels)
        ##### create left half block end #####

        ##### create right half block start #####
        lf8_connect = bottleneck(lf8, num_channels)
        _x = bottleneck(lf8_connect, num_channels)
        _x = bottleneck(_x, num_channels)
        _x = bottleneck(_x, num_channels)
        rf8 = layers.Add()([_x, lf8_connect])

        _xleft4 = bottleneck(lf4, num_channels)
        _xright4 = layers.UpSampling2D()(rf8)
        add4 = layers.Add()([_xleft4, _xright4])
        rf4 = bottleneck(add4, num_channels)

        _xleft2 = bottleneck(lf2, num_channels)
        _xright2 = layers.UpSampling2D()(rf4)
        add2 = layers.Add()([_xleft2, _xright2])
        rf2 = bottleneck(add2, num_channels)

        _xleft1 = bottleneck(lf1, num_channels)
        _xright1 = layers.UpSampling2D()(rf2)
        add1 = layers.Add()([_xleft1, _xright1])
        rf1 = bottleneck(add1, num_channels)
        ##### create right half block end #####

        ##### create heads start #####
        head = layers.Conv2D(
            num_channels,
            kernel_size=1,
            activation='relu',
            padding='same'
        )(rf1)
        head = layers.BatchNormalization()(head)

        head_parts = layers.Conv2D(
            N_KPOINTS,
            kernel_size=1,
            activation='linear',
            padding='same'
        )(head)

        head = layers.Conv2D(
            num_channels,
            kernel_size=1,
            activation='linear',
            padding='same'
        )(head)

        head_m = layers.Conv2D(
            num_channels,
            kernel_size=1,
            activation='linear',
            padding='same'
        )(head_parts)

        head_next_stage = layers.Add()([head, head_m, front_x])
        ##### create heads end #####
        output.append(head_parts)
        ##### hourglass module end #####

        return output
