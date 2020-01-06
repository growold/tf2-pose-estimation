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
from tensorflow.keras import models, layers

from network_base import max_pool, upsample, inverted_bottleneck, is_trainable

N_KPOINTS = 14

def out_channel_ratio(d): return int(d * 1.0)

def up_channel_ratio(d): return int(d * 1.0)

class HourglassModelBuilderV2():

    def __init__(self):
        # self.build_model()
        print("new Convolutional Pose Machines")

    def build_model(self, inputs=None, trainable=True):
        if inputs != None:
            # input must be (256, 256, 3)
            inputs = tf.keras.Input(tensor=inputs)
        else:
            # Returns a placeholder tensor
            inputs = tf.keras.Input(shape=(128, 128, 3))

        predictions = self.build_network(inputs, trainable=trainable)

        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

    def build_network(self, inputs, trainable):
        is_trainable(trainable)
        input, centermap = inputs
        # input : (batch_size, 128, 128, 3)
        center_map = layers.AveragePooling2D(pool_size=9, strides=8, padding='same')(centermap)

        ##### stage 1 #####
        x = layers.Conv2D(128, kernel_size=9, strides=1, padding='same', activation='relu')(input)
        x = layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = layers.Conv2D(128, kernel_size=9, strides=1, padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = layers.Conv2D(128, kernel_size=9, strides=1, padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=3, strides=2)(x)

        x = layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(512, kernel_size=5, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(512, kernel_size=5, strides=1, padding='same', activation='relu')(x)

        stage1_map = layers.Conv2D(N_KPOINTS + 1, kernel_size=1, strides=1, padding='same')(x)
        ##### stage 1 end #####

        ##### middle #####
        x = layers.Conv2D(128, kernel_size=9, strides=1, padding='same', activation='relu')(input)
        x = layers.MaxPool2D(kernel_size=3, strides=2)(x)
        x = layers.Conv2D(128, kernel_size=9, strides=1, padding='same', activation='relu')(x)
        x = layers.MaxPool2D(kernel_size=3, strides=2)(x)
        x = layers.Conv2D(128, kernel_size=9, strides=1, padding='same', activation='relu')(x)
        middle_map = layers.MaxPool2D(kernel_size=3, strides=2)(x)
        ##### middle end #####
        
        ##### stage 2 #####
        x = layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')(middle_map)
        x = layers.Concatenate()([x, stage1_map, center_map])
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=1, strides=1, padding='valid', activation='relu')(x)
        stage2_map = layers.Conv2D(N_KPOINTS + 1, kernel_size=1)(x)
        ##### stage 2 end #####

        ##### stage 3 #####
        x = layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')(middle_map)
        x = layers.Concatenate()([x, stage2_map, center_map])
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=1, strides=1, padding='valid', activation='relu')(x)
        stage3_map = layers.Conv2D(N_KPOINTS + 1, kernel_size=1)(x)
        ##### stage 3 end #####

        ##### stage 4 #####
        x = layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')(middle_map)
        x = layers.Concatenate()([x, stage3_map, center_map])
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=1, strides=1, padding='valid', activation='relu')(x)
        stage4_map = layers.Conv2D(N_KPOINTS + 1, kernel_size=1)(x)
        ##### stage 4 end #####

        ##### stage 5 #####
        x = layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')(middle_map)
        x = layers.Concatenate()([x, stage4_map, center_map])
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=1, strides=1, padding='valid', activation='relu')(x)
        stage5_map = layers.Conv2D(N_KPOINTS + 1, kernel_size=1)(x)
        ##### stage 5 end #####

        ##### stage 6 #####
        x = layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')(middle_map)
        x = layers.Concatenate()([x, stage5_map, center_map])
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=11, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, kernel_size=1, strides=1, padding='valid', activation='relu')(x)
        stage6_map = layers.Conv2D(N_KPOINTS + 1, kernel_size=1)(x)
        ##### stage 6 end #####

        return stage1_map, stage2_map, stage3_map, stage4_map, stage5_map, stage6_map