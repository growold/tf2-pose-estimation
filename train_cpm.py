# Copyright 2019 Felix (felix.fly.lw@gmail.com)
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
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import sys
from os import getcwd
from datetime import datetime

from config.path_manager import PROJ_HOME

import configparser
import argparse

from config.path_manager import TF_MODULE_DIR
from config.path_manager import EXPORT_DIR
from config.path_manager import COCO_DATALOAD_DIR
from config.path_manager import DATASET_DIR

from config.model_config import ModelConfig
from config.train_config import PreprocessingConfig
from config.train_config import TrainConfig

from data_loader_cpm.data_loader import DataLoader

from models.hourglass_model import HourglassModelBuilder
from models.vgg_19_model import VGG19_Model
from models.hourglass_model_v2 import HourglassModelBuilderV2
from models.conv_pose_machine import ConvolutionalPoseMachines

from callbacks_model import get_check_pointer_callback
from callbacks_model import get_tensorboard_callback
from callbacks_model import get_img_tensorboard_callback

print("tensorflow version   :", tf.__version__)
print("keras version        :", tf.keras.__version__)


def main():
    sys.path.insert(0, TF_MODULE_DIR)
    sys.path.insert(0, EXPORT_DIR)
    sys.path.insert(0, COCO_DATALOAD_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="choose a model to train, default is HourglassModel, available: vgg19/hourglass_v2")

    args = parser.parse_args()

    train_config = TrainConfig()
    model_config = ModelConfig(setuplog_dir=train_config.setuplog_dir)
    preproc_config = PreprocessingConfig(setuplog_dir=train_config.setuplog_dir)

    # ================================================
    # =============== dataset pipeline ===============
    # ================================================

    # dataloader instance gen
    dataloader_train, dataloader_valid = \
        [DataLoader(
            is_training=is_training,
            data_dir=DATASET_DIR,
            transpose_input=False,
            train_config=train_config,
            model_config=model_config,
            preproc_config=preproc_config,
            use_bfloat16=False) for is_training in [True, False]]

    dataset_train = dataloader_train.input_fn()
    dataset_valid = dataloader_valid.input_fn()

    data = dataset_train.repeat()
    # iterator = data.make_one_shot_iterator()
    # inputs, targets = iterator.get_next()
    # print(inputs['input_1'])
    # print(inputs['input_2'])
    # print(targets)
    # data = dataset_train

    # ================================================
    # ============== configure model =================
    # ================================================

    model_builder = ConvolutionalPoseMachines()
    model_builder.build_model()
    # model_builder.build_model(inputs=inputs)

    model = model_builder.model
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, epsilon=1e-8),  # 'adam',
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['accuracy', 'mse'])  # ,
    # metrics=['mse'])
    # target_tensors=[targets])#tf.metrics.Accuracy

    # ================================================
    # =============== setup output ===================
    # ================================================

    current_time = datetime.now().strftime("%m%d%H%M")
    output_path = os.path.join(PROJ_HOME, "outputs")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_model_name = "_hg"  # hourglass
    output_base_model_name = "_{}".format(model_config.base_model_name)
    output_learning_rate = "_lr{}".format(train_config.learning_rate)
    # output_decoder_filters = "_{}".format(model_config.filter_name)

    output_name = current_time + output_model_name + output_learning_rate  # + output_decoder_filters

    model_path = os.path.join(output_path, "models")
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    log_path = os.path.join(output_path, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    print("\n")
    print("model path:", model_path)
    print("log path  :", log_path)
    print("model name:", output_name)
    print("\n")

    images, labels = dataloader_valid.get_images(22, batch_size=6)

    # --------------------------------------------------------------------------------------------------------------------
    # output model file(.hdf5)
    check_pointer_callback = get_check_pointer_callback(model_path=model_path, output_name=output_name)

    # output tensorboard log
    tensorboard_callback = get_tensorboard_callback(log_path=log_path, output_name=output_name)

    # tensorboard image
    img_tensorboard_callback = get_img_tensorboard_callback(log_path=log_path, output_name=output_name, images=images,
                                                            labels=labels, model=model)
    # --------------------------------------------------------------------------------------------------------------------

    # ================================================
    # ==================== train! ====================
    # ================================================

    model.fit_generator(data,  # dataset_train_one_shot_iterator
              epochs=train_config.epochs,
              steps_per_epoch=train_config.steps_per_epoch,
              validation_steps=32,
              validation_data=dataset_valid,
              callbacks=[
                  check_pointer_callback,
                  tensorboard_callback,
                  img_tensorboard_callback])


if __name__ == '__main__':
    main()