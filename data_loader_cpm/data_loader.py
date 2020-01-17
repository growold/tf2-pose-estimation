# Copyright 2018 Felix Liu (felix.fly.lw@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys
import os

import torch
import torch.utils.data as data

import numpy as np

from os.path import join

from config.path_manager import DATASET_DIR
from pycocotools.coco import COCO

# for coco dataset
from data_loader_cpm import dataset_augment
from data_loader_cpm.dataset_prepare import CocoMetadata

sys.path.insert(0, DATASET_DIR)


class DataLoader(data.Dataset):
    """Generates DataSet input_fn for training or evaluation
        Args:
            is_training: `bool` for whether the input is for training
            data_dir:   `str` for the directory of the training and validation data;
                            if 'null' (the literal string 'null', not None), then construct a null
                            pipeline, consisting of empty images.
            use_bfloat16: If True, use bfloat16 precision; else use float32.
            transpose_input: 'bool' for whether to use the double transpose trick
    """

    def __init__(self, is_training,
                 data_dir,
                 use_bfloat16,
                 train_config,
                 model_config,
                 preproc_config,
                 transpose_input=True):

        self.image_preprocessing_fn = dataset_augment.preprocess_image
        self.is_training = is_training
        self.use_bfloat16 = use_bfloat16
        self.data_dir = data_dir
        self.anno = None
        self.train_config = train_config
        self.model_config = model_config
        self.preproc_config = preproc_config

        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input

        json_filename_split = os.path.split(
            DATASET_DIR)  # DATASET_DIR.split('/')
        if self.is_training:
            json_filename = json_filename_split[-1] + '_train.json'
        else:
            json_filename = json_filename_split[-1] + '_valid.json'

        # tf.logging.info('json loading from %s' % json_filename)
        dataset_path = join(DATASET_DIR, json_filename)
        self.anno = COCO(dataset_path)

        self.imgIds = self.anno.getImgIds()

    def _parse_function(self, img_id, ann=None):
        """
        :param img_id: Tensor
        :return:
        """
        try:
            img_id = img_id.numpy()
        except AttributeError:
            # print(AttributeError)
            var = None

        if ann is not None:
            self.anno = ann

        img_meta = self.anno.loadImgs([img_id])[0]
        anno_ids = self.anno.getAnnIds(imgIds=img_id)
        img_anno = self.anno.loadAnns(anno_ids)
        idx = img_meta['id']

        filename_item_list = img_meta['file_name'].split('/')
        filename = filename_item_list[1] + '/' + filename_item_list[2]

        img_path = join(DATASET_DIR, filename)

        img_meta_data = CocoMetadata(idx=idx,
                                     img_path=img_path,
                                     img_meta=img_meta,
                                     annotations=img_anno,
                                     sigma=self.preproc_config.heatmap_std)

        # print('joint_list = %s' % img_meta_data.joint_list)
        images, heatmap, centermap = self.image_preprocessing_fn(img_meta_data=img_meta_data,
                                                                 preproc_config=self.preproc_config)

        images = torch.from_numpy(images.transpose((2, 0, 1)))
        heatmap = torch.from_numpy(heatmap.transpose((2, 0, 1)))
        centermap = torch.from_numpy(centermap.transpose((2, 0, 1)))
        return images, heatmap, centermap

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img_id = self.imgIds[idx]
        img, heatmap, centermap = self._parse_function(img_id)
        return img, heatmap, centermap

    def get_images(self, idx, batch_size):
        img_list = []
        heatmap_list = []
        centermap_list = []
        for i in range(batch_size):
            img, label, centermap = self._parse_function(self.imgIds[i + idx])
            img_list.append(img)
            heatmap_list.append(label)
            centermap_list.append(centermap)

        return {'input_1': np.array(img_list), 'input_2': np.array(centermap_list)}, np.array(heatmap_list)
