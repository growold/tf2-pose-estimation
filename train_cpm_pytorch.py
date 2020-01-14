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
import sys
import time

from config.path_manager import PROJ_HOME

import argparse

from config.path_manager import TF_MODULE_DIR
from config.path_manager import EXPORT_DIR
from config.path_manager import COCO_DATALOAD_DIR
from config.path_manager import DATASET_DIR

from data_loader_cpm.model_config import ModelConfig
from data_loader_cpm.train_config import PreprocessingConfig
from data_loader_cpm.train_config import TrainConfig

from data_loader_cpm.data_loader import DataLoader

from models_pytorch.conv_pose_machines import ConvPoseMachines

import torch
import torch.nn as nn
from torch.utils import data

import torch.backends.cudnn as cudnn

from utils.utils import AverageMeter, adjust_learning_rate


def get_parameters(model, lr, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': lr},
            {'params': lr_2, 'lr': lr * 2.},
            {'params': lr_4, 'lr': lr * 4.},
            {'params': lr_8, 'lr': lr * 8.}]

    return params, [1., 2., 4., 8.]


def main():
    print('main')
    sys.path.insert(0, TF_MODULE_DIR)
    sys.path.insert(0, EXPORT_DIR)
    sys.path.insert(0, COCO_DATALOAD_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=False, help="retrain a model")
    parser.add_argument("--resume_model", help="retrain model name")
    parser.add_argument("--start_epoch", type=int, help="epoch value for start training")

    args = parser.parse_args()

    train_config = TrainConfig()
    model_config = ModelConfig(setuplog_dir=train_config.setuplog_dir)
    preproc_config = PreprocessingConfig(
        setuplog_dir=train_config.setuplog_dir)

    base_learning_rate = 0.000004
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ================================================
    # =============== setup output ===================
    # ================================================
    output_path = os.path.join(PROJ_HOME, "outputs")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model_path = os.path.join(output_path, "models")
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    log_path = os.path.join(output_path, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    model_name_template = 'model_cpm_epoch_{}.pth'

    print("\n")
    print("model path:", model_path)
    print("log path  :", log_path)
    print("\n")

    # ================================================
    # =============== dataset pipeline ===============
    # ================================================

    # dataloader instance gen
    dataloader_train, dataloader_valid = \
        [data.DataLoader(DataLoader(
            is_training=is_training,
            data_dir=DATASET_DIR,
            transpose_input=False,
            train_config=train_config,
            model_config=model_config,
            preproc_config=preproc_config,
            use_bfloat16=False),
            batch_size=train_config.batch_size, shuffle=True, num_workers=4, pin_memory=True) for is_training in [True, False]]

    criterion = nn.MSELoss().to(device)

    # ================================================
    # ============== configure model =================
    # ================================================
    model = ConvPoseMachines(k=14)
    model = torch.nn.DataParallel(model).to(device)

    if args.start_epoch is not None and args.start_epoch > 0:
        start_epoch = args.start_epoch
        saved_model = os.path.join(model_path, model_name_template.format(start_epoch))
        print('load model from file:' + saved_model)
        saved_dict = torch.load(saved_model)
        model.load_state_dict(saved_dict)
    else:
        start_epoch = 0

    params, multiple = get_parameters(model, base_learning_rate, False)
    optimizer = torch.optim.Adam(params, base_learning_rate, betas=(0.5, 0.999))

    # ================================================
    # ==================== train! ====================
    # ================================================

    cudnn.benchmark = True

    epoch_time = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(6)]
    end = time.time()

    heat_weight = 46 * 46 * 15 / 1.0

    for iters in range(start_epoch, train_config.epochs):
        for i, (inp, heatmap, centermap) in enumerate(dataloader_train):

            learning_rate = adjust_learning_rate(optimizer, iters,
                                                 base_learning_rate, policy='step',
                                                 policy_parameter={'gamma': 0.333, 'step_size': 13275},
                                                 multiple=multiple)

            data_time.update(time.time() - end)

            inp = inp.to(device, non_blocking=True)
            heatmap = heatmap.to(device, non_blocking=True)
            centermap = centermap.to(device, non_blocking=True)

            optimizer.zero_grad()

            heat1, heat2, heat3, heat4, heat5, heat6 = model(inp, centermap)

            loss1 = criterion(heat1, heatmap) * heat_weight
            loss2 = criterion(heat2, heatmap) * heat_weight
            loss3 = criterion(heat3, heatmap) * heat_weight
            loss4 = criterion(heat4, heatmap) * heat_weight
            loss5 = criterion(heat5, heatmap) * heat_weight
            loss6 = criterion(heat6, heatmap) * heat_weight

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            losses.update(loss.item(), inp.size(0))
            for cnt, l in enumerate([loss1, loss2, loss3, loss4, loss5, loss6]):
                losses_list[cnt].update(l.item(), inp.size(0))

            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            epoch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print(f'Train Iteration: {iters} / {train_config.epochs} time: {epoch_time.sum:.3f}s\n'
                      f'Train Step: {i}\n'
                      f'Time {batch_time.sum: .3f}s ({batch_time.avg:.3f})\t'
                      f'Data load {data_time.sum:.3f}s ({data_time.avg:.3f})\n'
                      f'Learning_rate= {learning_rate}\n'
                      f'Loss = {losses.val:.8f} (ave={losses.avg:.8f})\n')
                for cnt in range(6):
                    print(f'Loss{cnt+1} = {losses_list[cnt].val:.8f} (ave = {losses_list[cnt].avg:.8f})\t')

                print(time.strftime('%Y-%m-%d %H:%M:%S---------------\n', time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()

                for cnt in range(6):
                    losses_list[cnt].reset()

        epoch_time.reset()
        if iters % 5 == 0:
            save_model = os.path.join(model_path, model_name_template.format(iters))
            torch.save(model.state_dict(), save_model)


if __name__ == '__main__':
    main()
