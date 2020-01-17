# -*- coding: utf-8 -*-
# @Time    : 18-3-7 下午2:36
# @Author  : edvard_hua@live.com
# @FileName: dataset_augument.py
# @Software: PyCharm
# @updated by Jaewook Kang 20181010 for tf-tiny-pose-estimation

import math
import random

import cv2
import numpy as np
# from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid
from enum import Enum

from utils.utils import gaussian_kernel

from data_loader_cpm.model_config import ModelConfig

model_cfg = ModelConfig(setuplog_dir=None)

_network_w = int(model_cfg.input_size)
_network_h = _network_w
_scale = int(model_cfg.input_size / model_cfg.output_size)


class CocoPart(Enum):
    Top = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    Background = 14  # Background is not used


def pose_random_scale(meta):
    scalew = random.uniform(0.8, 1.2)
    scaleh = random.uniform(0.8, 1.2)
    neww = int(meta.width * scalew)
    newh = int(meta.height * scaleh)

    dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            adjust_joint.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = dst
    return meta


def pose_rotation(meta, preproc_config):
    deg = random.uniform(preproc_config.MIN_AUGMENT_ROTATE_ANGLE_DEG,
                         preproc_config.MAX_AUGMENT_ROTATE_ANGLE_DEG)
    img = meta.img

    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)  # x, y
    height, width, _ = img.shape
    # 构造旋转矩阵,一个点(x,y)旋转后坐标如下:
    # (rot_m[0][0] * x + rot_m[0][1] * y + rot_m[0][2], rot_m[1][0] * x + rot_m[1][1] * y + rot_m[1][2])
    rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)

    cos_val = np.abs(rot_m[0, 0])
    sin_val = np.abs(rot_m[0, 1])
    new_width = int(height * sin_val + width * cos_val)
    new_height = int(height * cos_val + width * sin_val)
    rot_m[0, 2] += (new_width * 0.5) - center[1]
    rot_m[1, 2] += (new_height * 0.5) - center[0]

    img = cv2.warpAffine(img, rot_m, (new_width, new_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    # if img.ndim == 3 and ret.ndim == 2:
    #     ret = ret[:, :, np.newaxis]
    # neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
    # neww = min(neww, ret.shape[1])
    # newh = min(newh, ret.shape[0])
    # newx = int(center[0] - neww * 0.5)
    # newy = int(center[1] - newh * 0.5)
    # # print(ret.shape, deg, newx, newy, neww, newh)
    # img = ret[newy:newy + newh, newx:newx + neww]

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            # x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point, deg)
            p = np.array([point[0], point[1], 1])
            p = rot_m.dot(p)
            adjust_joint.append((p[0], p[1]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = new_width, new_height
    meta.img = img

    return meta


def pose_flip(meta):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta

    img = meta.img
    img = cv2.flip(img, 1)

    # flip meta
    flip_list = [CocoPart.Top, CocoPart.Neck, CocoPart.LShoulder, CocoPart.LElbow, CocoPart.LWrist, CocoPart.RShoulder,
                 CocoPart.RElbow, CocoPart.RWrist,
                 CocoPart.LHip, CocoPart.LKnee, CocoPart.LAnkle, CocoPart.RHip, CocoPart.RKnee, CocoPart.RAnkle]

    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for cocopart in flip_list:
            point = joint[cocopart.value]
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((meta.width - point[0], point[1]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list

    meta.img = img
    return meta


def pose_resize_shortestedge_random(meta):
    ratio_w = float(_network_w) / float(meta.width)
    ratio_h = float(_network_h) / float(meta.height)
    ratio = min(ratio_w, ratio_h)

    target_size = int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5))
    target_size = int(target_size * random.uniform(0.95, 1.2))

    # target_size = int(min(_network_w, _network_h) * random.uniform(0.7, 1.5))
    return pose_resize_shortestedge(meta, target_size)


def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    new_x, new_y = newxy

    qx += ox - new_x
    qy += oy - new_y

    return int(qx + 0.5), int(qy + 0.5)


def pose_resize_shortestedge(meta, target_size):
    global _network_w, _network_h
    img = meta.img

    # adjust image
    scale = float(target_size) / float(min(meta.height, meta.width))

    if meta.height < meta.width:
        newh, neww = target_size, int(scale * meta.width + 0.5)
    else:
        newh, neww = int(scale * meta.height + 0.5), target_size

    dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    pw = ph = 0
    if neww < _network_w or newh < _network_h:
        pw = max(0, (_network_w - neww) // 2)
        ph = max(0, (_network_h - newh) // 2)
        mw = (_network_w - neww) % 2
        mh = (_network_h - newh) % 2
        color1 = random.randint(0, 255)
        color2 = random.randint(0, 255)
        color3 = random.randint(0, 255)
        dst = cv2.copyMakeBorder(dst, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=(color1, color2, color3))

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0 or int(point[0]*scale+0.5) > neww or int(point[1]*scale+0.5) > newh:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((int(point[0] * scale + 0.5) + pw, int(point[1] * scale + 0.5) + ph))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww + pw * 2, newh + ph * 2
    meta.img = dst
    return meta


def pose_crop(meta, x, y, w, h):
    # adjust image
    target_size = (w, h)

    img = meta.img
    resized = img[y:y + target_size[1], x:x + target_size[0], :]

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1000, -1000))
            #     continue
            new_x, new_y = point[0] - x, point[1] - y
            # if new_x <= 0 or new_y <= 0 or new_x > target_size[0] or new_y > target_size[1]:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((new_x, new_y))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = target_size
    meta.img = resized
    return meta


def pose_crop_random(meta):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)
    for _ in range(50):
        x = random.randrange(0, meta.width - target_size[0]) if meta.width > target_size[0] else 0
        y = random.randrange(0, meta.height - target_size[1]) if meta.height > target_size[1] else 0

        # check whether any face is inside the box to generate a reasonably-balanced datasets

        # --------------------------------------------------------------------------
        for joint in meta.joint_list:
            if x <= joint[CocoPart.RKnee.value][0] < x + target_size[0] and \
                    y <= joint[CocoPart.RKnee.value][1] < y + target_size[1] and \
                    x <= joint[CocoPart.RAnkle.value][0] < x + target_size[0] and \
                    y <= joint[CocoPart.RAnkle.value][1] < y + target_size[1] and \
                    x <= joint[CocoPart.LKnee.value][0] < x + target_size[0] and \
                    y <= joint[CocoPart.LKnee.value][1] < y + target_size[1] and \
                    x <= joint[CocoPart.LAnkle.value][0] < x + target_size[0] and \
                    y <= joint[CocoPart.LAnkle.value][1] < y + target_size[1]:
                break
        # --------------------------------------------------------------------------

    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_to_img(meta_l):
    global _network_w, _network_h, _scale
    return meta_l.img.astype(np.float32), \
           meta_l.get_heatmap(target_size=(model_cfg.output_size, model_cfg.output_size)).astype(np.float32)


def preprocess_image(img_meta_data, preproc_config):
    if preproc_config.is_scale:
        img_meta_data = pose_random_scale(img_meta_data)

    if preproc_config.is_rotate:
        img_meta_data = pose_rotation(img_meta_data, preproc_config)

    if preproc_config.is_flipping:
        img_meta_data = pose_flip(img_meta_data)

    if preproc_config.is_resize_shortest_edge:
        img_meta_data = pose_resize_shortestedge_random(img_meta_data)

    if preproc_config.is_crop:
        img_meta_data = pose_crop_random(img_meta_data)
    else:
        global _network_w, _network_h
        target_size = (_network_w, _network_h)
        pose_crop(img_meta_data, 0, 0, target_size[0], target_size[1])

    images, labels = pose_to_img(img_meta_data)

    centermap = np.zeros((_network_h, _network_w, 1), dtype=np.float32)
    center_map = gaussian_kernel(size_h=_network_h, size_w=_network_w,
                                 center_x=img_meta_data.center[0], center_y=img_meta_data.center[1],
                                 sigma=img_meta_data.sigma)
    center_map[center_map > 1] = 1
    center_map[center_map < 0.0099] = 0
    centermap[:, :, 0] = center_map

    return images, labels, centermap
