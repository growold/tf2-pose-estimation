import os
import tensorflow as tf
import numpy as np
import cv2
import scipy
import imageio
from scipy.ndimage import gaussian_filter, maximum_filter

from PIL import Image

from models.hourglass_model_v2 import HourglassModelBuilderV2
from data_loader.pose_image_processor import PoseImageProcessor

from config.path_manager import PROJ_HOME
kp_keys = [
    'Nose',       # 0
    'Neck',       # 1
    'RShoulder',  # 2
    'RElbow',     # 3
    'RWrist',     # 4
    'LShoulder',  # 5
    'LElbow',     # 6
    'LWrist',     # 7
    'RHip',       # 8
    'RKnee',      # 9
    'RAnkle',     # 10
    'LHip',       # 11
    'LKnee',      # 12
    'LAnkle',     # 13
]

def render_joints(cvmat, joints, conf_th=0.2):
    for _joint in joints:
        _x, _y, _conf = _joint
        if _conf > conf_th:
            cv2.circle(cvmat, center=(int(_x), int(_y)),
                       color=(255, 0, 0), radius=7, thickness=2)

    return cvmat

def normalize(img_data, color_mean):
    img_data = img_data / 255.0
    for i in range(img_data.shape[-1]):
        img_data[:, :, i] -= color_mean[i]

    return img_data


def non_max_suppression(plain, window_size=3, threshold=1e-6):
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))


def post_process_heatmap(heat_map, kpConfidenceTh=0.2):
    kp_list = list()
    for i in range(heat_map.shape[-1]):
        _map = heat_map[:, :, i]
        _map = gaussian_filter(_map, sigma=0.5)
        _nmsPeaks = non_max_suppression(_map, window_size=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            kp_list.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            kp_list.append((0, 0, 0))
    return kp_list


def read_image(img_path):
    img_str = open(img_path, "rb").read()
    if not img_str:
        print("image not read, path=%s" % img_path)
    nparr = np.fromstring(img_str, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


if __name__ == "__main__":

    num_stack = 1
    model_name = "01011241_hg_lr0.0001.hdf5"  # 替换成相应模型的名字
    output_path = os.path.join(PROJ_HOME, "outputs")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    model_path = os.path.join(output_path, "models")
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print('model file name:', model_path + "/" + model_name)
    load_model_path = os.path.join(model_path, model_name)
    model_builder = HourglassModelBuilderV2()
    model_builder.build_model()
    model = model_builder.model
    model.load_weights(load_model_path)
    model.summary()

    input_size = (128, 128)
    image_path = "images/tennis.jpg"  # 如果预测的图片中有多人, 会无法预测
    raw_img_data = read_image(image_path)
    image_shape = raw_img_data.shape
    print('input image shape:', image_shape)
    scale = ((image_shape[0] * 1.0) / input_size[0],
             (image_shape[1] * 1.0) / input_size[1])
    img_data = cv2.resize(raw_img_data, input_size)

    # 如果做了 normalize操作, 预测的结果会很差
    # mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
    # img_data = normalize(img_data, mean)

    input = img_data[np.newaxis, :, :, :]
    out = model.predict(input)
    print('predict out shape:', out.shape)

    kps = post_process_heatmap(out[0, :, :, :])

    mkps = list()
    for i, _kp in enumerate(kps):
        _conf = _kp[2]
        mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _conf))

    cvmat = render_joints(raw_img_data, mkps, conf_th=0.1)

    cv2.imshow('frame', cvmat)
    cv2.waitKey()
