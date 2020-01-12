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

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter


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


def render_joints(cvmat, joints, conf_th=0.2):
    for _joint in joints:
        _x, _y, _conf = _joint
        if _conf > conf_th:
            cv2.circle(cvmat, center=(int(_x), int(_y)),
                       color=(255, 0, 0), radius=7, thickness=2)

    return cvmat


class PoseImageProcessor:
    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    @staticmethod
    def display_image(inp, true_heat=None, pred_heat=None, as_numpy=False):
        global mplset
        mplset = True
        import matplotlib.pyplot as plt
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        import matplotlib
        matplotlib.use('Agg')

        fig = plt.figure()
        if true_heat is not None:
            a = fig.add_subplot(2, 2, 1)
            a.set_title('True Heatmap')
            plt.imshow(PoseImageProcessor.get_bgimg(inp, target_size=(true_heat.shape[1], true_heat.shape[0])),
                       alpha=0.5)
            tmp = np.amax(true_heat, axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.7)
            plt.colorbar()
        else:
            a = fig.add_subplot(2, 2, 1)
            a.set_title('Image')
            plt.imshow(PoseImageProcessor.get_bgimg(inp))

        if pred_heat is not None:
            a = fig.add_subplot(2, 2, 2)
            a.set_title('Pred Heatmap')
            plt.imshow(PoseImageProcessor.get_bgimg(inp, target_size=(pred_heat.shape[1], pred_heat.shape[0])),
                       alpha=0.5)
            tmp = np.amax(pred_heat, axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=1, vmin=0.0, vmax=1.0)
            # plt.imshow(tmp, cmap=plt.cm.gray, alpha=1)
            plt.colorbar()

        if true_heat is not None and pred_heat is not None:
            kps = post_process_heatmap(pred_heat)
            mkps = list()
            scale = inp.shape[0] // true_heat.shape[0]
            for i, _kp in enumerate(kps):
                _conf = _kp[2]
                mkps.append((_kp[0] * scale, _kp[1] * scale, _conf))

            cvmat = PoseImageProcessor.get_bgimg(inp)
            cvmat = render_joints(cvmat, mkps, conf_th=0.1)
            a = fig.add_subplot(2, 2, 3)
            a.set_title('Pred KeyPoints')
            plt.imshow(cvmat, cmap=plt.cm.gray, alpha=1, vmin=0.0, vmax=1.0)

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(),
                                 dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            fig.clear()
            plt.close()
            return data
