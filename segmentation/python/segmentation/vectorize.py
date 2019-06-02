"""
Copyright 2019 Xusen Yin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import numpy as np

from segmentation.utils import eprint


def pad_image(img):
    desired_size = 105

    old_size = img.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img


def compute_snn_features(frames, snn, wb):
    h, l, step = frames.shape
    input_images = []
    for i in range(l):
        input_images.append(
            pad_image(np.copy(frames[:, i:l, :].reshape((h, (l-i)*step)))))

    xs = np.asarray(input_images)
    features = wb * snn.predict(xs.reshape(xs.shape + (1,)))
    return features


def pixels_to_vectors(pixel_row, snn, wb, step=10, window=10):
    h, w = pixel_row.shape
    if w % step != 0:
        padding = np.full((h, step - (w % step)), 255, dtype='uint8')
        pixel_row = np.concatenate([pixel_row, padding], axis=1)
        h, w = pixel_row.shape
    eprint('h, w of pixel row is {}, {}'.format(h, w))
    num_frames = w / step
    frames = pixel_row.reshape((h, -1, step))
    all_features = []
    for i in range(num_frames):
        features = compute_snn_features(
            frames[:, max(0, i-window+1):i+1, :], snn, wb)
        all_features.append(features)
    return all_features
