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


class Page(object):
    def __init__(self, name, img, grid=None):
        """
        A page of cipher image

        :param name: the name of the page
        :param img: image with 3-dim shape
        :param grid: manual grid
        """
        super(Page, self).__init__()
        self.name = name
        self.img_2dim = Page.color_to_grey(img)
        self.img_3dim = cv2.cvtColor(self.img_2dim, cv2.COLOR_GRAY2BGR)
        self.binary_img = Page.get_binary_image(self.img_2dim)
        self.height, self.width = self.binary_img.shape
        self.grid = grid

    @classmethod
    def color_to_grey(cls, img, threshold=128):
        """
        Flatten 3-dim image to 2-dim in grey scale, then use threshold to
        remove noises. The result of the function is a image with only
        pure white (255) and pure black (0).

        :param img: image with 3-dim
        :param threshold: [0-255] in grey scale
        :return: noise-free black-n-white image
        """
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_img[np.where(new_img < threshold)] = 0
        new_img[np.where(new_img >= threshold)] = 255
        return new_img

    @classmethod
    def get_binary_image(cls, img, threshold=128):
        """
        convert black-n-white image into binary image,
        with black as 1, white as 0.

        :param img: black-n-white image with 2-dim
        :param threshold: the same threshold with color_to_grey
        :return: a binary image
        """
        new_img = np.copy(img)
        new_img[np.where(new_img < threshold)] = 1
        new_img[np.where(new_img >= threshold)] = 0
        return new_img

    def cnt_col_pixels(self, upper, lower):
        return np.sum(self.binary_img[upper:lower, :], axis=0)

    def cnt_row_pixels(self):
        return np.sum(self.binary_img, axis=1)


class Pages(object):
    @classmethod
    def from_image(cls, name, path):
        img = cv2.imread(path)
        return Page(name, img)
