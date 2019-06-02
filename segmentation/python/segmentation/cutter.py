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

import sys
from collections import namedtuple
from os import path
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from pathos.multiprocessing import ProcessingPool as PP

from segmentation.utils import eprint

CarmelArgs = namedtuple(
    'CarmelArgs',
    'n_row r_std1 r_std2 r_penalty n_col c_std1 c_std2 c_penalty')


class Cutter(object):
    def __init__(self, page, carmel_args, max_height):
        super(Cutter, self).__init__()
        self.page = page
        self.carmel_args = carmel_args
        self.max_height = max_height
        self.col_cuts = None
        self.row_cut = None
        self.pieces = None

    @classmethod
    def _cutting_points(
            cls, black_pixels, mean, std1, std2, p, marker='no_marker'):
        """
        Use Carmel to compute cutting positions.
        align3.sh, make-fsa2.sh, and make-fst2.sh are required.
        :param black_pixels: number of pixels being cut
        :param mean: the mean of total characters in a row,
         or total rows in a page
        :param std1: stddev over number of characters or number of rows
        :param std2: stddev over character size or row size
        :param p: penalty of cutting black pixels
        :param marker: marker to name Carmel running fsts.
        :return: cutting positions
        """
        pwd = path.dirname(path.realpath(__file__))
        exe = '{}/align3.sh'.format(pwd)

        f = NamedTemporaryFile(
            'w', prefix='carmel_input_{}_'.format(marker), delete=False)
        f.write('\n')
        f.write(' '.join(map(lambda c: 'c' + str(c), black_pixels)))
        f.close()

        cmd = "{} {} {} {} {} {}".format(exe, f.name, mean, std1, std2, p)
        pipe = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        out, err = pipe.communicate()
        ret = pipe.wait()
        eprint('marker: {}, returns: {}'.format(marker, out))
        eprint('marker: {}, return code is {}'.format(marker, ret))
        assert out != "", "marker: {}, no output received".format(marker)
        return map(lambda e: int(e), out.split())

    def _cut_rows(self):
        """
        cut along rows of a image
        :return: cutting positions
        """
        row_pixels = self.page.cnt_row_pixels()
        marker = '{}-row'.format(self.page.name)
        eprint('Run carmel on {}'.format(marker))
        cutting_idx = map(lambda idx: idx - 1, Cutter._cutting_points(
            row_pixels, self.carmel_args.n_row, self.carmel_args.r_std1,
            self.carmel_args.r_std2, self.carmel_args.r_penalty, marker))
        return cutting_idx

    def _cut_cols(self, upper, lower):
        """
        cut along columns of a image between upper and lower
        :return: cutting positions, and curves used
        """
        col_pixels = self.page.cnt_col_pixels(upper, lower)
        marker = '{}-col-{}-{}'.format(self.page.name, upper, lower)
        eprint('Run carmel on {}'.format(marker))
        cutting_idx = map(lambda idx: idx - 1, Cutter._cutting_points(
            col_pixels, self.carmel_args.n_col, self.carmel_args.c_std1,
            self.carmel_args.c_std2, self.carmel_args.c_penalty, marker))
        return cutting_idx

    def _cut_all_cols(self, row_ranges):
        """
        cut along columns of all image strips
        :return: array of _cut_cols results
        """
        if sys.platform == 'darwin':  # parallel running on HPC has error.
            eprint('parallel running on macOS ...')
            return PP().map(lambda (u, l): self._cut_cols(u, l), row_ranges)
        else:
            eprint('no parallel running ...')
            return map(lambda (u, l): self._cut_cols(u, l), row_ranges)

    # TODO: this method is wrong
    @classmethod
    def get_ranges(cls, cut_index):
        return filter(
            lambda (u, l): u < l, zip([0] + cut_index, cut_index + [-1]))

    def get_grid(self):
        if self.row_cut is not None and self.col_cuts is not None:
            return self.row_cut, self.col_cuts

        self.row_cut = self._cut_rows()
        row_ranges = Cutter.get_ranges(self.row_cut)
        self.col_cuts = self._cut_all_cols(row_ranges)

        return self.row_cut, self.col_cuts

    def draw_grid(self, row_cut, col_cuts):
        ret_img = np.copy(self.page.img_3dim)
        row_ranges = Cutter.get_ranges(row_cut)
        for i, ((upper, lower), col_cut) in enumerate(zip(row_ranges, col_cuts)):
            if upper != 0:  # do not draw the top border.
                cv2.line(ret_img, (0, upper), (self.page.width - 1, upper),
                         (0, 0, 255), thickness=1)
            for j, idx in enumerate(col_cut):
                cv2.line(ret_img, (idx, upper+1), (idx, lower-1), (0, 255, 0),
                         thickness=1)
                cv2.putText(ret_img, str(i), (idx, lower-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(0, 0, 255))
                cv2.putText(ret_img, str(j), (idx, upper+10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(0, 0, 255))
        return ret_img

    # TODO: Too many hard-code parameters
    @staticmethod
    def almost_white(img):
        h, w = img.shape
        white_cnt = 0
        for i in range(h):
            for j in range(w):
                if img[i, j] == 255:
                    white_cnt += 1
        if float(white_cnt) / (h * w) >= 0.99:
            return True
        return False

    @staticmethod
    def trim_left(image_strip):
        i = 0
        while i < len(image_strip) and Cutter.almost_white(image_strip[i]):
            i += 1
        if i == len(image_strip):
            raise ValueError('encounter all-white line of images')
        return image_strip[i:]

    @staticmethod
    def trim_right(image_strip):
        i = len(image_strip) - 1
        while i >= 0 and Cutter.almost_white(image_strip[i]):
            i -= 1
        if i == -1:
            raise ValueError('encounter all-white line of images')
        return image_strip[:i + 1]
        pass

    @staticmethod
    def trim(image_strip):
        return Cutter.trim_right(Cutter.trim_left(image_strip))

    @staticmethod
    def remove_all_white(image_strip):
        new_strip = []
        for s in image_strip:
            if not Cutter.almost_white(s):
                new_strip.append(s)
        return new_strip

    def cut_grid(self, row_cut, col_cuts):
        self.pieces = []
        counter = 0
        counter_row = 0
        row_ranges = Cutter.get_ranges(row_cut)
        for (upper, lower), col_cut in zip(row_ranges, col_cuts):
            image_strip = []
            indices = []
            col_ranges = Cutter.get_ranges(col_cut)
            counter_col = 0
            for left, right in col_ranges:
                cropped = self.page.img_2dim[upper:lower+1, left:right+1]
                image_strip.append(cropped)
                indices.append((counter, counter_row, counter_col))
                counter_col += 1
                counter += 1
            # instead of removing all spaces, it's better to consider space as
            # a special character.
            # try:
            #     images = Cutter.remove_all_white(image_strip)
            # except ValueError as e:
            #     eprint('Caught ValueError: {}'.format(e))
            #     eprint('Drop the line col-{}-{}.'.format(upper, lower))
            #     continue
            # eprint('col-{}-{}, before/after trimming: {}/{}'.format(
            #     upper, lower, len(image_strip), len(images)))
            self.pieces += list(zip(indices, image_strip))

            counter_row += 1
        return self.pieces


class CutterSampling(Cutter):
    def __init__(self, page, carmel_args, max_height,
                 samples_of_boundaries, row_cut):
        super(CutterSampling, self).__init__(page, carmel_args, max_height)
        self.samples_of_boundaries = samples_of_boundaries
        self.row_cut = row_cut
        self.row_ranges_w_index = \
            list(enumerate(Cutter.get_ranges(self.row_cut)))

    @classmethod
    def _cutting_points(
            cls, black_pixels, mean, std1, std2, p, marker='no_marker'):
        """
        Use Carmel to compute cutting positions.
        align3.sh, make-fsa2.sh, and make-fst2.sh are required.
        :param black_pixels: number of pixels being cut
        :param mean: the mean of total characters in a row,
         or total rows in a page
        :param std1: stddev over number of characters or number of rows
        :param std2: stddev over character size or row size
        :param p: penalty of cutting black pixels
        :param marker: marker to name Carmel running fsts.
        :return: cutting positions
        """
        pwd = path.dirname(path.realpath(__file__))
        exe = '{}/align-sampling.sh'.format(pwd)

        f = NamedTemporaryFile(
            'w', prefix='carmel_input_{}_'.format(marker), delete=False)
        f.write('\n')
        f.write(' '.join(map(lambda c: 'c' + str(c), black_pixels)))
        f.close()

        cmd = "{} {} {} {} {} {}".format(exe, f.name, mean, std1, std2, p)
        pipe = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        out, err = pipe.communicate()
        ret = pipe.wait()
        eprint('marker: {}, returns: {}'.format(marker, out))
        eprint('marker: {}, return code is {}'.format(marker, ret))
        assert out != "", "marker: {}, no output received".format(marker)
        return map(lambda e: int(e), out.split())

    def _cut_cols_with_samples(self, upper, lower, idx):
        """
        cut along columns of a image between upper and lower
        :return: cutting positions, and curves used
        """
        col_pixels = list(self.samples_of_boundaries[idx])
        marker = '{}-col-{}-{}'.format(self.page.name, upper, lower)
        eprint('Run carmel on {}'.format(marker))
        cutting_idx = map(lambda idx: idx - 1, CutterSampling._cutting_points(
            col_pixels, self.carmel_args.n_col, self.carmel_args.c_std1,
            self.carmel_args.c_std2, self.carmel_args.c_penalty, marker))
        return cutting_idx

    def _cut_all_cols_with_samples(self):
        """
        cut along columns of all image strips
        :return: array of _cut_cols results
        """
        if sys.platform == 'darwin':  # parallel running on HPC has error.
            eprint('parallel running on macOS ...')
            return map(
                lambda (i, (u, l)): self._cut_cols_with_samples(u, l, i),
                self.row_ranges_w_index)
        else:
            eprint('no parallel running ...')
            return map(
                lambda (i, (u, l)): self._cut_cols_with_samples(u, l, i),
                self.row_ranges_w_index)

    def get_grid(self):
        self.col_cuts = self._cut_all_cols_with_samples()
        return self.row_cut, self.col_cuts
