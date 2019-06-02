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

from __future__ import print_function

import sys
from itertools import chain, imap

import numpy as np
from keras.models import model_from_json


def flatmap(f, items):
    return list(chain.from_iterable(imap(f, items)))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def padding_img(img, step=10, specified_width=None):
    """
    Make pad to an image to make sure its width can be divided by step.
    :param img: a two-dim black-n-white image, white is the background color.
    :param step: the step to go over the width of the image.
    :param specified_width: A specified width to pad.
    :return: padded image with new width
    """
    h, w = img.shape
    pad_img = img
    if specified_width is None:
        if w % step != 0:
            padding = np.full((h, step - (w % step)), 255, dtype='uint8')
            pad_img = np.concatenate([img, padding], axis=1)
    else:
        padding = np.full((h, specified_width-w), 255, dtype='uint8')
        pad_img = np.concatenate([img, padding], axis=1)
    return pad_img


def load_model(path, prefix):
    """
    Load Keras NN model
    :param path:
    :param prefix:
    :return:
    """
    json_file = open('{}/{}.json'.format(path, prefix), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('{}/{}.h5'.format(path, prefix))
    return model
