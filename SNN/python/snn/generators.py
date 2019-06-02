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

import os
from abc import abstractmethod, ABCMeta

import dill as pickle
import numpy as np
import numpy.random as rng
from scipy import io
from sklearn.utils import shuffle

from snn.utils import eprint


class SNNGenerator(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def input_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def data_size(self):
        raise NotImplementedError()

    @abstractmethod
    def train_generator(self, batch_size):
        raise NotImplementedError()

    @abstractmethod
    def dev_generator(self, batch_size):
        raise NotImplementedError()

    @abstractmethod
    def mk_oneshot_task(self, n_way):
        raise NotImplementedError()


class OmniglotGenerator(SNNGenerator):
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, path):
        super(OmniglotGenerator, self).__init__()
        self.data = {}
        self.classes = {}
        with open(os.path.join(path, "train.pickle"), "r") as f:
            self.data['train'], self.classes['train'] = pickle.load(f)
        with open(os.path.join(path, "val.pickle"), "r") as f:
            self.data['val'], self.classes['val'] = pickle.load(f)
        num_trains, num_examples, w, h = self.data['train'].shape
        eprint('(#classes, #examples, w, h): {}'.format(
            (num_trains, num_examples, w, h)))
        self.input_shape = (w, h, 1)
        # estimated number of positive pairs, let C = num_trains,
        # E = num_examples, then it equals C * choose(E, 2).
        estimated_positive_pairs = \
            num_trains * num_examples * (num_examples - 1) // 2
        eprint('estimated # positive-pairs: {}'.format(
            estimated_positive_pairs))
        self.data_size = estimated_positive_pairs

    def input_shape(self):
        return self.input_shape

    def data_size(self):
        return self.data_size

    def _generator(self, batch_size, s="train"):
        while True:
            pairs, targets = self._get_batch(batch_size, s)
            yield pairs, targets

    def train_generator(self, batch_size):
        return self._generator(batch_size, s='train')

    def dev_generator(self, batch_size):
        return self._generator(batch_size, s='val')

    def _get_batch(self, batch_size, s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X = self.data[s]
        n_classes, n_examples, w, h = X.shape
        pairs = [np.zeros((batch_size, w, h, 1))] * 2
        targets = np.zeros((batch_size,))

        half_batch = batch_size // 2

        targets[half_batch:] = 1

        categories_1 = rng.choice(n_classes, size=(batch_size,), replace=False)
        first_half_2 = categories_1[:half_batch]
        second_half_2 = np.asarray(map(
            lambda c: (c + rng.randint(1, n_classes)) % n_classes,
            categories_1[half_batch:]))
        categories_2 = np.concatenate((first_half_2, second_half_2), axis=0)
        which_examples_1 = rng.choice(n_examples, size=batch_size, replace=True)
        which_examples_2 = rng.choice(n_examples, size=batch_size, replace=True)

        eprint('generating a batch of {}'.format(batch_size))

        pairs[0] = X[categories_1, which_examples_1].reshape(batch_size,
                                                             w, h, 1)
        pairs[1] = X[categories_2, which_examples_2].reshape(batch_size,
                                                             w, h, 1)
        return pairs, targets

    def mk_oneshot_task(self, n_way, s='val', language=None):
        X = self.data[s]
        n_classes, n_examples, w, h = X.shape
        low, high = self.classes[s][language] if (
            language is not None) else 0, n_classes
        true_class = rng.choice(range(low, high), size=(1,), replace=False)
        support_classes = rng.choice(
            range(low, true_class[0]) + range(true_class[0] + 1, high),
            size=(n_way - 1,),
            replace=False if n_way <= high - low else True)

        indices = rng.randint(0, n_examples, size=(n_way,))
        selected_classes = np.concatenate((true_class, support_classes))

        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))

        test_image = np.asarray([X[true_class, ex1, :, :]] * n_way) \
            .reshape(n_way, w, h, 1)

        support_set = X[selected_classes, indices, :, :]
        support_set[0, :, :] = X[true_class, ex2]
        support_set = support_set.reshape(n_way, w, h, 1)

        targets = np.zeros((n_way,))
        targets[0] = 1

        targets, test_image, support_set = shuffle(
            targets, test_image, support_set)

        pairs = [test_image, support_set]

        return pairs, targets


class MnistGenerator(SNNGenerator):
    def __init__(self, path):
        super(MnistGenerator, self).__init__()

        mnist = io.loadmat('{}/mnist-original.mat'.format(path))
        X = mnist['data'].T  # change from Matlab format to Numpy one
        y = mnist['label'].squeeze()  # ditto
        eprint('load X with shape {}'.format(X.shape))
        eprint('load y with shape {}'.format(y.shape))

        # scaled_X = np.asarray(
        #     map(lambda x: cv2.resize(x.reshape(28, 28), (105, 105)), X))
        scaled_X = np.asarray(map(lambda x: x.reshape(28, 28), X))
        rearranged_X = map(lambda i: scaled_X[y == i], range(10))
        trim_to_dim = min(map(lambda x: x.shape[0], rearranged_X))
        self.data = 255.0 - np.asarray(map(lambda x: x[:trim_to_dim],
                                           rearranged_X))

    def input_shape(self):
        raise NotImplementedError()

    def data_size(self):
        raise NotImplementedError()

    def train_generator(self, batch_size):
        pass

    def dev_generator(self, batch_size):
        pass

    def mk_oneshot_task(self, n_way):
        """
        Create pairs of test image,
        support set for testing N way one-shot learning.
        """
        X = self.data
        _, _, w, h = X.shape
        n_classes, n_examples = X.shape[0], X.shape[1]
        categories = rng.choice(range(n_classes), size=(n_way,),
                                replace=False)

        indices = rng.randint(0, n_examples, size=(n_way,))

        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
        test_image = np.asarray([X[true_category, ex1, :, :]] * n_way) \
            .reshape(n_way, w, h, 1)
        support_set = X[categories, indices, :, :]
        support_set[0, :, :] = X[true_category, ex2]
        support_set = support_set.reshape(n_way, w, h, 1)
        targets = np.zeros((n_way,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(
            targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets


class GoldGenerator(SNNGenerator):
    """
    one-shot tasks generator with gold transcriptions.
    In this generator, targets are given in a gold transcription. Images should
    first be matched with the gold transcription to get its target.
    """

    def __init__(self, path):
        super(GoldGenerator, self).__init__()

        loaded = np.load('{}/features.npz'.format(path))
        self.images = loaded['features']
        self.targets = loaded['targets']
        assert len(self.images) == len(self.targets), \
            '#images does not match #targets'
        self.h, self.w = self.images[0].shape[:2]
        eprint('targets = {}'.format(self.targets))
        self.alphabet = loaded['alphabet']
        eprint('number of chars in the alphabet: {}'.format(
            len(self.alphabet)))
        self.data = map(lambda g: self.images[self.targets == g],
                        self.alphabet)
        self.indices = map(lambda g: np.where(self.targets == g)[0],
                           self.alphabet)
        self.n_images, self.n_alphabet = len(self.images), len(self.alphabet)
        self.alphabet_basis = self._alphabet_basis()
        eprint('generate alphabet basis with shape {}'.format(
            self.alphabet_basis.shape))

        self.data_size = np.sum([len(d) * (len(d) - 1) // 2 for d in self.data])

    def _alphabet_basis(self):
        support_set = np.zeros((self.n_alphabet, self.h, self.w))

        for i in range(self.n_alphabet):
            idx = rng.choice(len(self.data[i]), size=1, replace=False)
            support_set[i, :, :] = self.data[i][idx, :, :]

        return support_set

    def input_shape(self):
        return self.h, self.w, 1

    def data_size(self):
        return self.data_size

    def _generator(self, batch_size):
        while True:
            pairs, targets = self._get_batch(batch_size)
            yield pairs, targets

    def train_generator(self, batch_size):
        return self._generator(batch_size)

    def dev_generator(self, batch_size):
        return self._generator(batch_size)

    def _get_batch(self, batch_size):
        """Create batch of n pairs, half same class, half different class"""
        X = self.alphabet_basis
        n_classes, n_examples, h, w = X.shape
        pairs = [np.zeros((batch_size, w, h, 1))] * 2
        targets = np.zeros((batch_size,))

        half_batch = batch_size // 2

        targets[half_batch:] = 1

        categories_1 = rng.choice(n_classes, size=(batch_size,), replace=False)
        first_half_2 = categories_1[:half_batch]
        second_half_2 = np.asarray(map(
            lambda c: (c + rng.randint(1, n_classes)) % n_classes,
            categories_1[half_batch:]))
        categories_2 = np.concatenate((first_half_2, second_half_2), axis=0)
        which_examples_1 = rng.choice(n_examples, size=batch_size, replace=True)
        which_examples_2 = rng.choice(n_examples, size=batch_size, replace=True)

        eprint('generating a batch of {}'.format(batch_size))

        pairs[0] = X[categories_1, which_examples_1].reshape(batch_size,
                                                             h, w, 1)
        pairs[1] = X[categories_2, which_examples_2].reshape(batch_size,
                                                             h, w, 1)
        return pairs, targets

    @classmethod
    def _high_entropy_choose(cls, a, size):
        try:
            return rng.choice(a, size=size, replace=False)
        except ValueError:  # cache non-enough data
            return rng.choice(a, size=size, replace=True)

    def mk_oneshot_task(self, n_way):
        X = self.data

        n_classes = len(X)
        categories = rng.choice(range(n_classes), size=(n_way,),
                                replace=False)

        true_category = categories[0]
        ex1, ex2 = self._high_entropy_choose(len(X[true_category]), size=(2,))

        test_image = np.asarray([X[true_category][ex1, :, :]] * n_way).reshape(
            (n_way, self.w, self.h, 1))
        test_pos = np.asarray([self.indices[true_category][ex1]] * n_way,
                              dtype='int64')

        support_set = np.zeros((n_way, self.h, self.w))
        support_set[0, :, :] = X[true_category][ex2, :, :]
        support_pos = np.zeros(n_way, dtype='int64')
        support_pos[0] = self.indices[true_category][ex2]
        for i in range(1, n_way):
            cat = categories[i]
            idx = rng.choice(len(X[cat]), 1, replace=False)
            support_set[i, :, :] = X[cat][idx, :, :]
            support_pos[i] = self.indices[cat][idx]

        support_set = support_set.reshape((n_way, self.h, self.w, 1))

        targets = np.zeros((n_way,))
        targets[0] = 1
        targets, test_image, support_set, test_pos, support_pos = shuffle(
            targets, test_image, support_set, test_pos, support_pos)
        pairs = [test_image, support_set]
        pairs_pos = [test_pos, support_pos]

        return pairs, targets, pairs_pos

