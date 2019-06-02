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

import numpy as np

from snn.utils import eprint


class OneshotTrialEvaluator(object):
    """
    Evaluating one-shot trials given tasks and models.
    """
    def __init__(self, snn_models):
        self.snn_models = snn_models

    def test_oneshot(self, data, save_probs=False, probs_path=None):
        """
        Test one-shot trials given data
        Implementing abstract classes to tell the class how to manipulate the
        data.
        """
        inputs = data[:, 0]
        targets = data[:, 1]

        n_correct = 0
        n_task = len(inputs)
        n_way = len(inputs[0][0])
        shape_of_image = inputs[0][0][0].shape

        part_a = np.asarray(map(lambda x: x[0], inputs)).reshape(
            (n_task * n_way,) + shape_of_image)
        part_b = np.asarray(map(lambda x: x[1], inputs)).reshape(
            (n_task * n_way,) + shape_of_image)

        probs = [m.predict([part_a, part_b]) for m in self.snn_models]
        probs = np.asarray([p.reshape((n_task, n_way)) for p in probs])

        probs = np.max(probs, axis=0)

        if save_probs:
            np.savez(probs_path, probs=probs)

        for i, (y_, y) in enumerate(zip(probs, targets)):
            if np.argmin(y_) == np.argmax(y):
                n_correct += 1
            else:
                comments = 'error: {}/{}\n{}\n{}'.format(
                    np.argmin(y_), np.argmax(y),
                    y_, y)
                eprint(comments)
            eprint('correct rate: {}/{}'.format(n_correct, i + 1))
        percent_correct = (100.0 * n_correct / n_task)
        eprint("Got an average of {}% {} way one-shot learning "
               "accuracy".format(percent_correct, n_way))
        return percent_correct
