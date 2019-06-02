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
import subprocess
import tempfile
from itertools import groupby

import numpy as np

from decipher.utils import levenshtein


def _carmel_align(num_golds, num_clusters, gold_lines, sys_lines):
    """
    Find the best alignment between gold transcription and auto transcription.
    Carmel is used to do the alignment.
    match.sh and make-fst-eval.sh are required to run Carmel.

    :param num_golds: the size of gold alphabet
    :param num_clusters: the number of clusters
    :param gold_lines: A list of gold strings
    :param sys_lines: A list of auto transcription strings
    :return: A list of tuples (x, y), cluster-x maps to gold-character-y.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    carmel_exe = '{}/match.sh'.format(current_dir)

    f = tempfile.NamedTemporaryFile(
        'w', prefix='carmel_input_', delete=False)

    for gline, sline in zip(gold_lines, sys_lines):
        f.write(sline + '\n')
        f.write(gline + '\n')
    f.close()

    ret_value = subprocess.Popen(
        carmel_exe + " " + f.name + " " + str(num_golds) + " " +
        str(num_clusters),
        shell=True, stdout=subprocess.PIPE).stdout.readlines()

    trans_prob = sorted(map(lambda l: (l[0], l[1], float(l[2])), map(
        lambda l: l.split(), ret_value)), key=lambda l: int(l[0][1:]))
    max_arcs = map(
        lambda (k, g): max(g, key=lambda l: l[2]), groupby(
            trans_prob, key=lambda l: l[0]))
    return max_arcs


def split_line(line, num_lines=10):
    ary = line.split()
    step = len(ary) / num_lines
    newlines = []
    i = 0
    while i < len(ary):
        start = i
        end = min(len(ary), i + step)
        newlines.append(' '.join(ary[start:end]))
        i += step
    return newlines


def em_align(gold_str, sys_str):
    """
    Use EM algorithm to align gold string and auto transcription string.

    :param gold_str: A string of gold
    :param sys_str: A string of auto-transcription
    :return: A dict of (k, v), cluster-k maps to gold-v.
    """
    idx_gold = ([(c, 'g{}'.format(i))
                 for i, c in enumerate(np.unique(list(gold_str)))])
    gold_dict = dict(idx_gold)
    inv_gold_dict = dict([(y, x) for x, y in idx_gold])
    num_gold = len(gold_dict)
    idx_sys = ([(c, 'c{}'.format(i))
                for i, c in enumerate(np.unique(list(sys_str)))])
    sys_dict = dict(idx_sys)
    inv_sys_dict = dict([(y, x) for x, y in idx_sys])
    num_sys = len(sys_dict)

    subst_gold = ' '.join([gold_dict[c] for c in gold_str])
    subst_sys = ' '.join([sys_dict[c] for c in sys_str])

    matched = _carmel_align(
        num_gold, num_sys, split_line(subst_gold), split_line(subst_sys))
    matched_dict = dict(
        map(lambda (x, y, w): (inv_sys_dict[x], inv_gold_dict[y]), matched))
    return matched_dict


def get_ned(gold_str, sys_str):
    """
    Compute Normalized Edit Distance after alignment.

    :param gold_str: A string of gold
    :param sys_str: A string of auto-transcription
    :return: Normalized Edit Distance
    """
    aligned = em_align(gold_str, sys_str)
    print(aligned)
    aligned_sys = [aligned[i] for i in sys_str]
    return levenshtein(aligned_sys, gold_str) * 1.0 / len(gold_str)
