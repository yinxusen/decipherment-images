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
from scipy.misc import logsumexp

from decipher.utils import eprint, forward_logmatmul, backward_logmatmul, \
    cross_entropy


"""
Decipher from transcriptions
Use a character-wise language model with a substitution model to decipher from
transcriptions.
This could can be treated as the Python version of Carmel decipherment code.
"""


def em_forward_backward(line, unigram_tbl, bigram_tbl, link_tbl):
    alpha_tbl = np.full((len(line), len(unigram_tbl)), -np.inf)
    beta_tbl = np.full((len(line), len(unigram_tbl)), -np.inf)
    alpha_tbl[0, :] = unigram_tbl
    for i in range(1, len(line)):
        alpha_tbl[i, :] = forward_logmatmul(
            (alpha_tbl[i-1, :] + link_tbl[:, line[i-1]]), bigram_tbl)

    # alpha-final, aka the probability of cipher
    prb_cf = logsumexp(alpha_tbl[-1, :] + link_tbl[:, line[-1]])

    beta_tbl[-1, :] = 0
    for j in reversed(range(len(line)-1)):
        beta_tbl[j, :] = backward_logmatmul(
            (beta_tbl[j+1, :] + link_tbl[:, line[j+1]]), bigram_tbl)
    return alpha_tbl, beta_tbl, prb_cf


def em_iter_count(line, unigram_tbl, bigram_tbl, link_tbl):
    cnt_tbl = np.full(link_tbl.shape, -np.inf)
    alpha_tbl, beta_tbl, prb_cf = em_forward_backward(
        line, unigram_tbl, bigram_tbl, link_tbl)
    for i, c in enumerate(line):
        cnt_tbl[:, c] = logsumexp(
            np.asarray(
                [cnt_tbl[:, c],
                 alpha_tbl[i, :] + link_tbl[:, c] + beta_tbl[i, :]]),
            axis=0)
    cnt_tbl -= prb_cf
    return cnt_tbl, prb_cf


def em_iter_update(cnt_tbl):
    link_tbl = cnt_tbl - logsumexp(cnt_tbl, axis=1)[:, np.newaxis]
    return link_tbl


def small_test_example(logspace=False):
    """
    Example comes from CSCI662 course note page 73~75.
    Right answer (alpha_tbl, beta_tbl) after 1-pass forward-backward algorithm is:
        [[ 0.6    0.4  ]
         [ 0.36   0.14 ]
         [ 0.171  0.079]]
        [[ 0.25  0.25]
         [ 0.5   0.5 ]
         [ 1.    1.  ]]
    """
    lines = [[0, 1, 0]]
    unigram_tbl = np.asarray([0.6, 0.4])
    bigram_tbl = np.asarray([[0.6, 0.4], [0.9, 0.1]])
    link_tbl = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    alpha_tbl = np.asarray(
        [[0.6, 0.4],
         [0.36, 0.14],
         [0.171, 0.079]])
    beta_tbl = np.asarray(
        [[0.25, 0.25],
         [0.5, 0.5],
         [1., 1.]])

    if logspace:
        unigram_tbl = np.log(unigram_tbl)
        bigram_tbl = np.log(bigram_tbl)
        link_tbl = np.log(link_tbl)
        alpha_tbl = np.log(alpha_tbl)
        beta_tbl = np.log(beta_tbl)
    return lines, unigram_tbl, bigram_tbl, link_tbl, alpha_tbl, beta_tbl


def em_decipher(line, unigram_tbl, bigram_tbl, link_tbl_size=None):
    # prepare initial data and parameters
    if link_tbl_size is None:  # only for simple substitution ciphers
        link_tbl_size = (len(unigram_tbl), len(unigram_tbl))
    link_tbl = np.random.random(link_tbl_size)
    sum_tbl = np.sum(link_tbl, axis=1)
    nonzero = sum_tbl > 0
    link_tbl[nonzero] /= sum_tbl[nonzero, None]
    link_tbl = np.log(link_tbl)

    # prepare hyper-parameters
    xe_gap = 0.99999
    max_iter = 300
    prev_xe = np.inf

    # start training
    c = 1
    while True:
        cnt_tbl, pcf = em_iter_count(line, unigram_tbl, bigram_tbl, link_tbl)
        link_tbl = em_iter_update(cnt_tbl)
        x_entropy = cross_entropy([pcf], [len(line)])
        best_pc = pcf
        eprint('iter {} cross entropy is {}, gap {}, logP(c) {}'.format(
            c, x_entropy, x_entropy / prev_xe, pcf))
        if c >= max_iter or x_entropy / prev_xe >= xe_gap:
            break
        else:
            prev_xe = x_entropy
            c += 1

    return link_tbl, x_entropy, best_pc
