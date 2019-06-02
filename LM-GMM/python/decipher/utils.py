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
from itertools import chain, imap, groupby

import numpy as np
from scipy.misc import logsumexp


def flatmap(f, items):
    return list(chain.from_iterable(imap(f, items)))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def line_breaks(indices):
    """
    Extract line breaks from indices.
    The indices is in a format:
      array([[u'copiale-001-pieces-square', u'0', u'0', u'0'],
             [u'copiale-001-pieces-square', u'1', u'0', u'1'],
             [u'copiale-001-pieces-square', u'2', u'0', u'2'],
             ...,
             [u'copiale-010-pieces-square', u'852', u'16', u'45'],
             [u'copiale-010-pieces-square', u'853', u'16', u'46'],
             [u'copiale-010-pieces-square', u'854', u'16', u'47']],
            dtype='<U25')
    the first column is image file name, the second column is a unified index
    for all image-characters, the third column is line index in each page,
    and the last column is image-character index in each line.
    :param indices: as explained above
    :return: a numpy array of index in shape (n, 2), n is the number of lines.
      e.g. [[0, 4], [5, 7], [8, 12], ...]
    """
    line_idx = list(indices[:, 2])
    count_per_line = map(lambda (x, y): (x, len(list(y))), groupby(line_idx))
    idx_per_line = []
    i = 0
    i_start = 0
    while i < len(count_per_line):
        cnt = count_per_line[i][1]
        idx_per_line.append((i_start, i_start+cnt))
        i_start += cnt
        i += 1
    return idx_per_line


def levenshtein(a, b):
    """
    Courtesy: http://hetland.org/coding/python/levenshtein.py
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def log_p_e(idx_line, unigram_tbl, bigram_tbl):
    """
    compute log p(line) under a given bigram model
    :param idx_line: line with characters indexed via alphabet
    :param unigram_tbl: log unigram model
    :param bigram_tbl: log bigram model, b[p, s] = p(s | p)
    :return: log of the probability
    """
    bi_part = [bigram_tbl[p, s] for p, s in zip(idx_line[:-1], idx_line[1:])]
    uni_part = unigram_tbl[idx_line[0]]
    return uni_part + np.sum(bi_part)


def log_tri_p_e(idx_line, unigram_tbl, bigram_tbl, trigram_tbl):
    """
    compute log p(line) under a given trigram model
    :param idx_line: line with characters indexed via alphabet
    :param unigram_tbl: log unigram model
    :param bigram_tbl: log bigram model, b[p, s] = p(s | p)
    :param trigram_tbl: log trigram model, t[p*k+s, s*k+r] = p(sr | ps)
    :return: log of the probability
    """
    k = len(unigram_tbl)
    tri_part = ([trigram_tbl[p*k+s, s*k+r]
                 for p, s, r in
                 zip(idx_line[:-2], idx_line[1:-1], idx_line[2:])])
    bi_part = bigram_tbl[idx_line[0], idx_line[1]]
    uni_part = unigram_tbl[idx_line[0]]
    return uni_part + bi_part + np.sum(tri_part)


def cross_entropy(pcfs, ns):
    """
    compute cross entropy given lines of inputs, and probabilities of each line.
    :param pcfs: probability of each line
    :param ns: length of each line
    :return: cross entropy across all lines
    """
    return np.average(map(lambda (p, n): -p / n, zip(pcfs, ns)))


def forward_logmatmul(alpha_prev_vec, transit_mat):
    """
    forward pass log matrix multiplication
    to compute alpha[t, :], n-dim:
    alpha_prev_vec: alpha[t-1, :], n-dim
    transit_mat: t: n*n-dim, row is start states, column is end states
    >>> prb_a = np.random.random((10))
    >>> prb_b = np.random.random((10, 10))
    >>> mul = np.matmul(prb_a, prb_b)
    >>> log_mul = forward_logmatmul(np.log(prb_a), np.log(prb_b))
    >>> np.all(np.isclose(np.log(mul), log_mul))
    True
    """
    alpha_current_vec = logsumexp(
        alpha_prev_vec.reshape((-1, 1)) + transit_mat, axis=0)
    return alpha_current_vec


def forward_argmax(alpha_prev_vec, transit_mat):
    """
    forward pass argmax selection for Viterbi decoding
    similar process with forward_logmatmul, just change logsumexp to argmax
    """
    forward_mat = alpha_prev_vec.reshape((-1, 1)) + transit_mat
    back_ptr_vec = np.argmax(forward_mat, axis=0)
    alpha_current_vec = forward_mat[back_ptr_vec, range(len(back_ptr_vec))]
    return alpha_current_vec, back_ptr_vec


def backward_logmatmul(beta_next_vec, transit_mat):
    """
    backward pass log matrix multiplication
    to compute beta[t, :], n-dim:
    beta_next_vec: alpha[t+1, :], n-dim
    transit_mat: t: n*n-dim, row is start states, column is end states
    >>> prb_a = np.random.random((10))
    >>> prb_b = np.random.random((10, 10))
    >>> mul2 = np.matmul(prb_b, prb_a)
    >>> log_mul2 = backward_logmatmul(np.log(prb_a), np.log(prb_b))
    >>> np.all(np.isclose(np.log(mul2), log_mul2))
    True
    """
    beta_current_vec = logsumexp(beta_next_vec + transit_mat, axis=1)
    return beta_current_vec


def gmm_log_likelihood(link_tbl, weights):
    """
    log likelihood of GMM
    ll(M) = \sum_{i=1}^n \log \sum_{j=1}^k p(z_j) * p(g_i|z_j)
    :param link_tbl: link table (k, n) in log probability format
    :param weights: p(z)
    :return: log likelihood of the GMM
    """
    log_weights = np.log(weights)
    weighted_link_tbl = link_tbl + log_weights[:, np.newaxis]
    ll_per_sample = logsumexp(weighted_link_tbl, axis=0)
    ll_total = np.sum(ll_per_sample)
    return ll_total


def gmm_update(features, weighted_tbl, cov_type='diag', scaling_fix_cov=0.1):
    """
    given fractional counts, update gmm

    :param weighted_tbl: this weighted table is the same as
      weighted_tbl_init_function.

    >>> line = np.array([[1.0, 2.0], [1.0, 2.2], [2.0, 1.2], [0.8, 10.0]])
    >>> prb_tbl = np.array(
    ...     [[0.8, 0.2, 1.6, 0.2], [1.0, 0.5, 2.5, 0.1], [1.0, 0.1, 1.5, 1.1]])
    >>> gmm, _ = gmm_update(line, prb_tbl)
    >>> gmm
    array([[  1.55714286,   2.12857143,   0.26387855,   4.92347039],
           [  1.60487805,   1.73170732,   0.24485525,   1.8841176 ],
           [  1.34594595,   4.05945946,   0.2976197 ,  15.0461953 ]])
    """
    n, m = features.shape
    k, _ = weighted_tbl.shape
    nk = np.sum(weighted_tbl, axis=1)
    # first n_feature is mean, second one is sd^2
    means = np.dot(weighted_tbl, features) / nk[:, np.newaxis]

    # estimate cov by diagonal cov type
    reg_covar = 1e-6
    avg_X2 = np.dot(weighted_tbl, features * features) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(weighted_tbl, features) / nk[:, np.newaxis]
    cov = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

    if cov_type == 'unit':
        single_mean = cov.mean()
        cov = np.full(cov.shape, single_mean)
    elif cov_type == 'unit-max':
        single_mean = cov.max()
        cov = np.full(cov.shape, single_mean)
    elif cov_type == 'spherical':
        vector_mean = cov.mean(axis=1).reshape((-1, 1))
        cov = np.tile(vector_mean, [1, m])
    elif cov_type == 'diag':
        pass
    elif cov_type == 'fix':
        cov = np.ones((k, m)) * scaling_fix_cov
    else:
        raise ValueError('unknown argument of cov_type: {}'.format(cov_type))

    gmm = np.hstack([means, cov])
    return gmm, nk / np.sum(nk)


def gmm_assign(gmm, features):
    """
    given gmm, update link_tbl for a line of features

    >>> line = np.array([[1.0, 2.0], [1.0, 2.2], [2.0, 1.2], [0.8, 10.0]])
    >>> prb_tbl = np.array(
    ...    [[0.8, 0.2, 1.6, 0.2], [1.0, 0.5, 2.5, 0.1], [1.0, 0.1, 1.5, 1.1]])
    >>> gmm, _ = gmm_update(line, prb_tbl, unit_var=False)
    >>> gmm_assign(gmm, line)
    array([[ -2.55859431,  -2.55743369,  -2.42793047,  -9.34722761],
           [ -2.21729496,  -2.25638941,  -1.84489142, -20.9162996 ],
           [ -2.9294749 ,  -2.90342899,  -3.57786383,  -4.26092732]])
    """
    k, _ = gmm.shape
    n, n_feature = features.shape
    mean = gmm[:, :n_feature]
    sd_square = gmm[:, n_feature:]
    precisions_chol = 1. / np.sqrt(sd_square)
    log_det = (np.sum(np.log(precisions_chol), axis=1))
    precisions = precisions_chol ** 2
    log_prob = (np.sum((mean ** 2 * precisions), 1) -
                2. * np.dot(features, (mean * precisions).T) +
                np.dot(features ** 2, precisions.T))
    link_tbl = -.5 * (n_feature * np.log(2 * np.pi) + log_prob) + log_det
    link_tbl = link_tbl.T

    return link_tbl


def viterbi(tokens, unigram_tbl, bigram_tbl, link_tbl):
    """
    Viterbi decoding with bigram LM.
    :param tokens: input string in terms of tokens. if using GMM to recognize
      images or speeches, use list(range(n)).
    :param unigram_tbl: a vector of unigram with length k
    :param bigram_tbl: states transition probability matrix in shape (k, k)
    :param link_tbl: probability matrix from states to observed tokens
    :return: the best score and path
    """
    n, k = len(tokens), len(unigram_tbl)
    lattice = np.zeros((n, k))
    back_ptr = np.zeros((n, k)).astype('int64')
    lattice[0, :] = unigram_tbl + link_tbl[:, tokens[0]]
    back_ptr[0, :] = -1
    for i in range(1, len(tokens)):
        forward_vec, back_ptr_vec = forward_argmax(lattice[i-1, :], bigram_tbl)
        lattice[i, :] = forward_vec + link_tbl[:, tokens[i]]
        back_ptr[i, :] = back_ptr_vec

    best_end_pos = np.argmax(lattice[-1, :])
    best_score = lattice[-1, best_end_pos]
    best_poses = np.zeros(len(tokens)).astype('int64')
    best_poses[-1] = best_end_pos
    for i in reversed(range(len(tokens) - 1)):
        best_poses[i] = back_ptr[i+1, best_poses[i+1]]
    return best_score, best_poses


def trigram_viterbi(tokens, unigram_tbl, bigram_tbl, trigram_tbl, link_tbl):
    """
    Viterbi decoding with trigram LM.
    :param tokens: input string in terms of tokens. if using GMM to recognize
      images or speeches, use list(range(n)).
    :param unigram_tbl: a vector of unigram with length k
    :param bigram_tbl: (x -> y) pairwise states transition probability matrix
                       in shape (k, k)
    :param trigram_tbl: (xy -> yz) tri-states transition probability matrix
                        in shape (k^2, k^2)
    :param link_tbl: probability matrix from states to observed tokens
    :return: the best score and path
    """
    n, k = len(tokens), len(unigram_tbl)
    n_trigram_states = len(trigram_tbl)

    lattice = np.zeros((n, n_trigram_states))
    back_ptr = np.zeros((n, n_trigram_states)).astype('int64')

    xp_unigram_tbl = np.tile(unigram_tbl, k)
    xp_link_tbl = np.tile(link_tbl, (k, 1))
    xp_bigram_tbl = np.tile(bigram_tbl.flatten('C'), (n_trigram_states, 1))

    # special handling for the first column using unigram_tbl
    lattice[0, :] = xp_unigram_tbl + xp_link_tbl[:, tokens[0]]
    back_ptr[0, :] = -1
    # special handling for the second column using bigram_tbl
    forward_vec_1, back_ptr_1 = forward_argmax(lattice[0, :], xp_bigram_tbl)
    lattice[1, :] = forward_vec_1 + xp_link_tbl[:, tokens[1]]
    back_ptr[1, :] = back_ptr_1

    for i in range(2, len(tokens)):
        forward_vec, back_ptr_vec = forward_argmax(lattice[i-1, :], trigram_tbl)
        lattice[i, :] = forward_vec + xp_link_tbl[:, tokens[i]]
        back_ptr[i, :] = back_ptr_vec

    best_end_pos = np.argmax(lattice[-1, :])
    best_score = lattice[-1, best_end_pos]
    best_poses = np.zeros(n).astype('int64')
    best_poses[-1] = best_end_pos
    for i in reversed(range(len(tokens) - 1)):
        best_poses[i] = back_ptr[i+1, best_poses[i+1]]
    return best_score, [p % k for p in best_poses]


def time_to_stop(t, max_iter, prev_score, curr_score, gap):
    """
    condition of whether to stop EM iterations

    Note that even we use EM here for LM-GMM, LM-C-GMM, and GMM,
    the EM's guarantee of monotonic convergence may not be hold here because of
    changing covariance of Gaussian distributions, especially when dealing with
    language model in the same time.

    Even so, the EM only changes back-n-force around a local maximum. So
    training results would not be affected.

    :param t: current step
    :param max_iter: max step
    :param prev_score: previous cross entropy
    :param curr_score: current cross entropy
    :param gap: gap between curr and prev
    """
    time_judge = t >= max_iter

    if curr_score >= prev_score:
        condition_judge = True
    else:
        condition_judge = abs(1.0 - curr_score / prev_score) <= gap
    return time_judge or condition_judge

