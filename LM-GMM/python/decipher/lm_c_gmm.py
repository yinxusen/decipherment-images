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

from decipher.utils import eprint, forward_logmatmul, forward_argmax, \
    backward_logmatmul, gmm_assign, gmm_update, gmm_log_likelihood, \
    cross_entropy, time_to_stop


class LM_C_GMM(object):
    def __init__(self, features, n_clusters, unigram_tbl, bigram_tbl, alphabet,
                 func_weighted_tbl_init, func_subst_tbl_init,
                 use_alternative_update=False, fix_gmm=False):
        super(LM_C_GMM, self).__init__()
        self.k = n_clusters
        self.features = features
        self.unigram_tbl = unigram_tbl
        self.bigram_tbl = bigram_tbl
        self.n, self.m = features.shape
        self.alphabet = alphabet
        self.use_alternative_update = use_alternative_update
        self.fix_gmm = fix_gmm
        self.func_weighted_tbl_init = func_weighted_tbl_init
        self.func_subst_tbl_init = func_subst_tbl_init
        self.link_tbl = None
        self.subst_tbl = None
        self.gmm = None
        self.weights = None
        self.xe = None
        self.ll = None

    def fit(self):
        (self.link_tbl, self.subst_tbl, self.gmm, self.weights, self.xe,
         self.ll) = em_restart(
            self.features, self.unigram_tbl, self.bigram_tbl,
            self.func_weighted_tbl_init, self.func_subst_tbl_init,
            restart=0, use_alternative_update=self.use_alternative_update)


def em_forward_backward(unigram_tbl, bigram_tbl, link_tbl, subst_tbl):
    """
    forward-backward algorithm for one line of observations.
    line: input features in shape (n, m)
    link_tbl: link table in shape (k, n)
    """
    k, n = link_tbl.shape
    r = len(unigram_tbl)

    alpha_tbl = np.full((n, r * k), -np.inf)
    beta_tbl = np.full((n, r), -np.inf)

    sa_tbl = np.full((n, r), -np.inf)
    sb_tbl = np.full((n, r * k), -np.inf)

    sa_tbl[0, :] = unigram_tbl

    for i in range(r):
        alpha_tbl[0, i*k:(i+1)*k] = sa_tbl[0, i] + subst_tbl[i, :]

    for i in range(1, n):
        current_end = logsumexp(
            np.reshape(
                alpha_tbl[i-1, :] + np.tile(link_tbl[:, i-1], r),
                (-1, k)),
            axis=1)
        sa_tbl[i, :] = forward_logmatmul(current_end, bigram_tbl)
        for j in range(r):
            alpha_tbl[i, j*k:(j+1)*k] = sa_tbl[i, j] + subst_tbl[j, :]

    # alpha-final, aka the probability of cipher
    last_end = logsumexp(
        np.reshape(
            alpha_tbl[-1, :] + np.tile(link_tbl[:, -1], r),
            (-1, k)),
        axis=1)
    prb_cf = logsumexp(last_end)

    beta_tbl[-1, :] = 0
    for i in range(r):
        sb_tbl[-1, i*k:(i+1)*k] = beta_tbl[-1, i] + link_tbl[:, -1]

    for i in range(n - 1)[::-1]:
        current_front = logsumexp(
            np.reshape(
                sb_tbl[i+1, :] + subst_tbl.reshape(-1),
                (-1, k)),
            axis=1)
        beta_tbl[i, :] = backward_logmatmul(current_front, bigram_tbl)
        for j in range(r):
            sb_tbl[i, j*k:(j+1)*k] = beta_tbl[i, j] + link_tbl[:, i]

    return alpha_tbl, beta_tbl, sa_tbl, sb_tbl, prb_cf


def em_iter_count(unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
                  only_subst_tbl=False):
    """
    given parameters, assign probabilities to alignments,
    normalize across alignments and count
    """
    k, n = link_tbl.shape
    r = len(unigram_tbl)

    normalization_factor = np.sum(logsumexp(link_tbl, axis=0))
    normalized_link_tbl = link_tbl - logsumexp(link_tbl, axis=0)[np.newaxis, :]
    alpha_tbl, beta_tbl, sa_tbl, sb_tbl, prb_cf = em_forward_backward(
        unigram_tbl, bigram_tbl, normalized_link_tbl, subst_tbl)

    cnt_tbl = (logsumexp(alpha_tbl.reshape(n, r, k)
                         + beta_tbl[:, :, np.newaxis], axis=1).T
               + normalized_link_tbl)
    cnt_subst_tbl = (logsumexp(np.repeat(sa_tbl.T, k, axis=0)
                               + sb_tbl.T, axis=1).reshape(r, k)
                     + subst_tbl)

    # normalize to get fractional count
    cnt_tbl -= prb_cf
    cnt_subst_tbl -= prb_cf

    if not only_subst_tbl:
        prb_cf += normalization_factor
    else:
        prb_cf = prb_cf

    return cnt_tbl, cnt_subst_tbl, prb_cf


def em_iter_update(cnt_tbl, line):
    """
    given count, update parameters
    """
    # only ration between pdfs matters, so we use column normalization
    normalized_tbl = cnt_tbl - logsumexp(cnt_tbl, axis=0)[np.newaxis, :]
    weighted_tbl = np.exp(normalized_tbl)
    gmm, weights = gmm_update(line, weighted_tbl, cov_type='fix')
    link_tbl = gmm_assign(gmm, line)
    return link_tbl, gmm, weights


def em_iter_update_subst(cnt_subst_tbl):
    subst_tbl = cnt_subst_tbl - logsumexp(cnt_subst_tbl, axis=1)[:, np.newaxis]
    return subst_tbl


def viterbi(tokens, unigram_tbl, bigram_tbl, link_tbl, subst_tbl):
    """
    Viterbi decoding with bigram LM.
    :param tokens: input string in terms of tokens. if using GMM to recognize
      images or speeches, use list(range(n)).
    :param unigram_tbl: a vector of unigram with length k
    :param bigram_tbl: states transition probability matrix in shape (k, k)
    :param link_tbl: probability matrix from states to observed tokens
    :return: the best score and path
    """
    n, r = len(tokens), len(unigram_tbl)
    lattice = np.zeros((n, r))
    back_ptr = np.zeros((n, r)).astype('int64')
    subst_link_tbl = np.full((r, n), -np.inf)
    for i in range(r):
        for j in range(n):
            subst_link_tbl[i, j] = logsumexp(subst_tbl[i, :] + link_tbl[:, j])
    lattice[0, :] = unigram_tbl + subst_link_tbl[:, tokens[0]]
    back_ptr[0, :] = -1
    for i in range(1, len(tokens)):
        forward_vec, back_ptr_vec = forward_argmax(lattice[i-1, :], bigram_tbl)
        lattice[i, :] = forward_vec + subst_link_tbl[:, tokens[i]]
        back_ptr[i, :] = back_ptr_vec

    best_end_pos = np.argmax(lattice[-1, :])
    best_score = lattice[-1, best_end_pos]
    best_poses = np.zeros(len(tokens)).astype('int64')
    best_poses[-1] = best_end_pos
    for i in reversed(range(len(tokens) - 1)):
        best_poses[i] = back_ptr[i+1, best_poses[i+1]]
    return best_score, best_poses


def em_decipher(line, unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
                xe_gap=1e-8, max_iter=0):
    """
    EM on a line of features.
    EM iterations stop if matches one of the following conditions:
      1) reach the max_iter
      2) current cross entropy / last cross entropy >= xe_gap
    :return: final link_tbl, gmm model, cross entropy, and log likelihood
    """
    # prepare hyper-parameters
    prev_xe = np.inf

    # start training
    c = 1
    while True:
        cnt_tbl, cnt_subst_tbl, prb_cf = em_iter_count(
            unigram_tbl, bigram_tbl, link_tbl, subst_tbl)
        _, gmm, weights = em_iter_update(cnt_tbl, line)
        # subst_tbl = em_iter_update_subst(cnt_subst_tbl)
        ll_gmm = gmm_log_likelihood(link_tbl, weights)
        x_entropy = cross_entropy([prb_cf], [len(line)])
        eprint('iter {} cross entropy is {}, gap {},'
               ' logP(c) {}, logP_GMM(c) {}'.format(
            c, x_entropy, abs(1.0 - x_entropy / prev_xe), prb_cf, ll_gmm))

        if time_to_stop(c, max_iter, prev_xe, x_entropy, xe_gap):
            break
        elif np.isnan(x_entropy):
            eprint('program end in iter {} caused by nan'.format(c))
            break
        else:
            prev_xe = x_entropy
            c += 1

    return link_tbl, subst_tbl, gmm, weights, x_entropy, prb_cf


def alternative_update_gmm(
        line, unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
        xe_gap, max_iter, prev_xe=np.inf):
    c = 1
    while True:
        cnt_tbl, cnt_subst_tbl, prb_cf = em_iter_count(
            unigram_tbl, bigram_tbl, link_tbl, subst_tbl)
        link_tbl, gmm, weights = em_iter_update(cnt_tbl, line)
        ll_gmm = gmm_log_likelihood(link_tbl, weights)
        x_entropy = cross_entropy([prb_cf], [len(line)])
        eprint('iter-GMM {} cross entropy is {}, gap {},'
               ' logP(c) {}, logP_GMM(c) {}'.format(
            c, x_entropy, abs(1.0 - x_entropy / prev_xe), prb_cf, ll_gmm))
        if time_to_stop(c, max_iter, prev_xe, x_entropy, xe_gap):
            break
        elif np.isnan(x_entropy):
            eprint('program end in iter {} caused by nan'.format(c))
            break
        else:
            prev_xe = x_entropy
            c += 1
    return link_tbl, gmm, weights, x_entropy, prb_cf


def alternative_update_subst(
        unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
        xe_gap, max_iter, prev_xe=np.inf):
    c = 1
    k, n = link_tbl.shape
    normalization_factor = np.sum(logsumexp(link_tbl, axis=0))
    while True:
        cnt_tbl, cnt_subst_tbl, prb_cf = em_iter_count(
            unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
            only_subst_tbl=True)
        subst_tbl = em_iter_update_subst(cnt_subst_tbl)
        x_entropy = cross_entropy([prb_cf], [n])
        eprint('iter-subst {} cross entropy is {}, gap {},'
               ' logP(c) {}'.format(
            c, x_entropy, abs(1.0 - x_entropy / prev_xe), prb_cf))
        if time_to_stop(c, max_iter, prev_xe, x_entropy, xe_gap):
            break
        elif np.isnan(x_entropy):
            eprint('program end in iter {} caused by nan'.format(c))
            break
        else:
            prev_xe = x_entropy
            c += 1
    return subst_tbl, x_entropy, prb_cf, normalization_factor


def em_decipher_alternative(
        line, unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
        xe_gap=1e-8, max_iter=300):

    prev_xe_gmm = np.inf
    prev_xe_subst = np.inf

    t = 1
    while True:
        eprint('start training GMM')
        link_tbl, gmm, weights, x_entropy, prb_cf_gmm = alternative_update_gmm(
            line, unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
            1e-8, max_iter, prev_xe=np.inf)

        if time_to_stop(t, max_iter, prev_xe_gmm, x_entropy, xe_gap):
            break
        else:
            prev_xe_gmm = x_entropy

        eprint('start training subst')
        subst_tbl, x_entropy, prb_cf_subst, normalization_factor =\
            alternative_update_subst(
                unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
                1e-5, max_iter, prev_xe=prev_xe_subst)

        # keep the real prb_cf for later use
        prb_cf_subst += normalization_factor

        if time_to_stop(t, max_iter, prev_xe_subst, x_entropy, 1e-5):
            break
        else:
            prev_xe_subst = x_entropy

        t += 1

    return link_tbl, subst_tbl, gmm, weights, x_entropy, prb_cf_gmm


def em_decipher_fix_gmm(
        unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
        xe_gap=1e-5, max_iter=300):

    eprint('start training subst')
    subst_tbl, x_entropy, prb_cf_subst, normalization_factor =\
        alternative_update_subst(
            unigram_tbl, bigram_tbl, link_tbl, subst_tbl,
            xe_gap, max_iter)

    # keep the real prb_cf for later use
    prb_cf_subst += normalization_factor

    return subst_tbl, x_entropy, prb_cf_subst


def em_restart_fix_gmm(
        unigram_tbl, bigram_tbl, link_tbl,
        subst_tbl_init_function, restart=10):
    """
    EM with random restarts.
    :param weighted_tbl_init_function: A function generating weighted table
      to initialize Gaussian distribution parameters. Let k=#clusters,
      n=#observations, then the table should have k * n dims. Each value in
      cells could be any positive numbers. The rations between values indicate
      the importance of each feature for composing clusters.
    :return: the best link_tbl, gmm model, cross entropy and likelihood after
      all restarts.
    """
    best_subst_tbl = None
    best_xe = np.inf
    best_ll = None

    eprint('start training...')
    for i in range(restart + 1):
        if i > 0:
            eprint('random restart --- {} restarts remaining, '
                   'best corss entropy so far is {}'.format(
                restart - i, best_xe))
        subst_init_tbl = subst_tbl_init_function()

        subst_tbl, xe, ll = em_decipher_fix_gmm(
            unigram_tbl, bigram_tbl, link_tbl, subst_init_tbl)
        if np.isnan(xe):  # jump over nan results
            continue
        if xe < best_xe:
            best_ll = ll
            best_xe = xe
            best_subst_tbl = subst_tbl
    eprint('with {} restarts, '
           'the best cross entropy is {}, '
           'the best log likelihood is {}'.format(restart, best_xe, best_ll))
    return best_subst_tbl, best_xe, best_ll


def em_restart(line, unigram_tbl, bigram_tbl,
               weighted_tbl_init_function, subst_tbl_init_function,
               restart=10, use_alternative_update=False):
    """
    EM with random restarts.
    :param weighted_tbl_init_function: A function generating weighted table
      to initialize Gaussian distribution parameters. Let k=#clusters,
      n=#observations, then the table should have k * n dims. Each value in
      cells could be any positive numbers. The rations between values indicate
      the importance of each feature for composing clusters.
    :return: the best link_tbl, gmm model, cross entropy and likelihood after
      all restarts.
    """
    best_link_tbl = None
    best_subst_tbl = None
    best_gmm = None
    best_xe = np.inf
    best_weights = None
    best_ll = None

    eprint('start training...')
    for i in range(restart + 1):
        eprint('init parameters')
        init = weighted_tbl_init_function()
        gmm, weights = gmm_update(line, init, cov_type='fix')
        link_tbl = gmm_assign(gmm, line)
        if i > 0:
            eprint('random restart --- {} restarts remaining, '
                   'best cross entropy so far is {}'.format(
                restart - i, best_xe))
        subst_init_tbl = subst_tbl_init_function()

        if use_alternative_update:
            decipher_func = em_decipher_alternative
        else:
            decipher_func = em_decipher

        link_tbl, subst_tbl, gmm, weights, xe, ll = decipher_func(
            line, unigram_tbl, bigram_tbl, link_tbl, subst_init_tbl)
        if np.isnan(xe):  # jump over nan results
            continue
        if xe < best_xe:
            best_ll = ll
            best_xe = xe
            best_link_tbl = link_tbl
            best_subst_tbl = subst_tbl
            best_gmm = gmm
            best_weights = weights
    eprint('with {} restarts, '
           'the best cross entropy is {}, '
           'the best log likelihood is {}'.format(restart, best_xe, best_ll))
    return (best_link_tbl, best_subst_tbl, best_gmm, best_weights, best_xe,
            best_ll)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
