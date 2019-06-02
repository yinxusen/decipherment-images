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
from sklearn import cluster

from decipher.utils import eprint, viterbi, gmm_assign, gmm_update, \
    cross_entropy, forward_logmatmul, backward_logmatmul, time_to_stop, \
    gmm_log_likelihood


class LMGMM(object):
    def __init__(
            self,
            features, n_clusters,
            unigram_tbl, bigram_tbl, alphabet,
            params_init='kmeans_init', init_2stage=None):
        super(LMGMM, self).__init__()
        self.k = n_clusters
        self.features = features
        self.unigram_tbl = unigram_tbl
        self.bigram_tbl = bigram_tbl
        self.n, self.m = features.shape
        self.alphabet = alphabet
        self.params_init = params_init
        self.init_2stage = init_2stage
        self.link_tbl = None
        self.gmm = None
        self.weights = None
        self.xe = None
        self.ll = None
        self.viterbi_score = None

    def weighted_tbl_init(self):
        if self.params_init == 'kmeans_init':
            eprint('initialize by KMeans')
            weighted_tbl = np.zeros((self.k, self.n))
            label = cluster.KMeans(
                n_clusters=self.k, n_init=1).fit(self.features).labels_
            weighted_tbl[label, np.arange(self.n)] = 1
        elif self.params_init == 'uniform':
            eprint('uniform initialization')
            weighted_tbl = np.ones((self.k, self.n)) / self.k
        elif self.params_init == "2stage":
            eprint('initialize from 2-stage deciphering result')
            idx_alphabet = dict([(c, i) for i, c in enumerate(self.alphabet)])
            init = list(self.init_2stage)
            init = [idx_alphabet[c] for c in init]
            weighted_tbl = np.zeros((self.k, self.n))
            for i in range(len(init)):
                weighted_tbl[init[i], i] = 1
            weighted_tbl = weighted_tbl + 2 * (np.random.random((self.k, self.n)))
            eprint(np.sum(weighted_tbl, axis=1))
        else:
            eprint('random initialization')
            weighted_tbl = np.random.uniform(size=(self.k, self.n))
            weighted_tbl /= np.sum(weighted_tbl, axis=0)[np.newaxis, :]
        return weighted_tbl

    def fit(self):
        self.link_tbl, self.gmm, self.weights, self.xe, self.ll = em_restart(
            self.features, self.unigram_tbl, self.bigram_tbl,
            self.weighted_tbl_init, restart=0)

    def transform(self):
        pass

    def viterbi_path(self):
        score, path = viterbi(
            list(range(self.n)), self.unigram_tbl, self.bigram_tbl,
            self.link_tbl)
        return score, path

    def argmax_path(self):
        return np.argmax(self.link_tbl, axis=0)


class SupervisedLMGMM(LMGMM):
    """
    Supervised GMM for verification purpose
    """
    def __init__(self, features, n_clusters, gold, alphabet,
                 unigram_tbl, bigram_tbl, use_em=False):
        """
        :param gold: the plaintext characters (not cipher characters) of the
                        image-tokens
        :param use_em: "em_gmm" or "em_decipher", continue training with GMM or
                      LM-GMM
        """
        super(SupervisedLMGMM, self).__init__(
            features, n_clusters, unigram_tbl, bigram_tbl, alphabet)
        self.gold = gold
        self.use_em = use_em
        idx_alphabet = dict([(c, i) for i, c in enumerate(self.alphabet)])
        self.targets = [idx_alphabet[c] for c in self.gold]

    def fit(self):
        cnt_tbl = np.zeros((self.k, self.n))

        for i, y in enumerate(self.targets):
            cnt_tbl[y, i] = 1

        eprint('initialize use gold')
        self.gmm, self.weights = gmm_update(self.features, cnt_tbl,
                                            cov_type='fix', scaling_fix_cov=0.1)
        self.link_tbl = gmm_assign(self.gmm, self.features)
        if self.use_em:
            eprint('continue training with LM-GMM')
            self.link_tbl, self.gmm, self.weights, self.xe, self.ll =\
                em_decipher(self.features, self.unigram_tbl, self.bigram_tbl,
                            self.link_tbl)
        else:
            pass

        if self.unigram_tbl is not None and self.bigram_tbl is not None:
            _, _, prb_cf = em_forward_backward(
                self.features, self.unigram_tbl, self.bigram_tbl, self.link_tbl)
            eprint('log likelihood of LM-GMM is {}'.format(prb_cf))
            self.ll = prb_cf
            self.xe = cross_entropy([prb_cf], [self.n])


def em_forward_backward(line, unigram_tbl, bigram_tbl, link_tbl):
    """
    forward-backward algorithm for one line of observations.
    line: input features in shape (n, m)
    link_tbl: link table in shape (k, n)
    """
    alpha_tbl = np.full((len(line), len(unigram_tbl)), -np.inf)
    beta_tbl = np.full((len(line), len(unigram_tbl)), -np.inf)

    alpha_tbl[0, :] = unigram_tbl
    for i in range(1, len(line)):
        alpha_tbl[i, :] = forward_logmatmul(
            (alpha_tbl[i-1, :] + link_tbl[:, i-1]), bigram_tbl)

    # alpha-final, aka the probability of cipher
    prb_cf = logsumexp(alpha_tbl[-1, :] + link_tbl[:, -1])

    beta_tbl[-1, :] = 0
    for j in reversed(range(len(line)-1)):
        beta_tbl[j, :] = backward_logmatmul(
            (beta_tbl[j+1, :] + link_tbl[:, j+1]), bigram_tbl)

    return alpha_tbl, beta_tbl, prb_cf


def em_iter_count(line, unigram_tbl, bigram_tbl, link_tbl):
    """
    given parameters, assign probabilities to alignments,
    normalize across alignments and count
    """
    normalization_factor = np.sum(logsumexp(link_tbl, axis=0))
    normalized_link_tbl = link_tbl - logsumexp(link_tbl, axis=0)[np.newaxis, :]
    alpha_tbl, beta_tbl, prb_cf = em_forward_backward(
        line, unigram_tbl, bigram_tbl, normalized_link_tbl)

    cnt_tbl = (alpha_tbl.T + normalized_link_tbl + beta_tbl.T) - prb_cf

    return cnt_tbl, prb_cf + normalization_factor


def em_iter_update(cnt_tbl, line):
    """
    given count, update parameters
    """
    # only ratio between pdfs is useful
    # that's why we use column normalization here
    normalized_tbl = cnt_tbl - logsumexp(cnt_tbl, axis=0)[np.newaxis, :]
    weighted_tbl = np.exp(normalized_tbl)
    gmm, weights = gmm_update(line, weighted_tbl,
                              cov_type='fix', scaling_fix_cov=0.1)
    link_tbl = gmm_assign(gmm, line)
    return link_tbl, gmm, weights


def em_decipher(line, unigram_tbl, bigram_tbl, link_tbl,
                xe_gap=1e-8, max_iter=300):
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
        cnt_tbl, prb_cf = em_iter_count(line, unigram_tbl, bigram_tbl, link_tbl)
        link_tbl, gmm, weights = em_iter_update(cnt_tbl, line)
        ll_gmm = gmm_log_likelihood(link_tbl, weights)
        x_entropy = cross_entropy([prb_cf], [len(line)])
        eprint('iter {} cross entropy is {}, gap {},'
               ' logP(c) {}, logP_GMM(c) {}'.format(
            c, x_entropy, x_entropy / prev_xe, prb_cf, ll_gmm))

        if time_to_stop(c, max_iter, prev_xe, x_entropy,
                        xe_gap + np.finfo(np.float).eps):
            break
        elif np.isnan(x_entropy):
            eprint('program end in iter {} caused by nan'.format(c))
            break
        else:
            prev_xe = x_entropy
            c += 1

    return link_tbl, gmm, weights, x_entropy, prb_cf


def em_restart(line, unigram_tbl, bigram_tbl, weighted_tbl_init_function,
               restart=10):
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
    best_gmm = None
    best_xe = np.inf
    best_weights = None
    best_ll = None

    eprint('start training...')
    for i in range(restart + 1):
        eprint('init parameters')
        init = weighted_tbl_init_function()
        # print('init:\n{}'.format(init))
        gmm, _ = gmm_update(line, init, cov_type='fix', scaling_fix_cov=0.1)
        link_tbl = gmm_assign(gmm, line)
        if i > 0:
            eprint('random restart --- {} restarts remaining, '
                   'best corss entropy so far is {}'.format(
                restart - i, best_xe))
        link_tbl, gmm, weights, xe, ll = em_decipher(
            line, unigram_tbl, bigram_tbl, link_tbl)
        if np.isnan(xe):  # jump over nan results
            continue
        if xe < best_xe:
            best_ll = ll
            best_xe = xe
            best_link_tbl = link_tbl
            best_gmm = gmm
            best_weights = weights
    eprint('with {} restarts, '
           'the best cross entropy is {}, '
           'the best log likelihood is {}'.format(restart, best_xe, best_ll))
    return best_link_tbl, best_gmm, best_weights, best_xe, best_ll


if __name__ == '__main__':
    import doctest
    doctest.testmod()
