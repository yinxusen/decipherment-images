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

from decipher.utils import eprint, gmm_log_likelihood, gmm_update, gmm_assign, \
    cross_entropy, time_to_stop


class GMM(object):
    def __init__(
            self, features, n_clusters, params_init='kmeans_init',
            cov_type='fix', scaling_factor=0.1):
        super(GMM, self).__init__()
        self.cluster = None
        self.k = n_clusters
        self.features = features
        self.n = len(features)
        self.link_tbl = None
        self.gmm = None
        self.weights = None
        self.params_init = params_init
        self.cov_type = cov_type
        self.scaling_factor = scaling_factor
        eprint('cov type is {}'.format(cov_type))
        eprint('scaling factor is {}'.format(scaling_factor))

    def link_tbl_init(self):
        if self.params_init == 'kmeans_init':
            eprint('initialize by KMeans')
            link_tbl = np.zeros((self.k, self.n))
            label = cluster.KMeans(
                n_clusters=self.k, n_init=1).fit(self.features).labels_
            link_tbl[label, np.arange(self.n)] = 1
        else:
            eprint('random initialization')
            link_tbl = np.random.uniform(size=(self.k, self.n))
            link_tbl /= np.sum(link_tbl, axis=0)[np.newaxis, :]
        return link_tbl

    def fit(self):
        self.link_tbl, self.gmm, self.weights, xe = em_gmm_restart(
            self.features, self.link_tbl_init, restart=10,
            cov_type=self.cov_type, scaling_factor=self.scaling_factor)

    def labels(self):
        path = np.argmax(self.link_tbl, axis=0)
        return path


def em_iter_gmm_count(link_tbl, weights):
    """
    compute fractional counts for GMM.
    :param link_tbl: \log p(g_i | z_j)
    :param weights: p(z_j)
    :return: count table and log likelihood
    """
    ll_gmm = gmm_log_likelihood(link_tbl, weights)
    log_weights = np.log(weights)
    weighted_link_tbl = link_tbl + log_weights[:, np.newaxis]
    cnt_tbl = weighted_link_tbl - logsumexp(
        weighted_link_tbl, axis=0)[np.newaxis, :]
    return cnt_tbl, ll_gmm


def em_iter_update(cnt_tbl, line, cov_type='fix', scaling_factor=0.1):
    """
    given count, update parameters
    """
    # TODO: verify here with GMM implementation in SKlearn.
    # TODO: SKlearn should has column normalization.
    # TODO: But no-column normalization should be right.
    normalized_tbl = cnt_tbl - logsumexp(cnt_tbl, axis=0)[np.newaxis, :]
    weighted_tbl = np.exp(normalized_tbl)
    # weighted_tbl = np.exp(cnt_tbl)
    gmm, weights = gmm_update(line, weighted_tbl,
                              cov_type=cov_type, scaling_fix_cov=scaling_factor)
    link_tbl = gmm_assign(gmm, line)
    return link_tbl, gmm, weights


def em_gmm(line, link_tbl, weights, xe_gap=1e-8, max_iter=300,
           cov_type='fix', scaling_factor=0.1):
    """
    EM on a line of features.
    EM iterations stop if matches one of the following conditions:
      1) reach the max_iter
      2) current cross entropy / last cross entropy >= xe_gap
    :return: final link_tbl, gmm model, and cross entropy
    """
    # prepare hyper-parameters
    prev_xe = np.inf

    # start training
    c = 1
    while True:
        cnt_tbl, prb_cf = em_iter_gmm_count(link_tbl, weights)
        link_tbl, gmm, weights = em_iter_update(
            cnt_tbl, line, cov_type=cov_type, scaling_factor=scaling_factor)
        x_entropy = cross_entropy([prb_cf], [len(line)])
        eprint('iter {} cross entropy is {}, gap {}, logP(c) {}'.format(
            c, x_entropy, x_entropy / prev_xe, prb_cf))
        if time_to_stop(c, max_iter, prev_xe, x_entropy, xe_gap):
            break
        elif np.isnan(x_entropy):
            eprint('program end in iter {} caused by nan'.format(c))
            break
        else:
            prev_xe = x_entropy
            c += 1

    return link_tbl, gmm, weights, x_entropy


def em_gmm_restart(line, link_tbl_init_function, restart=10,
                   cov_type='fix', scaling_factor=0.1):
    """
    EM with random restarts.
    :return: the best link_tbl, gmm model and cross entropy after all restarts.
    """
    best_link_tbl = None
    best_gmm = None
    best_xe = np.inf
    best_weights = None

    eprint('start training...')
    for i in range(restart + 1):
        eprint('init parameters')
        gmm, weights = gmm_update(
            line, link_tbl_init_function(), cov_type=cov_type,
            scaling_fix_cov=scaling_factor)
        link_tbl = gmm_assign(gmm, line)
        if i > 0:
            eprint('random restart --- {} restarts remaining, '
                   'best cross entropy so far is {}'.format(
                restart - i, best_xe))
        link_tbl, gmm, weights, xe = em_gmm(
            line, link_tbl, weights, cov_type=cov_type,
            scaling_factor=scaling_factor)
        if xe < best_xe:
            best_xe = xe
            best_link_tbl = link_tbl
            best_gmm = gmm
            best_weights = weights
    eprint('with {} restarts, '
           'the best cross entropy is {}'.format(restart, best_xe))
    return best_link_tbl, best_gmm, best_weights, best_xe


if __name__ == '__main__':
    import doctest
    doctest.testmod()
