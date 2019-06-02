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

import math

import numpy as np
from scipy.misc import logsumexp

from decipher.gmm import GMM
from decipher.utils import gmm_update, gmm_assign
from segmentation.utils import eprint

"""
Refer to the papers:

Bayesian Methods for Hidden Markov Models: Recursive Computing in the 21st Century
A segmental framework for fully-unsupervised large-vocabulary speech recognition
"""


def choose(probs):
    selected = np.random.random()
    i = 0
    while i < len(probs):
        selected -= probs[i]
        if selected <= 0:
            return i
        i += 1
    return i


def estimate_gmm(feature_bins, feature_zs, col_sizes, k, ignore=-1,
                 cov_type='spherical', scaling_factor=1.0):
    n_row = len(feature_bins)
    features = []
    weighted_tbl = []
    for row in range(n_row):
        if row != ignore:
            w_tbl = np.zeros((k, col_sizes[row]))
            features += list(feature_bins[row, :col_sizes[row]])
            w_tbl[feature_zs[row, :col_sizes[row]], range(col_sizes[row])] = 1.0
            weighted_tbl.append(w_tbl)
    features = np.asarray(features)
    weighted_tbl = np.concatenate(weighted_tbl, axis=1)
    gmm, weights = gmm_update(features, weighted_tbl,
                              cov_type=cov_type, scaling_fix_cov=scaling_factor)
    return gmm, np.log(weights)


def boundaries_sampler(pixel_rows,
                       k, m,
                       init_boundaries,
                       init_feature_bins,
                       max_n_col,
                       step, window,
                       snn_features,
                       p_gaussian, p_penalty,
                       t0=3, t1=1,
                       n_epoch=5000,
                       burn_in=1000,
                       collection_rate=0.05,
                       checkpoint_dir=None,
                       checkpoint_epoch_step=10):
    """
    block-wise sample boundaries for rows of cipher images.
    cipher images are first cut into rows, then each row is cut into frames.
    frames between two boundaries are treated as one image character.

    :param pixel_rows: array of pixel rows [np.array((h, w))]
    :param k: number of total clusters
    :param m: dimension of features
    :param init_boundaries: initial boundaries to cut pixel rows
    :param init_feature_bins: initial features generated from initial boundaries
    :param max_n_col: maximum number of characters in each row
    :param step: number of pixel for each frame
    :param window: maximum steps looking back for finding boundaries
      at each frame
    :param snn_features: pre-computed features from snn encoders,
      see vectorize.py
    :param mu0: base mean of Gaussian distributions
    :param sigma0: base variance of Gaussian distributions
    :param sigma1: variance as regularizer
    :param t0: high temperature for simulated annealing
    :param t1: low temperature
    :param n_epoch: number of epochs for sampling
    :return: boundaries of pixel rows
    """

    n_row = len(pixel_rows)
    eprint('n_row={}'.format(n_row))

    feature_bins = np.zeros((n_row, max_n_col, m))
    boundaries = np.zeros((n_row, max_n_col), dtype='int64')
    col_sizes = np.zeros(n_row, dtype='int64')
    feature_zs = np.full((n_row, max_n_col), -1, dtype='int64')

    # GMM init
    cluster = GMM(
        features=np.concatenate(init_feature_bins, axis=0),
        n_clusters=k, params_init="kmeans_init",
        cov_type='spherical', scaling_factor=1.0)

    cluster.fit()

    init_feature_zs = cluster.labels()
    eprint('init_feature_zs: {}'.format(init_feature_zs))

    i = 0
    for row in range(n_row):
        col_sizes[row] = len(init_feature_bins[row])
        feature_bins[row, :col_sizes[row], :] = init_feature_bins[row]
        boundaries[row, :col_sizes[row]] = init_boundaries[row]
        feature_zs[row, :col_sizes[row]] = init_feature_zs[i:i+col_sizes[row]]
        i += col_sizes[row]

    # init gmm
    gmm, zs_probs = estimate_gmm(feature_bins, feature_zs, col_sizes, k,
                                 ignore=-1,
                                 cov_type='spherical', scaling_factor=1.0)
    # compute p(current)
    probs_of_rows = np.full(n_row, -np.inf)
    p_current = 0
    for row in range(n_row):
        gaussian_probs = gmm_assign(gmm, feature_bins[row, :col_sizes[row]])
        p_row = gaussian_probs + zs_probs[:, np.newaxis]
        probs_of_rows[row] = np.sum(
            p_row[feature_zs[row, :col_sizes[row]], range(col_sizes[row])])
        p_current += probs_of_rows[row]

    samples_of_boundaries = np.zeros((n_row, max_n_col))

    # sampling body
    t_step = 1.0 * (t0 - t1) / burn_in
    for epoch in range(n_epoch):
        ti = max(t0 - epoch * t_step, t1)
        if epoch < burn_in:
            eprint('burn-in epoch {}'.format(epoch))
        else:
            eprint('sampling epoch {}'.format(epoch))
        total_elements = np.sum(col_sizes)
        eprint('epoch-log_p-temperature: {},{},{}'.format(
            epoch, p_current / total_elements, ti))
        eprint(col_sizes)
        for row in range(n_row):
            # remove row_j from GMM
            feature_bins[row, :, :] = 0
            p_current -= probs_of_rows[row]

            # evaluate GMM
            gmm, zs_probs = estimate_gmm(
                feature_bins, feature_zs, col_sizes, k,
                ignore=row, cov_type='spherical', scaling_factor=1.0)

            # re-sample character boundaries
            alpha_tbl, all_probs, n_frames = iter_forward_filtering(
                pixel_rows[row], gmm, step, window,
                snn_features[row], zs_probs)

            new_boundaries, new_features = iter_backward_sampling(
                pixel_rows[row], step, window, alpha_tbl, all_probs,
                snn_features[row], ti, p_gaussian, p_penalty)

            # update row size
            col_sizes[row] = len(new_boundaries)
            feature_bins[row, :col_sizes[row], :] = new_features
            boundaries[row, :col_sizes[row]] = new_boundaries

            # re-sample cluster assignments
            new_zs = iter_cluster_sampling(new_features, gmm, zs_probs, ti)
            feature_zs[row, :col_sizes[row]] = new_zs

            # re-count p_row
            gaussian_probs = gmm_assign(gmm, new_features)
            p_row = gaussian_probs + zs_probs[:, np.newaxis]
            probs_of_rows[row] = np.sum(
                p_row[feature_zs[row, :col_sizes[row]], range(col_sizes[row])])
            p_current += probs_of_rows[row]

            for i in range(col_sizes[row]):
                if epoch >= burn_in and np.random.random() <= collection_rate:
                    samples_of_boundaries[row, new_boundaries[i]] += 1

        if (checkpoint_dir is not None and epoch >= burn_in
                and epoch % checkpoint_epoch_step == 0):
            eprint('save checkpoint of epoch-{} to {}'.format(
                epoch, checkpoint_dir))
            np.savez('{}/checkpoint.npz'.format(checkpoint_dir),
                     epoch=epoch,
                     feature_bins=feature_bins,
                     feature_zs=feature_zs,
                     p_current=p_current,
                     boundaries=boundaries,
                     col_sizes=col_sizes,
                     gmm=gmm,
                     zs_probs=zs_probs,
                     samples_of_boundaries=samples_of_boundaries,
                     probs_of_rows=probs_of_rows)

    return (feature_bins, feature_zs, p_current, boundaries, col_sizes,
            gmm, zs_probs, samples_of_boundaries)


def iter_forward_filtering(pixel_row, gmm, step, window,
                           all_features, zs_probs):
    h, w = pixel_row.shape
    n_frames = w / step

    alpha_tbl = np.full(n_frames, -np.inf)
    alpha_tbl[0] = 0.0
    all_probs = np.full((n_frames, window), -np.inf)
    for i in range(n_frames):
        features = all_features[i]
        gaussian_probs = gmm_assign(gmm, features)
        gaussian_probs -= logsumexp(gaussian_probs, axis=0)[np.newaxis, :]
        resp = logsumexp(gaussian_probs + zs_probs[:, np.newaxis], axis=0)
        # the larger the group of pixels, the heavier the penalty
        weighted_resp = resp * (np.asarray(range(len(resp))[::-1]) + 1.)
        all_probs[i, :len(resp)] = weighted_resp
        alpha_tbl[i] = logsumexp(
            all_probs[i, :len(resp)] + alpha_tbl[max(0, i-window+1):i+1])
    return alpha_tbl, all_probs, n_frames


def backward_gaussian_smooth(mean, std, x):
    p = ((1 / (std * math.sqrt(2 * 3.1415))) *
         math.exp(-((x - mean) ** 2) / (2 * std ** 2)))
    return p


def iter_backward_sampling(pixel_row, step, window, alpha_tbl,
                           all_probs, all_features, ti,
                           p_gaussian, p_penalty):
    h, w = pixel_row.shape
    n_frames = w / step
    sampled = [n_frames]
    features = []
    t = n_frames - 1
    while t > 0:
        n_black_pixels = np.sum(
            pixel_row[:, max(0, t-window+1)*10:(t+1)*10:step], axis=0)
        penalty_probs = n_black_pixels * np.log(1-p_penalty) + np.log(p_penalty)
        size_probs = np.asarray(p_gaussian[:min(window, t+1)][::-1])
        probs = (alpha_tbl[max(0, t-window+1):t+1]
                 + all_probs[t, :min(window, t+1)] + penalty_probs + size_probs) / ti

        probs -= logsumexp(probs)
        sampled_back_position = choose(np.exp(probs))
        sampled_size = len(probs) - sampled_back_position
        features.append(all_features[t][sampled_back_position])
        t = max(0, t - sampled_size)
        sampled.append(t+1)
    features.append(all_features[t][0])
    return np.asarray(sampled[::-1]), np.asarray(features[::-1])


def iter_cluster_sampling(features, gmm, zs_probs, ti):
    gaussian_probs = gmm_assign(gmm, features)
    resp = (zs_probs[:, np.newaxis] + gaussian_probs) / ti
    resp -= logsumexp(resp, axis=0)
    sampled_zs = map(lambda probs: choose(np.exp(probs)), resp.T)
    return sampled_zs
