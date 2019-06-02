import sys

import cv2
import numpy as np

from segmentation.cutter import Cutter
from segmentation.page import Page
from segmentation.sampling import backward_gaussian_smooth, boundaries_sampler
from segmentation.utils import eprint


if __name__ == '__main__':
    im_name = sys.argv[1]
    im_path = sys.argv[2]
    feature_path = sys.argv[3]
    output_path = sys.argv[4]

    img = cv2.imread(im_path)
    page = Page(im_name, img)
    img_2d = page.img_2dim

    grid = np.load('{}.grid.npz'.format(im_name))
    row_cut = list(grid['row_cut'])
    col_cuts = list(grid['col_cuts'])

    # padding image with given step size
    step = 10
    h, w = img_2d.shape
    if w % step != 0:
        padding = np.full((h, step - (w % step)), 255, dtype='uint8')
        img_2d = np.concatenate([img_2d, padding], axis=1)
        h, w = img_2d.shape

    row_ranges = Cutter.get_ranges(row_cut)
    pixel_rows = []
    binary_img = Page.get_binary_image(img_2d)
    for upper, lower in row_ranges:
        pixel_rows.append(binary_img[upper:lower, :])

    pixel_rows = np.asarray(pixel_rows)

    features_ = np.load(feature_path)
    all_features = features_['vectors']
    init_feature_bins = features_['feature_bins']

    k = 23  # 22 unique characters and one space
    m = 4096
    max_n_col = int(w / step + 1)
    eprint('max_n_col={}'.format(max_n_col))

    mean_size = 6
    std_size = 3
    p_gaussian = map(
        lambda x: backward_gaussian_smooth(mean_size, std_size, x+1),
        range(step))
    p_penalty = 0.9

    (feature_bins, feature_zs, p_current, boundaries, col_sizes, gmm, zs_probs,
     samples_of_boundaries) = boundaries_sampler(
        pixel_rows, k, m,
        init_boundaries=col_cuts, init_feature_bins=init_feature_bins,
        max_n_col=max_n_col, step=10, window=10, snn_features=all_features,
        n_epoch=100, burn_in=50, collection_rate=0.5,
        p_gaussian=p_gaussian, p_penalty=p_penalty,
        checkpoint_dir=output_path)

    np.savez('{}/{}-sampling_results.npz'.format(sys.argv[4], im_name),
             feature_bins=feature_bins, feature_zs=feature_zs,
             p_current=p_current, boundaries=boundaries, col_sizes=col_sizes,
             gmm=gmm, zs_probs=zs_probs,
             samples_of_boundaries=samples_of_boundaries)

    new_col_cuts = []
    for row in range(len(col_sizes)):
        new_col_cuts.append(boundaries[row, :col_sizes[row]] * 10)

    cutter = Cutter(page, None, None)
    example = cutter.draw_grid(row_cut, new_col_cuts)
    cv2.imwrite('{}/{}-segmentation.png'.format(output_path, im_name), example)
