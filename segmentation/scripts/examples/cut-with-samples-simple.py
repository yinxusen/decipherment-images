import cv2

from segmentation.cutter import CutterSampling, CarmelArgs
from segmentation.page import Page
from segmentation.utils import *


def get_best_k_in_order(samples, k):
    n = len(samples)
    samples = np.asarray(samples)
    pos = np.sort(np.argsort(samples)[n-k:])
    return samples[pos]


if __name__ == '__main__':
    im_name = sys.argv[1]
    im_path = sys.argv[2]
    grid_path = sys.argv[3]
    sample_path = sys.argv[4]

    carmel_conf = CarmelArgs(n_row=20, r_std1=0.0001, r_std2=2, r_penalty=0.9,
                             n_col=34, c_std1=5, c_std2=4, c_penalty=0.98)

    img = cv2.imread(im_path)
    page = Page(im_name, img)

    grid = np.load(grid_path)
    row_cut = list(grid['row_cut'])

    samples = np.load(sample_path)
    samples_of_boundaries = np.asarray(samples['samples_of_boundaries'],
                                       dtype='int64')

    n_row, _ = samples_of_boundaries.shape
    h, w = page.img_2dim.shape
    step = 10
    max_n_col = (w / step) + 1
    samples_of_boundaries = samples_of_boundaries[:, :max_n_col]

    cutter = CutterSampling(page, carmel_conf, max_height=140,
                            samples_of_boundaries=samples_of_boundaries,
                            row_cut=row_cut)

    num_char_per_line = [33,34,32,32,33,33,33,34,33,35,32,30,34,35,33,31,32,31,33,30]
    col_cuts = []
    for i in range(n_row):
        col_cuts.append(get_best_k_in_order(samples_of_boundaries[i], num_char_per_line[i]))

    new_col_cuts = []
    for row in range(n_row):
        new_col_cuts.append(map(lambda x: x*step, col_cuts[row]))

    grid_image = cutter.draw_grid(row_cut, new_col_cuts)
    pieces = cutter.cut_grid(row_cut, col_cuts)
    cv2.imwrite('{}.grid.png'.format(im_name), grid_image)
