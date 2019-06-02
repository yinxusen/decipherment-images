import cv2

from segmentation.cutter import CutterSampling, CarmelArgs
from segmentation.page import Page
from segmentation.utils import *
from segmentation.vectorize import pad_image


if __name__ == '__main__':
    im_name = sys.argv[1]
    im_path = sys.argv[2]
    grid_path = sys.argv[3]
    sample_path = sys.argv[4]
    output_path = sys.argv[5]

    carmel_conf = CarmelArgs(n_row=20, r_std1=0.0001, r_std2=2, r_penalty=0.9,
                             n_col=38, c_std1=2, c_std2=2, c_penalty=0.98)

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

    _, col_cuts = cutter.get_grid()

    new_col_cuts = []
    for row in range(n_row):
        new_col_cuts.append(map(lambda x: x*step, col_cuts[row]))

    grid_image = cutter.draw_grid(row_cut, new_col_cuts)
    cv2.imwrite('{}.grid.png'.format(im_name), grid_image)

    pieces = []
    row_ranges = zip([0]+list(row_cut[:-1]), list(row_cut))
    for i, (upper, lower) in enumerate(row_ranges):
        col_cut = new_col_cuts[i]
        col_ranges = zip([0]+list(col_cut[:-1]), list(col_cut))
        for j, (c_upper, c_lower) in enumerate(col_ranges):
            pieces.append(
                pad_image(page.img_2dim[upper:lower, c_upper:c_lower]))

    for i, f in enumerate(pieces):
        cv2.imwrite('{}/{:03d}.png'.format(output_path, i), f)

    np.savez('{}.pieces.npz'.format(im_name), features=pieces)
