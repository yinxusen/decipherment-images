import cv2
import copy

from segmentation.cutter import Cutter, CarmelArgs
from segmentation.page import Page
from segmentation.utils import *


def apply_modification(cuts, modification):
    """
    Apply modification to column cutting results.
    :param cuts: column cuts, an array of array
    :param modification: sparse matrix, every line is (x, y, opt, [step])
      x, y: row-index and col-index of a cutting point
      opt has three options: 'a' for adding a new line after (x, y),
      'x' for deleting the current line,
      'r' for moving the current line with a step.
    :return: modified column cuts, an array of array
    """
    column_cuts = copy.deepcopy(cuts)
    for l in modification:
        try:
            cmd = l.strip().split(',')
            row_idx = int(cmd[0])
            col_idx = int(cmd[1])
            operator = cmd[2]
            if operator == 'r':
                step = int(cmd[3])
                column_cuts[row_idx][col_idx] += step
            elif operator == 'a':
                step = int(cmd[3])
                column_cuts[row_idx][col_idx] = (column_cuts[row_idx][col_idx],
                                                 ('a', step))
            elif operator == 'x':
                column_cuts[row_idx][col_idx] = (column_cuts[row_idx][col_idx],
                                                 'x')
            else:
                raise ValueError('unknown operator: {}'.format(operator))
        except Exception as e:
            eprint('error when read modification: \n{}'.format(e))
            eprint('bad line: \n{}'.format(l))

    new_col_cuts = []
    for col_cut in column_cuts:
        tmp_cut = []
        for idx in col_cut:
            if isinstance(idx, int):
                tmp_cut.append(idx)
            else:
                original_idx, operator = idx
                if operator == 'x':
                    continue
                else:
                    _, step = operator
                    tmp_cut.append(original_idx)
                    tmp_cut.append(original_idx + step)
        new_col_cuts.append(list(np.unique(sorted(tmp_cut))))
    return new_col_cuts


def apply_row_modification(cut, modification):
    """
    Apply modification to column cutting results.
    :param cut: column cuts, an array of array
    :param modification: sparse matrix, every line is (x, y, opt, [step])
      x, y: row-index and col-index of a cutting point
      opt has three options: 'a' for adding a new line after (x, y),
      'x' for deleting the current line,
      'r' for moving the current line with a step.
    :return: modified column cuts, an array of array
    """
    row_cut = copy.deepcopy(cut)
    for l in modification:
        try:
            cmd = l.strip().split(',')
            row_idx = int(cmd[0])
            operator = cmd[1]
            if operator == 'r':
                step = int(cmd[2])
                row_cut[row_idx] += step
            elif operator == 'a':
                step = int(cmd[2])
                row_cut[row_idx] = (row_cut[row_idx], ('a', step))
            elif operator == 'x':
                row_cut[row_idx] = (row_cut[row_idx], 'x')
            else:
                raise ValueError('unknown operator: {}'.format(operator))
        except Exception as e:
            eprint('error when read modification: \n{}'.format(e))
            eprint('bad line: \n{}'.format(l))

    new_row_cut = []
    for idx in row_cut:
        if isinstance(idx, int):
            new_row_cut.append(idx)
        else:
            original_idx, operator = idx
            if operator == 'x':
                continue
            else:
                _, step = operator
                new_row_cut.append(original_idx)
                new_row_cut.append(original_idx + step)
    return list(np.unique(sorted(new_row_cut)))


if __name__ == '__main__':
    im_name = sys.argv[1]
    im_path = sys.argv[2]
    carmel_conf = CarmelArgs(n_row=20, r_std1=0.0001, r_std2=2, r_penalty=0.9,
                             n_col=34, c_std1=5, c_std2=10, c_penalty=0.98)

    img = cv2.imread(im_path)
    page = Page(im_name, img)
    cutter = Cutter(page, carmel_conf, max_height=140)

    try:
        grid = np.load('{}.grid.npz'.format(im_name))
        row_cut = list(grid['row_cut'])
        col_cuts = list(grid['col_cuts'])
        new_cut = False
    except IOError as e:
        eprint('load grid error: {}'.format(e))
        row_cut, col_cuts = cutter.get_grid()
        new_cut = True

    grid_image = cutter.draw_grid(row_cut, col_cuts)
    pieces = cutter.cut_grid(row_cut, col_cuts)
    cv2.imwrite('{}.grid.png'.format(im_name), grid_image)

    try:
        with open('row-modification-{}.txt'.format(im_name), 'r') as f_rm:
            row_lines = f_rm.readlines()
            row_cut = apply_row_modification(row_cut, row_lines)
    except EnvironmentError as e:
        eprint('no row modification')

    try:
        with open('modification-{}.txt'.format(im_name), 'r') as f_m:
            lines = f_m.readlines()
            col_cuts = apply_modification(col_cuts, lines)
            grid_image = cutter.draw_grid(row_cut, col_cuts)
            pieces = cutter.cut_grid(row_cut, col_cuts)
            cv2.imwrite('{}.grid.modified.png'.format(im_name), grid_image)
            for idx, piece in pieces:
                f_img = '-'.join(map(lambda i: '{:05d}'.format(i), idx))
                success = cv2.imwrite(
                    "{}-piece-{}.png".format(im_name, f_img), piece)
                if not success:
                    eprint("Save image {}-piece-{}.png failed".format(
                        im_name, f_img))
    except EnvironmentError as e:
        eprint('no column modification')

    if new_cut:
        np.savez('{}.grid.npz'.format(im_name),
                 row_cut=row_cut, col_cuts=col_cuts,
                 carmel_conf=carmel_conf, raw_image=page.img_3dim)
