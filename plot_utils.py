from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def calc_row_col(num_ex, num_items, max_items_per_row=9):
    num_rows_per_ex = int(np.ceil(num_items / max_items_per_row))
    if num_items > max_items_per_row:
        num_col = max_items_per_row
        num_row = num_rows_per_ex * num_ex
    else:
        num_row = num_ex
        num_col = num_items

    def calc(ii, jj):
        col = jj % max_items_per_row
        row = num_rows_per_ex * ii + int(jj / max_items_per_row)

        return row, col

    return num_row, num_col, calc


def set_axis_off(axarr, num_row, num_col):
    for row in xrange(num_row):
        for col in xrange(num_col):
            if num_col > 1:
                ax = axarr[row, col]
            else:
                ax = axarr[row]
            ax.set_axis_off()
    pass


def plot_thumbnails(fname, img, axis, max_items_per_row=9):
    """Plot activation map.

    Args:
        img: [B, T, H, W, 3] or [B, H, W, D]
    """
    num_ex = img.shape[0]
    num_items = img.shape[axis]
    num_row, num_col, calc = calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            if axis == 3:
                x = img[ii, :, :, jj]
            elif axis == 1:
                x = img[ii, jj]
            if num_col > 1:
                ax = axarr[row, col]
            else:
                ax = axarr[row]
            # BGR => RGB
            x = x[:, :, [2, 1, 0]]
            ax.imshow(x)
            ax.text(0, -0.5, '[{:.2g}, {:.2g}]'.format(x.min(), x.max()),
                    color=(0, 0, 0), size=8)

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')

    pass
