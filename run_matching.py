import cslab_environ

from saver import Saver
import logger
import numpy as np
import sys
import tensorflow as tf
import plot_utils as pu

import matching_model as model
import matching_data as data

log = logger.get()


def get_model(opt, device='/cpu:0'):
    return model.get_model(opt, device)


def get_dataset(opt):
    dataset = {}
    folder = '/ais/gobi3/u/mren/data/kitti/tracking/training'
    dataset = data.get_dataset(folder, opt, split=None, seqs=[20])

    return dataset


def get_batch_fn(dataset):
    def get_batch(idx):
        x1_bat = dataset['images_0'][idx]
        x2_bat = dataset['images_1'][idx]
        y_bat = dataset['labels'][idx]
        x1_bat, x2_bat, y_bat = preprocess(x1_bat, x2_bat, y_bat)

        return x1_bat, x2_bat, y_bat

    return get_batch


def plot_output(fname, x1, x2, y_gt, y_out):
    num_ex = y_out.shape[0]
    num_items = 2
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=9)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    pu.set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            if jj == 0:
                axarr[row, col].imshow(x1[ii])
            else:
                axarr[row, col].imshow(x2[ii])

            axarr[row, col].text(0, 0, '{:.2f} {:.2f}'.format(
                y_gt[ii], y_out[ii]),
                color=(0, 0, 0), size=8)

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')


def preprocess(x1, x2, y):
    """Preprocess training data."""
    return (x1.astype('float32') / 255,
            x2.astype('float32') / 255,
            y.astype('float32'))


if __name__ == '__main__':
    restore_folder = sys.argv[1]
    saver = Saver(restore_folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    data_opt = ckpt_info['data_opt']
    ckpt_fname = ckpt_info['ckpt_fname']
    step = ckpt_info['step']
    model_id = ckpt_info['model_id']

    log.info('Building model')
    m = get_model(model_opt)

    log.info('Loading dataset')
    dataset = get_dataset(data_opt)

    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    idx = np.arange(10)
    get_batch = get_batch_fn(dataset)
    x1, x2, y_gt = get_batch(idx)

    y_out = sess.run(m['y_out'],
                     feed_dict={
                         m['x1']: x1, m['x2']: x2, m['phase_train']: False})
    print y_out
    print y

    plot_output('/u/mren/test_matching.png', x1, x2, y_gt, y_out)
