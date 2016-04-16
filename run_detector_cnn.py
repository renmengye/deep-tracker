"""
Train a matching network.

Usage: python train_matching.py --help
"""
from __future__ import division

import cslab_environ

import argparse
import datetime
import h5py
import numpy as np
import os
import pickle as pkl
import sys
import tensorflow as tf
import time

import logger
from batch_iter import BatchIterator
from lazy_registerer import LazyRegisterer
from log_manager import LogManager
from saver import Saver
from time_series_logger import TimeSeriesLogger

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_utils as pu

import kitti as data
import detector_cnn as model


log = logger.get()


def get_dataset():
    dataset = {}
    folder = '/ais/gobi3/u/mren/data/kitti/tracking/training'
    dataset['train'] = data.get_dataset(folder, split='train')
    dataset['valid'] = data.get_dataset(folder, split='valid')
    return dataset


def plot_output(fname, x):
    num_ex = y_out.shape[0]
    num_items = 1
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=9)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    pu.set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            axarr[row, col].imshow(x[ii])

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')


def _get_batch_fn(dataset):
    def get_batch(idx):
        x_bat = dataset['images'][idx]
        return x_bat
    return get_batch


def _run_model(sess, m, names, feed_dict):
    symbol_list = [m[r] for r in names]
    results = sess.run(symbol_list, feed_dict=feed_dict)
    results_dict = {}
    for rr, name in zip(results, names):
        results_dict[name] = rr

    return results_dict


def preprocess(x):
    """Preprocess training data."""
    x = cv2.resize(x, (448, 128))
    return x.astype('float32') / 255


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', default=None)
    parser.add_argument(
        '--results', default='/ais/gobi3/u/mren/results/deep-tracker')
    parser.add_argument(
        '--output', default=None)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()
    tf.set_random_seed(1234)
    saver = None
    model_folder = os.path.join(args.results, args.model_id)
    saver = Saver(model_folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    model_opt_new = {
        'inp_height': 128,
        'inp_width': 448,
        'inp_depth': model_opt['inp_depth'],
        'cnn_filter_size': model_opt['cnn_filter_size'],
        'cnn_depth': model_opt['cnn_depth'],
        'cnn_pool': model_opt['cnn_pool'],
        'trained_model': os.path.join(args.results, args.model_id, 'weights.h5')
    }

    # Train loop options
    log.info('Building model')
    m = model.get_model(model_opt_new)

    log.info('Loading dataset')
    dataset = get_dataset()

    sess = tf.Session()
    batch_size = 10
    log.info('Batch size: {}'.format(batch_size))
    num_ex_train = dataset['train']['labels'].shape[0]
    get_batch_train = _get_batch_fn(dataset['train'])
    log.info('Number of training examples: {}'.format(num_ex_train))
    num_ex_valid = dataset['valid']['labels'].shape[0]
    get_batch_valid = _get_batch_fn(dataset['valid'])
    log.info('Number of validation examples: {}'.format(num_ex_valid))

    def run_samples():
        """Samples"""
        def _run_samples(x, fname):
            _outputs = ['y_out']
            _feed_dict = {m['x']: x}
            r = _run_model(sess, m, _outputs, _feed_dict)
            plot_output(fname, r['y_out'])

            pass

        # Plot some samples.
        _ssets = ['train', 'valid']
        for _set in _ssets:
            _is_train = _set == 'train'
            _get_batch = get_batch_train if _is_train else get_batch_valid
            _num_ex = num_ex_train if _is_train else num_ex_valid
            log.info('Plotting {} samples'.format(_set))
            _x = _get_batch(np.arange(min(_num_ex, args.num_samples_plot)))
            fname_output = os.path.join(args.output, _ssets + '_output.png')
            _run_samples(_x, fname_output)
        pass

    run_samples()

    pass
