"""
Train a matching network.

Usage: python train_matching.py --help
"""
from __future__ import division

import cslab_environ

import argparse
import cv2
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
from sharded_hdf5 import ShardedFileReader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_utils as pu

import kitti as data
import detector_cnn as model


log = logger.get()


def get_dataset():
    dataset = {}
    folder = '/ais/gobi3/u/mren/data/kitti/tracking'
    dataset = ShardedFileReader(data.get_dataset(folder, split='train'))[0]
    return dataset


def plot_output(fname, x, y):
    num_ex = y.shape[0]
    num_items = 2
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=9)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    pu.set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            if jj % 2 == 0:
                axarr[row, col].imshow(cv2.resize(x[ii], (448, 128)))
            else:
                axarr[row, col].imshow(cv2.resize(y[ii], (448, 128)))

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')


def plot_output(fname, filters):
    num_ex = int(filters.shape[0]
    num_items = 8
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=9)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    pu.set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            kk = ii * num_items + jj
            axarr[row, col].imshow(filters[:, :, kk, :])

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')


def _get_batch_fn(dataset):
    def get_batch(idx):
        x_bat = dataset['images_0'][idx]
        return preprocess(x_bat)
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
    x_new = np.zeros([x.shape[0], args.height, args.width, 3])
    for ii in xrange(x.shape[0]):
        x_new[ii] = cv2.resize(x[ii], (args.width, args.height))
    return x_new.astype('float32') / 255


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', default=None)
    parser.add_argument(
        '--results', default='/ais/gobi3/u/mren/results/deep-tracker')
    parser.add_argument(
        '--output', default=None)
    parser.add_argument('--height', default=128, type=int)
    parser.add_argument('--width', default=448, type=int)

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
        'inp_height': args.height,
        'inp_width': args.width,
        'inp_depth': model_opt['inp_depth'],
        'cnn_filter_size': model_opt['cnn_filter_size'],
        'cnn_depth': model_opt['cnn_depth'],
        'cnn_pool': model_opt['cnn_pool'],
        'trained_model': os.path.join(args.results, args.model_id, 'weights.h5')
    }
    log.warning(model_opt_new['trained_model'])

    # Train loop options
    log.info('Building model')
    m = model.get_model(model_opt_new)

    log.info('Loading dataset')
    dataset = get_dataset()
    num_ex = dataset['images_0'].shape[0]

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    get_batch = _get_batch_fn(dataset)

    def run_samples():
        """Samples"""
        def _run_samples(x, fname):
            _outputs = ['y_out']
            _feed_dict = {m['x']: x}
            r = _run_model(sess, m, _outputs, _feed_dict)
            y_out = r['y_out']
            height = y_out.shape[1]
            y_out = y_out[:, :int(height * 0.9), :, :]
            y_out /= y_out.max(axis=1, keepdims=True).max(axis=2, keepdims=True)
            height = x.shape[1]
            x = x[:, :int(height * 0.9), :, :]
            plot_output(fname, x, y_out)

            pass

        idx = np.arange(num_ex)
        random = np.random.RandomState(2)
        random.shuffle(idx)
        _x = get_batch(idx[:10])
        fname_output = os.path.join(
            args.output, 'output_{}x{}.png'.format(args.height, args.width))
        _run_samples(_x, fname_output)
        pass

    run_samples()
    pass
