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

import patch_data as data
import detector_model as model


log = logger.get()


def get_model(opt, device='/cpu:0'):
    return model.get_model(opt, device)


def get_dataset(opt):
    dataset = {}
    folder = '/ais/gobi3/u/mren/data/kitti/tracking/training'
    dataset['train'] = data.get_dataset(
        folder, opt, split='train', usage='detect')
    dataset['valid'] = data.get_dataset(
        folder, opt, split='valid', usage='detect')

    return dataset


def plot_output(fname, x, y_gt, y_out):
    num_ex = int(y_out.shape[0] / 8)
    num_items = 8
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=9)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    pu.set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            kk = ii * num_items + jj
            axarr[row, col].imshow(x[kk])
            axarr[row, col].text(0, 0, '{:.2f} {:.2f}'.format(
                y_gt[kk], y_out[kk]),
                color=(0, 0, 0), size=8)

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')


def _get_batch_fn(dataset):
    def get_batch(idx):
        x_bat = dataset['images'][idx]
        y_bat = dataset['labels'][idx]
        x_bat, y_bat = preprocess(x_bat, y_bat)

        return x_bat, y_bat

    return get_batch


def _run_model(sess, m, names, feed_dict):
    symbol_list = [m[r] for r in names]
    results = sess.run(symbol_list, feed_dict=feed_dict)
    results_dict = {}
    for rr, name in zip(results, names):
        results_dict[name] = rr

    return results_dict


def preprocess(x, y):
    """Preprocess training data."""
    return (x.astype('float32') / 255,
            y.astype('float32'))


def _add_dataset_args(parser):
    kPatchHeight = 48
    kPatchWidth = 48
    kPadding = 0.2
    kPaddingNoise = 0.2
    kCenterNoise = 0.2
    kNumExPos = 50
    kNumExNeg = 50
    parser.add_argument('--patch_height', default=kPatchHeight, type=int)
    parser.add_argument('--patch_width', default=kPatchWidth, type=int)
    parser.add_argument('--padding', default=kPadding, type=float)
    parser.add_argument('--padding_noise', default=kPaddingNoise, type=float)
    parser.add_argument('--center_noise', default=kCenterNoise, type=float)
    parser.add_argument('--num_ex_pos', default=kNumExPos, type=int)
    parser.add_argument('--num_ex_neg', default=kNumExNeg, type=int)

    pass


def _add_model_args(parser):
    kCnnFilterSize = '3,3,3,3,3,3,3,3'
    kCnnDepth = '16,16,32,32,64,64,96,96'
    kCnnPool = '1,2,1,2,1,2,1,2'
    kMlpDims = '1'
    kMlpDropout = 0
    kWeightDecay = 5e-5
    kBaseLearnRate = 1e-3
    kStepsPerLearnRateDecay = 5000
    kLearnRateDecay = 0.96

    parser.add_argument('--cnn_filter_size', default=kCnnFilterSize)
    parser.add_argument('--cnn_depth', default=kCnnDepth)
    parser.add_argument('--cnn_pool', default=kCnnPool)
    parser.add_argument('--mlp_dims', default=kMlpDims)
    parser.add_argument('--mlp_dropout', default=kMlpDropout, type=float)
    parser.add_argument('--weight_decay', default=kWeightDecay, type=float)
    parser.add_argument('--base_learn_rate', default=kBaseLearnRate)
    parser.add_argument('--learn_rate_decay',
                        default=kLearnRateDecay, type=float)
    parser.add_argument('--steps_per_learn_rate_decay',
                        default=kStepsPerLearnRateDecay, type=int)

    pass


def _add_training_args(parser):
    kNumSteps = 500000
    kStepsPerCkpt = 1000
    kStepsPerValid = 250
    kStepsPerTrainval = 100
    kStepsPerPlot = 100
    kNumSamplesPlot = 20
    kStepsPerLog = 20
    kBatchSize = 64

    # Training options
    parser.add_argument('--num_steps', default=kNumSteps, type=int)
    parser.add_argument('--steps_per_ckpt', default=kStepsPerCkpt, type=int)
    parser.add_argument('--steps_per_valid', default=kStepsPerValid, type=int)
    parser.add_argument('--steps_per_trainval',
                        default=kStepsPerTrainval, type=int)
    parser.add_argument('--steps_per_plot', default=kStepsPerPlot, type=int)
    parser.add_argument('--num_samples_plot',
                        default=kNumSamplesPlot, type=int)
    parser.add_argument('--steps_per_log', default=kStepsPerLog, type=int)
    parser.add_argument('--batch_size', default=kBatchSize, type=int)
    parser.add_argument('--results', default='../results')
    parser.add_argument('--logs', default='../results')
    parser.add_argument('--localhost', default='localhost')
    parser.add_argument('--restore', default=None)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--save_ckpt', action='store_true')

    pass


def _make_model_opt(args):
    """Convert command-line arguments into model opt dict."""
    cnn_fsize_list = args.cnn_filter_size.split(',')
    cnn_fsize_list = [int(fsize) for fsize in cnn_fsize_list]
    cnn_depth_list = args.cnn_depth.split(',')
    cnn_depth_list = [int(depth) for depth in cnn_depth_list]
    cnn_pool_list = args.cnn_pool.split(',')
    cnn_pool_list = [int(pool) for pool in cnn_pool_list]
    mlp_dim_list = args.mlp_dims.split(',')
    mlp_dim_list = [int(dim) for dim in mlp_dim_list]
    rnd_vflip = False
    rnd_hflip = True
    rnd_transpose = True
    rnd_colour = False

    model_opt = {
        'inp_height': args.patch_height,
        'inp_width': args.patch_width,
        'inp_depth': 3,
        'padding': args.padding,
        'weight_decay': args.weight_decay,
        'base_learn_rate': args.base_learn_rate,
        'learn_rate_decay': args.learn_rate_decay,
        'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
        'cnn_filter_size': cnn_fsize_list,
        'cnn_pool': cnn_pool_list,
        'cnn_depth': cnn_depth_list,
        'mlp_dims': mlp_dim_list,
        'mlp_dropout': args.mlp_dropout,
        'rnd_hflip': rnd_hflip,
        'rnd_vflip': rnd_vflip,
        'rnd_transpose': rnd_transpose,
        'rnd_colour': rnd_colour
    }

    return model_opt


def _make_data_opt(args):
    """Make command-line arguments into data opt dict."""
    data_opt = {
        'patch_height': args.patch_height,
        'patch_width': args.patch_width,
        'center_noise': args.center_noise,
        'padding_noise': args.padding_noise,
        'padding_mean': args.padding,
        'num_ex_pos': args.num_ex_pos,
        'num_ex_neg': args.num_ex_neg,
        'shuffle': True
    }

    return data_opt


def _make_train_opt(args):
    """Train opt"""
    train_opt = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt,
        'steps_per_valid': args.steps_per_valid,
        'steps_per_trainval': args.steps_per_trainval,
        'num_samples_plot': args.num_samples_plot,
        'steps_per_plot': args.steps_per_plot,
        'steps_per_log': args.steps_per_log,
        'results': args.results,
        'restore': args.restore,
        'save_ckpt': args.save_ckpt,
        'logs': args.logs,
        'gpu': args.gpu,
        'localhost': args.localhost,
    }

    return train_opt


def _parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(
        description='Recurrent Instance Segmentation + Attention')

    _add_dataset_args(parser)
    _add_model_args(parser)
    _add_training_args(parser)

    args = parser.parse_args()

    return args


def _get_model_id(task_name):
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)

    return model_id


def _get_ts_loggers(model_opt, restore_step=0):
    loggers = {}
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['acc'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'acc.csv'), ['train', 'valid'],
        name='Accuracy',
        buffer_size=1,
        restore_step=restore_step)
    loggers['step_time'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'step_time.csv'), 'step time',
        name='Step Time',
        buffer_size=1,
        restore_step=restore_step)

    return loggers


def _get_plot_loggers(model_opt, train_opt):
    samples = {}
    _ssets = ['train', 'valid']
    for _set in _ssets:
        labels = ['output']
        for name in labels:
            key = '{}_{}'.format(name, _set)
            samples[key] = LazyRegisterer(
                os.path.join(logs_folder, '{}.png'.format(key)),
                'image', 'Samples {} {}'.format(name, _set))

    return samples


def _register_raw_logs(log_manager, log, model_opt, saver):
    log_manager.register(log.filename, 'plain', 'Raw logs')
    cmd_fname = os.path.join(logs_folder, 'cmd.log')
    with open(cmd_fname, 'w') as f:
        f.write(' '.join(sys.argv))
    log_manager.register(cmd_fname, 'plain', 'Command-line arguments')
    model_opt_fname = os.path.join(logs_folder, 'model_opt.yaml')
    saver.save_opt(model_opt_fname, model_opt)
    log_manager.register(model_opt_fname, 'plain', 'Model hyperparameters')

    pass


def _get_num_batch_valid():
    return 10


if __name__ == '__main__':
    # Command-line arguments
    args = _parse_args()
    tf.set_random_seed(1234)
    saver = None
    train_opt = _make_train_opt(args)
    model_opt = _make_model_opt(args)
    data_opt = _make_data_opt(args)

    # Restore previously saved checkpoints.
    if train_opt['restore']:
        saver = Saver(train_opt['restore'])
        ckpt_info = saver.get_ckpt_info()
        model_opt = ckpt_info['model_opt']
        data_opt = ckpt_info['data_opt']
        ckpt_fname = ckpt_info['ckpt_fname']
        step = ckpt_info['step']
        model_id = ckpt_info['model_id']
        exp_folder = train_opt['restore']
    else:
        model_id = _get_model_id('matching')
        step = 0
        exp_folder = os.path.join(train_opt['results'], model_id)
        saver = Saver(exp_folder, model_opt=model_opt, data_opt=data_opt)

    if not train_opt['save_ckpt']:
        log.warning(
            'Checkpoints saving is turned off. Use -save_ckpt flag to save.')

    # Logger
    if train_opt['logs']:
        logs_folder = train_opt['logs']
        logs_folder = os.path.join(logs_folder, model_id)
        log = logger.get(os.path.join(logs_folder, 'raw.log'))
    else:
        log = logger.get()

    # Log arguments
    log.log_args()

    # Set device
    if train_opt['gpu'] >= 0:
        device = '/gpu:{}'.format(train_opt['gpu'])
    else:
        device = '/cpu:0'

    # Train loop options
    log.info('Building model')
    m = get_model(model_opt, device=device)

    log.info('Loading dataset')
    dataset = get_dataset(data_opt)

    sess = tf.Session()

    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series loggers
    loggers = {}
    if train_opt['logs']:
        log_manager = LogManager(logs_folder)
        loggers = _get_ts_loggers(model_opt)
        _register_raw_logs(log_manager, log, model_opt, saver)
        samples = _get_plot_loggers(model_opt, train_opt)
        _log_url = 'http://{}/deep-dashboard?id={}'.format(
            train_opt['localhost'], model_id)
        log.info('Visualization can be viewed at: {}'.format(_log_url))

    batch_size = args.batch_size
    log.info('Batch size: {}'.format(batch_size))
    num_ex_train = dataset['train']['labels'].shape[0]
    get_batch_train = _get_batch_fn(dataset['train'])
    log.info('Number of training examples: {}'.format(num_ex_train))
    num_ex_valid = dataset['valid']['labels'].shape[0]
    get_batch_valid = _get_batch_fn(dataset['valid'])
    log.info('Number of validation examples: {}'.format(num_ex_valid))

    def run_samples():
        """Samples"""
        def _run_samples(x, y_gt, fname):
            _outputs = ['y_out']
            _feed_dict = {m['x']: x, m['phase_train']: False}
            r = _run_model(sess, m, _outputs, _feed_dict)
            plot_output(fname, x, y_gt, r['y_out'])

            pass

        # Plot some samples.
        _ssets = ['train', 'valid']
        for _set in _ssets:
            _is_train = _set == 'train'
            _get_batch = get_batch_train if _is_train else get_batch_valid
            _num_ex = num_ex_train if _is_train else num_ex_valid
            log.info('Plotting {} samples'.format(_set))
            _x, _y = _get_batch(
                np.arange(min(_num_ex, args.num_samples_plot)))

            labels = ['output']
            fname_output = samples['output_{}'.format(_set)].get_fname()
            _run_samples(_x, _y, fname_output)

            if not samples['output_{}'.format(_set)].is_registered():
                for _name in labels:
                    samples['{}_{}'.format(_name, _set)].register()
        pass

    def get_outputs_valid():
        _outputs = ['loss', 'acc']

        return _outputs

    def get_outputs_trainval():
        _outputs = ['loss', 'acc', 'learn_rate']

        return _outputs

    def run_stats(step, num_batch, batch_iter, outputs, write_log, phase_train):
        """Validation"""
        nvalid = num_batch * batch_size
        r = {}

        for bb in xrange(num_batch):
            _x, _y = batch_iter.next()
            _feed_dict = {m['x']: _x, m['phase_train']: phase_train,
                          m['y_gt']: _y, }
            _r = _run_model(sess, m, outputs, _feed_dict)
            bat_sz = _x.shape[0]

            for key in _r.iterkeys():
                if key in r:
                    r[key] += _r[key] * bat_sz / nvalid
                else:
                    r[key] = _r[key] * bat_sz / nvalid

        log.info('{:d} loss {:.4f}'.format(step, r['loss']))
        write_log(step, loggers, r)

        pass

    def write_log_valid(step, loggers, r):
        loggers['loss'].add(step, ['', r['loss']])
        loggers['acc'].add(step, ['', r['acc']])

        pass

    def write_log_trainval(step, loggers, r):
        loggers['loss'].add(step, [r['loss'], ''])
        loggers['acc'].add(step, [r['acc'], ''])

        pass

    def train_step(step, x, y):
        """Train step"""

        _start_time = time.time()
        _outputs = ['loss', 'train_step']
        _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y}
        r = _run_model(sess, m, _outputs, _feed_dict)
        _step_time = (time.time() - _start_time) * 1000

        # Print statistics.
        if step % train_opt['steps_per_log'] == 0:
            log.info('{:d} loss {:.4f} t {:.2f}ms'.format(step, r['loss'],
                                                          _step_time))
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['step_time'].add(step, _step_time)

        pass

    def train_loop(step=0):
        """Train loop"""
        batch_iter_valid = BatchIterator(num_ex_valid,
                                         batch_size=batch_size,
                                         get_fn=get_batch_valid,
                                         cycle=True,
                                         progress_bar=False)
        outputs_valid = get_outputs_valid()
        num_batch_valid = _get_num_batch_valid()
        batch_iter_trainval = BatchIterator(num_ex_train,
                                            batch_size=batch_size,
                                            get_fn=get_batch_train,
                                            cycle=True,
                                            progress_bar=False)
        outputs_trainval = get_outputs_trainval()

        for _x, _y in BatchIterator(num_ex_train,
                                          batch_size=batch_size,
                                          get_fn=get_batch_train,
                                          cycle=True,
                                          progress_bar=False):
            # Run validation stats
            if step % train_opt['steps_per_valid'] == 0:
                log.info('Running validation')
                run_stats(step, num_batch_valid, batch_iter_valid,
                          outputs_valid, write_log_valid, False)
                pass

            # Train stats
            if step % train_opt['steps_per_trainval'] == 0:
                log.info('Running train validation')
                run_stats(step, num_batch_valid, batch_iter_trainval,
                          outputs_trainval, write_log_trainval, True)
                pass

            # Plot samples
            if step % train_opt['steps_per_plot'] == 0:
                run_samples()
                pass

            # Train step
            train_step(step, _x, _y)

            # Model ID reminder
            if step % (10 * train_opt['steps_per_log']) == 0:
                log.info('model id {}'.format(model_id))
                pass

            # Save model
            if args.save_ckpt and step % train_opt['steps_per_ckpt'] == 0:
                saver.save(sess, global_step=step)
                pass

            step += 1

            # Termination
            if step > train_opt['num_steps']:
                break

        pass

    train_loop(step=step)

    sess.close()
    for logger in loggers.itervalues():
        logger.close()

    pass
