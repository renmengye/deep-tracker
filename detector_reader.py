import cslab_environ

import argparse
import h5py
import os
import sys
import tensorflow as tf

import logger
from saver import Saver

import detector_model as model

log = logger.get()


def read(folder):
    log.info('Reading pretrained network from {}'.format(folder))
    saver = Saver(folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    ckpt_fname = ckpt_info['ckpt_fname']
    model_id = ckpt_info['model_id']
    m = model.get_model(model_opt)
    cnn_nlayers = len(model_opt['cnn_filter_size'])
    mlp_nlayers = 1
    timespan = 1
    weights = {}
    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    output_list = []
    for net, nlayers in zip(['cnn', 'mlp'], [cnn_nlayers, mlp_nlayers]):
        for ii in xrange(nlayers):
            for w in ['w', 'b']:
                key = '{}_{}_{}'.format(net, w, ii)
                log.info(key)
                output_list.append(key)
            if net == 'cnn':
                for tt in xrange(timespan):
                    for w in ['beta', 'gamma']:
                        key = '{}_{}_{}_{}'.format(net, ii, tt, w)
                        log.info(key)
                        output_list.append(key)

    output_var = []
    for key in output_list:
        output_var.append(m[key])

    output_var_value = sess.run(output_var)

    for key, value in zip(output_list, output_var_value):
        weights[key] = value
        log.info(key)
        log.info(value.shape)

    return weights


def save(fname, folder):
    weights = read(folder)
    h5f = h5py.File(fname, 'w')
    for key in weights:
        h5f[key] = weights[key]
    h5f.close()
    log.info('Saved weights to {}'.format(fname))

    pass


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
    args = parse_args()
    exp_folder = os.path.join(args.results, args.model_id)
    if args.output is None:
        output = os.path.join(exp_folder, 'weights.h5')
    else:
        output = args.output
    save(output, exp_folder)
