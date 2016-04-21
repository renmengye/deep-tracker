import cslab_environ

import h5py
import logger
import nnlib as nn
import numpy as np
import tensorflow as tf

log = logger.get()


def get_device_fn(device):
    """Choose device for different ops."""
    OPS_ON_CPU = set(['ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'CumMin',
                      'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense',
                      'BatchMatMul', 'Gather'])

    def _device_fn(op):
        if op.type in OPS_ON_CPU:
            return "/cpu:0"
        else:
            # Other ops will be placed on GPU if available, otherwise CPU.
            return device

    return _device_fn


def get_model(opt, device='/cpu:0'):
    model = {}
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    cnn_filter_size = opt['cnn_filter_size']
    cnn_depth = opt['cnn_depth']
    cnn_pool = opt['cnn_pool']
    trained_model = opt['trained_model']

############################
# Input definition
############################
    with tf.device(get_device_fn(device)):
        x = tf.placeholder(
            'float', [None, inp_height, inp_width, inp_depth], name='x')
        phase_train = tf.constant(False)

############################
# Feature CNN definition
############################
        cnn_filters = cnn_filter_size + [1]
        cnn_channels = [inp_depth] + cnn_depth + [1]
        cnn_nlayers = len(cnn_filter_size)
        cnn_pool = cnn_pool + [1]
        cnn_use_bn = [True] * cnn_nlayers + [False]
        # cnn_act = [tf.nn.relu] * cnn_nlayers + [tf.sigmoid]
        cnn_act = [tf.nn.relu] * cnn_nlayers + [None]

        h5f = h5py.File(trained_model, 'r')
        cnn_init_w = [{'w': h5f['cnn_w_{}'.format(ii)][:],
                       'b': h5f['cnn_b_{}'.format(ii)][:],
                       'beta_0': h5f['cnn_{}_0_beta'.format(ii)][:],
                       'gamma_0': h5f['cnn_{}_0_gamma'.format(ii)][:]}
                      for ii in xrange(cnn_nlayers)]

        cnn_init_w.append({
            'w': h5f['mlp_w_0'][:].reshape([1, 1, cnn_depth[-1], 1]),
            'b': h5f['mlp_b_0'][:].reshape([1])
        })

        cnn_frozen = [True] * (cnn_nlayers + 1)

        cnn = nn.cnn(cnn_filters, cnn_channels, cnn_pool, cnn_act,
                     cnn_use_bn, init_weights=cnn_init_w,
                     frozen=cnn_frozen,
                     model=model,
                     phase_train=phase_train, scope='cnn')

############################
# Computation graph
############################
        f = cnn(x)
        y_out = f[-1]

############################
# Computation nodes
############################
        model['x'] = x
        model['f'] = f[-1]
        model['y_out'] = y_out

    return model
