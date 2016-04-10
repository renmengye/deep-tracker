import cslab_environ

import nnlib as nn
import numpy as np
import tensorflow as tf


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


def f_bce(y_out, y_gt):
    """Binary cross entropy."""
    eps = 1e-5
    return -y_gt * tf.log(y_out + eps) - (1 - y_gt) * tf.log(1 - y_out + eps)


def get_model(opt, device='/cpu:0'):
    model = {}
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    cnn_filter_size = opt['cnn_filter_size']
    cnn_depth = opt['cnn_depth']
    cnn_pool = opt['cnn_pool']
    mlp_dims = opt['mlp_dims']
    wd = opt['weight_decay']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay = opt['learn_rate_decay']
    steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']

############################
# Input definition
############################
    with tf.device(get_device_fn(device)):
        x1 = tf.placeholder(
            'float', [None, inp_height, inp_width, inp_depth], name='x1')
        x2 = tf.placeholder(
            'float', [None, inp_height, inp_width, inp_depth], name='x2')
        phase_train = tf.placeholder('bool', name='phase_train')
        y_gt = tf.placeholder('float', [None], name='y_gt')
        global_step = tf.Variable(0.0)

############################
# Feature CNN definition
############################
        cnn_channels = [inp_depth] + cnn_depth
        cnn_nlayers = len(cnn_filter_size)
        cnn_use_bn = [True] * cnn_nlayers
        cnn_act = [tf.nn.relu] * cnn_nlayers
        cnn = nn.cnn(cnn_filter_size, cnn_channels, cnn_pool, cnn_act,
                     cnn_use_bn, phase_train=phase_train, wd=wd, scope='cnn')

        subsample = np.array(cnn_pool).prod()
        cnn_h = inp_height / subsample
        cnn_w = inp_width / subsample
        feat_dim = cnn_h * cnn_w * cnn_channels[-1]

############################
# Matching MLP definition
############################
        mlp_nlayers = len(mlp_dims)
        mlp_dims = [2 * feat_dim] + mlp_dims
        mlp_act = [tf.nn.relu] * (mlp_nlayers - 1) + [tf.sigmoid]
        mlp = nn.mlp(mlp_dims, mlp_act)

############################
# Computation graph
############################
        f1 = cnn(x1)
        f1 = tf.reshape(f1[-1], [-1, feat_dim])
        f2 = cnn(x2)
        f2 = tf.reshape(f2[-1], [-1, feat_dim])
        f_join = tf.concat(1, [f1, f2])
        y_out = mlp(f_join)[-1]
        y_out = tf.reshape(y_out, [-1])

############################
# Loss function
############################
        bce = f_bce(y_out, y_gt)
        bce = tf.reduce_sum(bce)
        tf.add_to_collection('losses', bce)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

####################
# Optimizer
####################
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        eps = 1e-7
        train_step = tf.train.AdamOptimizer(learn_rate, epsilon=eps).minimize(
            total_loss, global_step=global_step)

############################
# Computation nodes
############################
        model['x1'] = x1
        model['x2'] = x2
        model['y_gt'] = y_gt
        model['y_out'] = y_out
        model['loss'] = total_loss
        model['learn_rate'] = learn_rate
        model['train_step'] = train_step

    return model
