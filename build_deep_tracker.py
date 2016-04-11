import tensorflow as tf
import nnlib as nn
import numpy as np

from grad_clip_optim import GradientClipOptimizer

def get_device_fn(device):
    """Choose device for different ops."""
    OPS_ON_CPU = set(['ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'CumMin', 'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense', 'BatchMatMul'])

    def _device_fn(op):
        if op.type in OPS_ON_CPU:
            return "/cpu:0"
        else:
            # Other ops will be placed on GPU if available, otherwise CPU.
            return device

    return _device_fn

def compute_IOU(bboxA, bboxB):
    """Compute the Intersection Over Union.
    Args:
        bboxA: [n, 4] tensor; order is left, top, right, bottom
        bboxB: [n, 4] tensor

    Return:
        IOU: [n, 1]
    """

    x1A, y1A, x2A, y2A = tf.split(1, 4, bboxA)
    x1B, y1B, x2B, y2B = tf.split(1, 4, bboxB)

    # compute intersection
    x1_max = tf.maximum(x1A, x1B)
    y1_max = tf.maximum(y1A, y1B)
    x2_min = tf.minimum(x2A, x2B)
    y2_min = tf.minimum(y2A, y2B)

    overlap_flag = tf.logical_and( tf.less(x1_max, x2_min), tf.less(y1_max, y2_min) )
    overlap_area = tf.mul(tf.to_float(overlap_flag), tf.mul( x2_min - x1_max, y2_min - y1_max ) )

    # compute union
    areaA = tf.mul( x2A - x1A, y2A - y1A )
    areaB = tf.mul( x2B - x1B, y2B - y1B )
    union_area = areaA + areaB - overlap_area

    return tf.div(overlap_area, union_area)

def build_tracking_model(opt, device='/cpu:0'):
    model = {}
    
    batch_size = opt['batch_size']
    cnn_filter_size = opt['cnn_filter_size']
    cnn_num_channel = opt['cnn_num_channel']
    cnn_pool_size = opt['cnn_pool_size']
    img_num_channel = opt['img_channel']
    use_bn = opt['use_batch_norm']  
    height = opt['img_height']
    width = opt['img_width']
    weight_decay = opt['weight_decay']
    rnn_hidden_dim = opt['rnn_hidden_dim']
    mlp_hidden_dim = opt['mlp_hidden_dim']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay_step = opt['learn_rate_decay_step']
    learn_rate_decay_rate = opt['learn_rate_decay_rate']

    with tf.device(get_device_fn(device)):
        phase_train = tf.placeholder('bool')
        imgs = tf.placeholder(tf.float32, [batch_size, height, width, img_num_channel])
        init_bbox = tf.placeholder(tf.float32, [1, 4])
        gt_bbox = tf.placeholder(tf.float32, [batch_size, 4])
        gt_score = tf.placeholder(tf.float32, [batch_size, 1])

        model['imgs'] = imgs
        model['gt_bbox'] = gt_bbox
        model['gt_score'] = gt_score
        model['init_bbox'] = init_bbox
        model['phase_train'] = phase_train

        # define a CNN model
        cnn_filter = cnn_filter_size
        cnn_nlayer = len(cnn_filter)
        cnn_channel = [img_num_channel] + cnn_num_channel
        cnn_pool = cnn_pool_size
        cnn_act = [tf.nn.relu] * cnn_nlayer
        cnn_use_bn = [use_bn] * cnn_nlayer

        cnn_model = nn.cnn(cnn_filter, cnn_channel, cnn_pool, cnn_act,
                      cnn_use_bn, phase_train=phase_train, wd=weight_decay)
        
        h_cnn = cnn_model(imgs) # h_cnn is a list and stores the output of every layer
        cnn_output = h_cnn[-1]
        model['cnn_output'] = cnn_output

        # define a RNN(LSTM) model
        cnn_subsample = np.array(cnn_pool).prod()
        rnn_h = height / cnn_subsample
        rnn_w = width / cnn_subsample
        rnn_dim = cnn_channel[-1]       
        rnn_inp_dim = rnn_h * rnn_w * rnn_dim

        # define a linear mapping: initial bbox -> hidden state
        W_bbox = tf.Variable(tf.truncated_normal([rnn_hidden_dim, 4], stddev=0.01))
        W_score = tf.Variable(tf.truncated_normal([rnn_hidden_dim, 1], stddev=0.01))

        rnn_state = [None] * (batch_size + 1)
        rnn_state[-1] = tf.concat(1, [tf.zeros([1, rnn_hidden_dim], tf.float32), tf.matmul(init_bbox, W_bbox, False, True)])
        rnn_hidden_feat = [None] * batch_size
        predict_bbox = tf.zeros([batch_size, 4])
        predict_score = tf.zeros([batch_size, 1])
        IOU_score = [None] * batch_size

        rnn_cell = nn.lstm(rnn_inp_dim, rnn_hidden_dim, wd=weight_decay)

        cnn_feat = tf.split(0, batch_size, cnn_output)

        for tt in xrange(batch_size):
            cnn_feat[tt] = tf.reshape(cnn_feat[tt], [1, rnn_inp_dim])
            rnn_state[tt], _, _, _ = rnn_cell(cnn_feat[tt], rnn_state[tt - 1])
            rnn_hidden_feat[tt] = tf.slice(rnn_state[tt], [0, rnn_hidden_dim], [-1, rnn_hidden_dim])
            
        predict_bbox = tf.matmul(tf.concat(0, rnn_hidden_feat), W_bbox)
        predict_score = tf.matmul(tf.concat(0, rnn_hidden_feat), W_score)

        IOU_score = compute_IOU(predict_bbox, gt_bbox)

        model['predict_bbox'] = predict_bbox
        model['predict_score'] = predict_score

        # IOU loss + cross-entropy loss
        IOU_loss = tf.reduce_sum(gt_score * (- IOU_score))
        cross_entropy = -tf.reduce_sum(gt_score * tf.log(predict_score))

        model['IOU_loss'] = IOU_loss
        model['CE_loss'] = cross_entropy
        
        global_step = tf.Variable(0.0)
        eps = 1e-7

        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, learn_rate_decay_step,
            learn_rate_decay_rate, staircase=True)
        model['learn_rate'] = learn_rate

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=1.0).minimize(IOU_loss + cross_entropy, global_step=global_step)
        model['train_step'] = train_step

    return model
