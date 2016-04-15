from __future__ import division

import cv2
import tensorflow as tf
import nnlib as nn
import numpy as np

from grad_clip_optim import GradientClipOptimizer

def get_device_fn(device):
    """Choose device for different ops."""
    OPS_ON_CPU = set(['ResizeBilinear', 'Print', 'ResizeBilinearGrad', 'Mod', 'CumMin', 'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense', 'BatchMatMul'])

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
        bboxA: [N X 4 tensor] format = [left, top, right, bottom]
        bboxB: [N X 4 tensor] 

    Return:
        IOU: [N X 1 tensor]
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

def transform_box(bbox, height, width):
    """ Transform the bounding box format 
        Args:
            bbox: [N X 4] input N bbox
                  fromat = [cx, cy, log(w/W), log(h/H)]
            height: height of original image
            width: width of original image

        Return:
            bbox: [N X 4] output N bbox
                  format = [left top right bottom]
    """
    x, y, w, h = tf.split(1, 4, bbox)
    
    h = tf.exp(h) * height
    w = tf.exp(w) * width
    x = (x + 1) * width/2
    y = (y + 1) * height/2
    
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2 

    bbox_out = tf.concat(1, [x1, y1, x2, y2])

    return bbox_out

def build_tracking_model(opt, device='/cpu:0'):
    model = {}
    
    rnn_seq_len = opt['rnn_seq_len']
    cnn_filter_size = opt['cnn_filter_size']
    cnn_num_filter = opt['cnn_num_filter']
    cnn_pool_size = opt['cnn_pool_size']
    num_channel = opt['img_channel']
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
        # input image [B, T, H, W, C]
        imgs = tf.placeholder(tf.float32, [None, rnn_seq_len, height, width, num_channel])
        img_shape = tf.shape(imgs)
        batch_size = img_shape[0]

        init_bbox = tf.placeholder(tf.float32, [batch_size, 4])
        gt_bbox = tf.placeholder(tf.float32, [batch_size, rnn_seq_len, 4])
        gt_score = tf.placeholder(tf.float32, [batch_size, rnn_seq_len, 1])

        model['imgs'] = imgs
        model['gt_bbox'] = gt_bbox
        model['gt_score'] = gt_score
        model['init_bbox'] = init_bbox
        model['phase_train'] = phase_train

        IOU_score = [None] * batch_size
        predict_bbox = tf.zeros([batch_size, rnn_seq_len, 4])
        predict_score = tf.zeros([batch_size, rnn_seq_len, 1])

        # define a CNN model
        cnn_filter = cnn_filter_size
        cnn_nlayer = len(cnn_filter)
        cnn_channel = [num_channel] + cnn_num_filter
        cnn_pool = cnn_pool_size
        cnn_act = [tf.nn.relu] * cnn_nlayer
        cnn_use_bn = [use_bn] * cnn_nlayer

        cnn_model = nn.cnn(cnn_filter, cnn_channel, cnn_pool, cnn_act,
                      cnn_use_bn, phase_train=phase_train, wd=weight_decay)
        
        # define a RNN(LSTM) model
        cnn_subsample = np.array(cnn_pool).prod()
        rnn_h = int(height / cnn_subsample)
        rnn_w = int(width / cnn_subsample)
        rnn_dim = cnn_channel[-1]
        rnn_inp_dim = rnn_h * rnn_w * rnn_dim   # input dimension of RNN

        # define a linear mapping: initial bbox -> hidden state
        init_bbox = inverse_transform_box(init_bbox, height, width)
        W_bbox = tf.Variable(tf.truncated_normal([rnn_hidden_dim, 4], stddev=0.1))
        W_score = tf.Variable(tf.truncated_normal([rnn_hidden_dim, 1], stddev=0.01))
        b_bbox = tf.Variable(tf.zeros([4], stddev=0.1))
        b_score = tf.Variable(tf.zeros([1], stddev=0.1))

        rnn_state = [None] * (rnn_seq_len + 1)
        predict_bbox = [None] * (rnn_seq_len + 1)
        predict_score = [None] * (rnn_seq_len + 1)
        predict_bbox[-1] = init_bbox
        predict_score[-1] = 1
        rnn_state[-1] = tf.zeros(tf.pack([batch_size, rnn_hidden_dim * 2]))
        rnn_hidden_feat = [None] * rnn_seq_len

        rnn_cell = nn.lstm(rnn_inp_dim, rnn_hidden_dim, wd=weight_decay)

        for tt in xrange(rnn_seq_len):
            # extract global CNN feature map
            h_cnn_global = cnn_model(imgs[:, tt])
            cnn_global_feat_map = h_cnn_global[-1]
            model['cnn_global_feat_map'] = cnn_global_feat_map

            # extract ROI CNN feature map 
            x1, y1, x2, y2 = tf.split(1, 4, predict_bbox[tt-1])          
            mask = tf.zeros(batch_size, height, width, num_channel)
            for ii in xrange(batch_size):
                mask[ii, y1 : y2, x1 : x2, :] = imgs[ii, y1 : y2, x1 : x2, :]

            h_cnn_roi = cnn_model(mask)
            cnn_roi_feat_map = h_cnn_roi[-1]
            model['cnn_roi_feat_map'] = cnn_roi_feat_map

            # going through a RNN
            # RNN input = global CNN feat map + ROI CNN feat map 
            rnn_input = tf.concat([tf.reshape(cnn_global_feat_map, [-1, rnn_inp_dim]), tf.reshape(cnn_roi_feat_map, [-1, rnn_inp_dim])])        
            rnn_state[tt], _, _, _ = rnn_cell(rnn_input, rnn_state[tt - 1])
            rnn_hidden_feat[tt] = tf.slice(rnn_state[tt], [0, rnn_hidden_dim], [-1, rnn_hidden_dim])

            # predict bbox and score            
            raw_predict_bbox = tf.matmul(tf.concat(0, rnn_hidden_feat[tt]), W_bbox) + b_bbox
            predict_bbox[tt] = transform_box(raw_predict_bbox, height, width)
            predict_score[tt] = tf.sigmoid(tf.matmul(tf.concat(0, rnn_hidden_feat[tt]), W_score)) b_score
        
        IOU_score = compute_IOU(tf.concat(0, predict_bbox[:-1]), gt_bbox)

        model['IOU_score'] = IOU_score
        model['predict_bbox'] = predict_bbox[]
        model['predict_score'] = predict_score

        # IOU loss + cross-entropy loss
        IOU_loss = tf.reduce_sum(gt_score * (1-IOU_score)) / batch_size
        cross_entropy = -tf.reduce_sum(gt_score * tf.log(predict_score) + (1 - gt_score) * tf.log(1 - predict_score)) / batch_size

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
