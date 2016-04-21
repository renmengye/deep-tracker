from __future__ import division

import cv2
import h5py
import tensorflow as tf
import nnlib as nn
import numpy as np

from grad_clip_optim import GradientClipOptimizer

def get_device_fn(device):
    """Choose device for different ops."""
    OPS_ON_CPU = set(['ResizeBilinear', 'Print', 'ResizeBilinearGrad', 'Mod', 'CumMin',
                      'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense', 'BatchMatMul'])

    def _device_fn(op):
        if op.type in OPS_ON_CPU:
            return "/cpu:0"
        else:
            # Other ops will be placed on GPU if available, otherwise CPU.
            return device

    return _device_fn

def compute_soft_IOU_score(heat_map_A, heat_map_B):
    """Compute the Intersection Over Union.
    Args:
        heat_map_A: [B X T X H X W tensor]
        heat_map_B: [B X T X H X W tensor] 

    Return:
        IOU: [N X 1 tensor]
    """
    intersect_tensor = heat_map_A * heat_map_B
    area_A = tf.reduce_sum(heat_map_A, [2, 3])
    area_B = tf.reduce_sum(heat_map_B, [2, 3])
    intersect_area = tf.reduce_sum(intersect_tensor, [2, 3])    
    union_area = area_A + area_B - intersect_area
    iou_all = intersect_area / (union_area + 1.0e-5)
    
    return tf.reduce_sum(iou_all) / tf.reduce_prod(tf.to_float(tf.shape(iou_all)))

def build_tracking_model(opt, device='/cpu:0'):
    """
    Given the T+1 sequence of input, return T sequence of output.
    """
    model = {}

    # data parameters
    num_channel = opt['img_channel']
    height = opt['img_height']
    width = opt['img_width']

    # segmentation CNN    
    seg_cnn_filter_size = opt['seg_cnn_filter_size']
    seg_cnn_num_filter = opt['seg_cnn_num_filter']
    seg_cnn_pool_size = opt['seg_cnn_pool_size']    
    seg_cnn_use_bn = opt['seg_cnn_use_bn']
    
    # matching CNN

    # convolutional LSTM
    conv_lstm_seq_len = opt['conv_lstm_seq_len']
    conv_lstm_filter_size = opt['conv_lstm_filter_size']
    conv_lstm_hidden_depth = opt['conv_lstm_hidden_depth']

    # optimization parameters    
    weight_decay = opt['weight_decay']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay_step = opt['learn_rate_decay_step']
    learn_rate_decay_rate = opt['learn_rate_decay_rate']

    pretrain_model_filename = opt['pretrain_model_filename']    
    is_pretrain = opt['is_pretrain']

    with tf.device(get_device_fn(device)):
        phase_train = tf.placeholder('bool')
        anneal_threshold = tf.placeholder(tf.float32, [1])

        # input image [B, T+1, H, W, C]            
        imgs = tf.placeholder(tf.float32, [None, conv_lstm_seq_len + 1, height, width, num_channel])        
        init_heat_map = tf.placeholder(tf.float32, [None, None, None])
        gt_heat_map = tf.placeholder(tf.float32, [None, conv_lstm_seq_len + 1, None, None])
        
        img_shape = tf.shape(imgs)
        batch_size = img_shape[0]        
        
        IOU_score = [None] * (conv_lstm_seq_len + 1)
        
        model['imgs'] = imgs        
        model['gt_heat_map'] = gt_heat_map
        model['init_heat_map'] = init_heat_map
        model['phase_train'] = phase_train
        model['anneal_threshold'] = anneal_threshold

        conv_lstm_state = [None] * (conv_lstm_seq_len + 1)
        predict_heat_map = [None] * (conv_lstm_seq_len + 1)
        predict_heat_map[0] = gt_heat_map[:, 0, :, :]
        
        # define a CNN model
        seg_cnn_filter = seg_cnn_filter_size
        seg_cnn_nlayer = len(seg_cnn_filter)
        seg_cnn_channel = [num_channel] + seg_cnn_num_filter
        seg_cnn_pool = seg_cnn_pool_size
        seg_cnn_act = [tf.nn.relu] * seg_cnn_nlayer
        seg_cnn_bn = [seg_cnn_use_bn] * seg_cnn_nlayer

        # load pretrained CNN model
        if is_pretrain:
            h5f = h5py.File(pretrain_model_filename, 'r')

            # for key, value in h5f.iteritems():
            #     print key, value

            cnn_init_w = [{'w': h5f['cnn_w_{}'.format(ii)][:],
                           'b': h5f['cnn_b_{}'.format(ii)][:]}
                          for ii in xrange(seg_cnn_nlayer)]

            for ii in xrange(seg_cnn_nlayer):
                for tt in xrange(3 * conv_lstm_seq_len):
                    for w in ['beta', 'gamma']:
                        cnn_init_w[ii]['{}_{}'.format(w, tt)] = h5f[
                            'cnn_{}_0_{}'.format(ii, w)][:]

        seg_cnn_model = nn.cnn(seg_cnn_filter, seg_cnn_channel, seg_cnn_pool, seg_cnn_act,
                           seg_cnn_bn, phase_train=phase_train, wd=weight_decay, init_weights=cnn_init_w)

        # define a convolutional LSTM model
        seg_cnn_subsample = np.array(seg_cnn_pool).prod()
        conv_lstm_h = int(height / seg_cnn_subsample)
        conv_lstm_w = int(width / seg_cnn_subsample)

        conv_lstm_state = [None] * (conv_lstm_seq_len + 1)
        conv_lstm_state[-1] = tf.zeros(tf.pack([batch_size, conv_lstm_h, conv_lstm_w, conv_lstm_hidden_depth * 2]))

        conv_lstm_hidden_feat = [None] * conv_lstm_seq_len

        conv_lstm_cell = nn.conv_lstm(3 * seg_cnn_channel[-1], conv_lstm_hidden_depth, conv_lstm_filter_size, wd=weight_decay)

        # define 1 layer conv:
        post_cnn_model = nn.cnn([1], [conv_lstm_hidden_depth, 1], [None], [tf.sigmoid], [None], phase_train=phase_train, wd=weight_decay)

        # training through time
        for tt in xrange(conv_lstm_seq_len):
            # extract global segment CNN feature map of the current frame
            seg_cnn_map = seg_cnn_model(imgs[:, tt, :, :, :])
            seg_cnn_feat_map = seg_cnn_map[-1]
            seg_cnn_feat_map = tf.stop_gradient(seg_cnn_feat_map)  # fix CNN during training
            model['seg_cnn_feat_map'] = seg_cnn_feat_map

            # extract ROI segment CNN feature map of the current frame            
            use_pred_bbox = tf.to_float(tf.less(tf.random_uniform([1]), anneal_threshold))
            tmp_mask = use_pred_bbox * predict_heat_map[tt] + (1 - use_pred_bbox) * gt_heat_map[:,tt,:,:]
                
            seg_cnn_roi_feat_map = seg_cnn_feat_map * tf.expand_dims(tmp_mask, 3)
            model['seg_cnn_roi_feat_map'] = seg_cnn_roi_feat_map

            # extract global CNN feature map of the next frame
            seg_cnn_map_next = seg_cnn_model(imgs[:, tt + 1, :, :, :])
            seg_cnn_feat_map_next = seg_cnn_map_next[-1]
            seg_cnn_feat_map_next = tf.stop_gradient(
                seg_cnn_feat_map_next)  # fix CNN during training
            model['seg_cnn_feat_map_next'] = seg_cnn_feat_map_next

            # going through a convolutional LSTM
            # RNN input = global CNN feat map + ROI CNN feat map            
            conv_lstm_input = tf.concat(3, [seg_cnn_feat_map, seg_cnn_roi_feat_map, seg_cnn_feat_map_next])
            
            conv_lstm_state[tt] = conv_lstm_cell(conv_lstm_input, conv_lstm_state[tt-1])

            conv_lstm_hidden_feat[tt] = tf.slice(conv_lstm_state[tt], [0, 0, 0, conv_lstm_hidden_depth], [-1, -1, -1, conv_lstm_hidden_depth])

            # predict heat map
            post_cnn_map = post_cnn_model(conv_lstm_hidden_feat[tt])
            predict_heat_map[tt+1] = tf.squeeze(post_cnn_map[-1])
    
        # compute IOU loss
        predict_heat_map = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in predict_heat_map[1:]])
        IOU_loss = -compute_soft_IOU_score(predict_heat_map, gt_heat_map[:,1:,:,:])

        model['IOU_loss'] = IOU_loss
        model['predict_heat_map'] = predict_heat_map
        
        global_step = tf.Variable(0.0)
        eps = 1e-7

        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, learn_rate_decay_step,
            learn_rate_decay_rate, staircase=True)
        model['learn_rate'] = learn_rate

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=1.0).minimize(IOU_loss, global_step=global_step)
        model['train_step'] = train_step

    return model
