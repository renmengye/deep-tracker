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


def get_idx_map(shape):
    """Get index map for a image.
    Args:
        shape: [B, T, H, W] or [B, H, W]
    Returns:
        idx: [B, T, H, W, 2], or [B, H, W, 2]
    """
    s = shape
    ndims = tf.shape(s)
    wdim = ndims - 1
    hdim = ndims - 2
    idx_shape = tf.concat(0, [s, tf.constant([1])])
    ones_h = tf.ones(hdim - 1, dtype='int32')
    ones_w = tf.ones(wdim - 1, dtype='int32')
    h_shape = tf.concat(0, [ones_h, tf.constant([-1]), tf.constant([1, 1])])
    w_shape = tf.concat(0, [ones_w, tf.constant([-1]), tf.constant([1])])

    idx_y = tf.zeros(idx_shape, dtype='float')
    idx_x = tf.zeros(idx_shape, dtype='float')

    h = tf.slice(s, ndims - 2, [1])
    w = tf.slice(s, ndims - 1, [1])
    idx_y += tf.reshape(tf.to_float(tf.range(h[0])), h_shape)
    idx_x += tf.reshape(tf.to_float(tf.range(w[0])), w_shape)
    idx = tf.concat(ndims[0], [idx_y, idx_x])

    return idx


def get_filled_box_idx(idx, top_left, bot_right):
    """Fill a box with top left and bottom right coordinates.
    Args:
        idx: [B, T, H, W, 2] or [B, H, W, 2] or [H, W, 2]
        top_left: [B, T, 2] or [B, 2] or [2]
        bot_right: [B, T, 2] or [B, 2] or [2]
    """
    ss = tf.shape(idx)
    ndims = tf.shape(ss)
    batch = tf.slice(ss, [0], ndims - 3)
    coord_shape = tf.concat(0, [batch, tf.constant([1, 1, 2])])
    top_left = tf.reshape(top_left, coord_shape)
    bot_right = tf.reshape(bot_right, coord_shape)
    lower = tf.reduce_prod(tf.to_float(idx >= top_left), ndims - 1)
    upper = tf.reduce_prod(tf.to_float(idx <= bot_right), ndims - 1)
    box = lower * upper

    return box


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

    # overlap_flag = tf.logical_and( tf.less(x1_max, x2_min), tf.less(y1_max, y2_min))

    overlap_flag = tf.to_float(tf.less(x1_max, x2_min)) * \
        tf.to_float(tf.less(y1_max, y2_min))

    overlap_area = tf.mul(overlap_flag, tf.mul(
        x2_min - x1_max, y2_min - y1_max))

    # compute union
    areaA = tf.mul(x2A - x1A, y2A - y1A)
    areaB = tf.mul(x2B - x1B, y2B - y1B)
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
            bbox: [N X 4] output rounded N bbox
                  format = [left top right bottom]
    """
    x, y, w, h = tf.split(1, 4, bbox)

    h = tf.exp(h) * height
    w = tf.exp(w) * width
    x = (x + 1) * width / 2
    y = (y + 1) * height / 2

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    bbox_out = tf.concat(1, [x1, y1, x2, y2])

    return bbox_out


def inverse_transform_box(bbox, height, width):
    """ Transform the bounding box format 
        Args:
            bbox: [N X 4] input N bbox
                  format = [left top right bottom]                  
            height: height of original image
            width: width of original image

        Return:
            bbox: [N X 4] output rounded N bbox
                  fromat = [cx, cy, log(w/W), log(h/H)]
    """
    x1, y1, x2, y2 = tf.split(1, 4, bbox)

    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2

    x /= width / 2
    y /= height / 2
    x -= 1
    y -= 1
    w = tf.log(w / width)
    h = tf.log(h / height)

    bbox_out = tf.concat(1, [x, y, h, w])

    return bbox_out


def _get_reduction_indices(a):
    """Gets the list of axes to sum over."""
    dim = tf.shape(tf.shape(a))

    return tf.concat(0, [dim - 2, dim - 1])


def f_inter(a, b):
    """Computes intersection."""
    reduction_indices = _get_reduction_indices(a)
    return tf.reduce_sum(a * b, reduction_indices=reduction_indices)


def f_union(a, b, eps=1e-5):
    """Computes union."""
    reduction_indices = _get_reduction_indices(a)
    return tf.reduce_sum(a + b - (a * b) + eps,
                         reduction_indices=reduction_indices)


def f_iou(a, b, timespan=None, pairwise=False):
    """
    Computes IOU score.
    Args:
        a: [B, N, H, W], or [N, H, W], or [H, W]
        b: [B, N, H, W], or [N, H, W], or [H, W]
           in pairwise mode, the second dimension can be different,
           e.g. [B, M, H, W], or [M, H, W], or [H, W]
        pairwise: whether the inputs are already aligned, outputs [B, N] or
                  the inputs are orderless, outputs [B, N, M].
    """

    if pairwise:
        # N * [B, 1, M]
        y_list = [None] * timespan
        # [B, N, H, W] => [B, N, 1, H, W]
        a = tf.expand_dims(a, 2)
        # [B, N, 1, H, W] => N * [B, 1, 1, H, W]
        a_list = tf.split(1, timespan, a)
        # [B, M, H, W] => [B, 1, M, H, W]
        b = tf.expand_dims(b, 1)

        for ii in xrange(timespan):
            # [B, 1, M]
            y_list[ii] = f_inter(a_list[ii], b) / f_union(a_list[ii], b)

        # N * [B, 1, M] => [B, N, M]
        return tf.concat(1, y_list)
    else:
        return f_inter(a, b) / f_union(a, b)


def f_inter_box(top_left_a, bot_right_a, top_left_b, bot_right_b):
    """Computes intersection area with boxes.
    Args:
        top_left_a: [B, T, 2] or [B, 2]
        bot_right_a: [B, T, 2] or [B, 2]
        top_left_b: [B, T, 2] or [B, 2]
        bot_right_b: [B, T, 2] or [B, 2]
    Returns:
        area: [B, T]
    """
    top_left_max = tf.maximum(top_left_a, top_left_b)
    bot_right_min = tf.minimum(bot_right_a, bot_right_b)
    ndims = tf.shape(tf.shape(top_left_a))

    # Check if the resulting box is valid.
    overlap = tf.to_float(top_left_max < bot_right_min)
    overlap = tf.reduce_prod(overlap, ndims - 1)
    area = tf.reduce_prod(bot_right_min - top_left_max, ndims - 1)
    area = overlap * tf.abs(area)

    return area


def f_iou_box(top_left_a, bot_right_a, top_left_b, bot_right_b):
    """Computes IoU of boxes.
    Args:
        top_left_a: [B, T, 2] or [B, 2]
        bot_right_a: [B, T, 2] or [B, 2]
        top_left_b: [B, T, 2] or [B, 2]
        bot_right_b: [B, T, 2] or [B, 2]
    Returns:
        iou: [B, T]
    """
    inter_area = f_inter_box(top_left_a, bot_right_a, top_left_b, bot_right_b)
    inter_area = tf.maximum(inter_area, 1e-6)
    ndims = tf.shape(tf.shape(top_left_a))
    # area_a = tf.reduce_prod(bot_right_a - top_left_a, ndims - 1)
    # area_b = tf.reduce_prod(bot_right_b - top_left_b, ndims - 1)
    check_a = tf.reduce_prod(tf.to_float(top_left_a < bot_right_a), ndims - 1)
    area_a = check_a * tf.reduce_prod(bot_right_a - top_left_a, ndims - 1)
    check_b = tf.reduce_prod(tf.to_float(top_left_b < bot_right_b), ndims - 1)
    area_b = check_b * tf.reduce_prod(bot_right_b - top_left_b, ndims - 1)
    union_area = (area_a + area_b - inter_area + 1e-5)
    union_area = tf.maximum(union_area, 1e-5)
    iou = inter_area / union_area
    iou = tf.maximum(iou, 1e-5)
    iou = tf.minimum(iou, 1.0)

    return iou


def build_tracking_model(opt, device='/cpu:0'):
    """
    Given the T+1 sequence of input, return T sequence of output.
    """
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
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay_step = opt['learn_rate_decay_step']
    learn_rate_decay_rate = opt['learn_rate_decay_rate']
    pretrain_model_filename = opt['pretrain_model_filename']
    is_pretrain = opt['is_pretrain']

    with tf.device(get_device_fn(device)):
        phase_train = tf.placeholder('bool')

        # input image [B, T+1, H, W, C]
        anneal_threshold = tf.placeholder(tf.float32, [1])
        imgs = tf.placeholder(
            tf.float32, [None, rnn_seq_len + 1, height, width, num_channel])
        img_shape = tf.shape(imgs)
        batch_size = img_shape[0]

        init_bbox = tf.placeholder(tf.float32, [None, 4])
        gt_bbox = tf.placeholder(tf.float32, [None, rnn_seq_len + 1, 4])
        gt_score = tf.placeholder(tf.float32, [None, rnn_seq_len + 1])
        IOU_score = [None] * (rnn_seq_len + 1)
        IOU_score[0] = 1

        model['imgs'] = imgs
        model['gt_bbox'] = gt_bbox
        model['gt_score'] = gt_score
        model['init_bbox'] = init_bbox
        model['phase_train'] = phase_train
        model['anneal_threshold'] = anneal_threshold

        # define a CNN model
        cnn_filter = cnn_filter_size
        cnn_nlayer = len(cnn_filter)
        cnn_channel = [num_channel] + cnn_num_filter
        cnn_pool = cnn_pool_size
        cnn_act = [tf.nn.relu] * cnn_nlayer
        cnn_use_bn = [use_bn] * cnn_nlayer

        # load pretrained model
        if is_pretrain:
            h5f = h5py.File(pretrain_model_filename, 'r')

            # for key, value in h5f.iteritems():
            #     print key, value

            cnn_init_w = [{'w': h5f['cnn_w_{}'.format(ii)][:],
                           'b': h5f['cnn_b_{}'.format(ii)][:]}
                          for ii in xrange(cnn_nlayer)]

            for ii in xrange(cnn_nlayer):
                for tt in xrange(3 * rnn_seq_len):
                    for w in ['beta', 'gamma']:
                        cnn_init_w[ii]['{}_{}'.format(w, tt)] = h5f[
                            'cnn_{}_0_{}'.format(ii, w)][:]

        cnn_model = nn.cnn(cnn_filter, cnn_channel, cnn_pool, cnn_act,
                           cnn_use_bn, phase_train=phase_train, wd=weight_decay, init_weights=cnn_init_w)

        # define a RNN(LSTM) model
        cnn_subsample = np.array(cnn_pool).prod()
        rnn_h = int(height / cnn_subsample)
        rnn_w = int(width / cnn_subsample)
        rnn_dim = cnn_channel[-1]
        cnn_out_dim = rnn_h * rnn_w * rnn_dim   # input dimension of RNN
        rnn_inp_dim = cnn_out_dim * 3

        rnn_state = [None] * (rnn_seq_len + 1)
        predict_bbox = [None] * (rnn_seq_len + 1)
        predict_score = [None] * (rnn_seq_len + 1)
        predict_bbox[0] = init_bbox
        predict_score[0] = 1

        # rnn_state[-1] = tf.zeros(tf.pack([batch_size, rnn_hidden_dim * 2]))

        rnn_state[-1] = tf.concat(1, [inverse_transform_box(gt_bbox[:, 0, :],
                                                            height, width), tf.zeros(tf.pack([batch_size, rnn_hidden_dim * 2 - 4]))])

        rnn_hidden_feat = [None] * rnn_seq_len

        rnn_cell = nn.lstm(rnn_inp_dim, rnn_hidden_dim, wd=weight_decay)

        # define two linear mapping MLPs:
        # RNN hidden state -> bbox
        # RNN hidden state -> score
        bbox_mlp_dims = [rnn_hidden_dim, 4]
        bbox_mlp_act = [None]

        bbox_mlp = nn.mlp(bbox_mlp_dims, bbox_mlp_act, add_bias=True,
                          phase_train=phase_train, wd=weight_decay)

        score_mlp_dims = [rnn_hidden_dim, 1]
        score_mlp_act = [tf.sigmoid]

        score_mlp = nn.mlp(score_mlp_dims, score_mlp_act,
                           add_bias=True, phase_train=phase_train, wd=weight_decay)

        # training through time
        for tt in xrange(rnn_seq_len):
            # extract global CNN feature map of the current frame
            h_cnn_global_now = cnn_model(imgs[:, tt, :, :, :])
            cnn_global_feat_now = h_cnn_global_now[-1]
            cnn_global_feat_now = tf.stop_gradient(
                cnn_global_feat_now)  # fix CNN during training
            model['cnn_global_feat_now'] = cnn_global_feat_now

            # extract ROI CNN feature map of the current frame
            use_pred_bbox = tf.to_float(
                tf.less(tf.random_uniform([1]), anneal_threshold))
            x1, y1, x2, y2 = tf.split(
                1, 4, use_pred_bbox * predict_bbox[tt] + (1 - use_pred_bbox) * gt_bbox[:, tt, :])
            idx_map = get_idx_map(tf.pack([batch_size, height, width]))
            mask_map = get_filled_box_idx(idx_map, tf.concat(
                1, [y1, x1]), tf.concat(1, [y2, x2]))

            ROI_img = []
            for cc in xrange(num_channel):
                ROI_img.append(imgs[:, tt, :, :, cc] * mask_map)

            h_cnn_roi_now = cnn_model(
                tf.transpose(tf.pack(ROI_img), [1, 2, 3, 0]))
            cnn_roi_feat_now = h_cnn_roi_now[-1]
            cnn_roi_feat_now = tf.stop_gradient(
                cnn_roi_feat_now)   # fix CNN during training
            model['cnn_roi_feat_now'] = cnn_roi_feat_now

            # extract global CNN feature map of the next frame
            h_cnn_global_next = cnn_model(imgs[:, tt + 1, :, :, :])
            cnn_global_feat_next = h_cnn_global_next[-1]
            cnn_global_feat_next = tf.stop_gradient(
                cnn_global_feat_next)  # fix CNN during training
            model['cnn_global_feat_next'] = cnn_global_feat_next

            # going through a RNN
            # RNN input = global CNN feat map + ROI CNN feat map
            rnn_input = tf.concat(1, [tf.reshape(cnn_global_feat_now, [-1, cnn_out_dim]), tf.reshape(
                cnn_roi_feat_now, [-1, cnn_out_dim]), tf.reshape(cnn_global_feat_next, [-1, cnn_out_dim])])

            rnn_state[tt], _, _, _ = rnn_cell(rnn_input, rnn_state[tt - 1])
            rnn_hidden_feat[tt] = tf.slice(
                rnn_state[tt], [0, rnn_hidden_dim], [-1, rnn_hidden_dim])

            # predict bbox and score
            raw_predict_bbox = bbox_mlp(rnn_hidden_feat[tt])[0]
            predict_bbox[
                tt + 1] = transform_box(raw_predict_bbox, height, width)
            predict_score[
                tt + 1] = score_mlp(rnn_hidden_feat[tt])[-1]

            # compute IOU
            IOU_score[
                tt + 1] = compute_IOU(predict_bbox[tt + 1], gt_bbox[:, tt + 1, :])

        # # [B, T, 4]
        # predict_bbox_reshape = tf.concat(
        #     1, [tf.expand_dims(tmp, 1) for tmp in predict_bbox[:-1]])
        # # [B, T]
        # IOU_score = f_iou_box(predict_bbox_reshape[:, :, 0: 1], predict_bbox_reshape[
        #                       :, :, 2: 3], gt_bbox[:, :, 0: 1], gt_bbox[:, :, 2: 3])

        predict_bbox = tf.transpose(tf.pack(predict_bbox[1:]), [1, 0, 2])

        model['IOU_score'] = tf.transpose(tf.pack(IOU_score[1:]), [1, 0, 2])

        # model['IOU_score'] = IOU_score
        model['predict_bbox'] = predict_bbox
        model['predict_score'] = tf.transpose(tf.pack(predict_score[1:]))

        # compute IOU loss
        batch_size_f = tf.to_float(batch_size)
        rnn_seq_len_f = tf.to_float(rnn_seq_len)
        # IOU_loss = tf.reduce_sum(gt_score * (- tf.concat(1, IOU_score))) / (batch_size_f * rnn_seq_len_f)

        valid_seq_length = tf.reduce_sum(gt_score[:, 1:], [1])
        valid_seq_length = tf.maximum(1.0, valid_seq_length)
        IOU_loss = gt_score[:, 1:] * (- tf.concat(1, IOU_score[1:]))
        # [B,T] => [B, 1]
        IOU_loss = tf.reduce_sum(IOU_loss, [1])
        # [B, 1]
        IOU_loss /= valid_seq_length
        # [1]
        IOU_loss = tf.reduce_sum(IOU_loss) / batch_size_f

        # compute L2 loss
        diff_bbox = gt_bbox[:, 1:, :] - predict_bbox
        diff_x1 = diff_bbox[:, :, 0] / width
        diff_y1 = diff_bbox[:, :, 1] / height
        diff_x2 = diff_bbox[:, :, 2] / width
        diff_y2 = diff_bbox[:, :, 3] / height
        diff_bbox = tf.transpose(
            tf.pack([diff_x1, diff_y1, diff_x2, diff_y2]), [1, 2, 0])

        L2_loss = tf.reduce_sum(diff_bbox * diff_bbox, [1, 2]) / 4
        L2_loss /= valid_seq_length
        L2_loss = tf.reduce_sum(L2_loss) / batch_size_f

        # cross-entropy loss
        cross_entropy = -tf.reduce_sum(gt_score[:, 1:] * tf.log(tf.concat(1, predict_score[1:])) + (
            1 - gt_score[:, 1:]) * tf.log(1 - tf.concat(1, predict_score[1:]))) / (batch_size_f * rnn_seq_len_f)

        model['IOU_loss'] = IOU_loss
        model['L2_loss'] = L2_loss
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
