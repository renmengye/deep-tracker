import cslab_environ

import argparse
import numpy as np
import progress_bar as pb
import sharded_hdf5 as sh
import tensorflow as tf

import cv2
import logger
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import progress_bar as pb
import plot_utils as pu

from deep_dashboard_utils import log_register, TimeSeriesLogger

from build_conv_lstm_tracker import build_tracking_model

# from tud import get_dataset
from kitti import get_dataset

import logger
log = logger.get()

def plot_frame_with_bbox(fname, data, pred_bbox, gt_bbox, iou, num_row, num_col):
    f, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))

    for ii in xrange(num_row):
        for jj in xrange(num_col):
            axarr[ii, jj].set_axis_off()
            idx = ii * num_col + jj
            axarr[ii, jj].imshow(data[idx], cmap=cm.Greys_r)

            axarr[ii, jj].add_patch(patches.Rectangle(
                (pred_bbox[idx][0], pred_bbox[idx][1]),
                pred_bbox[idx][2] - pred_bbox[idx][0],
                pred_bbox[idx][3] - pred_bbox[idx][1],
                fill=False,
                color='r'))

            axarr[ii, jj].add_patch(patches.Rectangle(
                (gt_bbox[idx][0], gt_bbox[idx][1]),
                gt_bbox[idx][2] - gt_bbox[idx][0],
                gt_bbox[idx][3] - gt_bbox[idx][1],
                fill=False,
                color='b'))

            axarr[ii, jj].text(0, 0, ("%5.2f" % iou[idx]),
                               color=(0, 0, 0), size=8)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=80)
    plt.close('all')


def collect_draw_sequence(draw_raw_imgs, draw_raw_gt_bbox, seq_length, height, width):

    count_draw = 0
    idx_draw_frame = 0
    skip_empty = True
    draw_imgs = []
    draw_gt_box = []

    while count_draw <= seq_length:
        if draw_raw_gt_bbox[0, idx_draw_frame, 4] == 1:
            skip_empty = False

        if not skip_empty:
            draw_imgs.append(cv2.resize(
                draw_raw_imgs[idx_draw_frame], (width, height), interpolation=cv2.INTER_CUBIC))

            # draw 0-th object in the sequence
            tmp_box = np.array(draw_raw_gt_bbox[0, idx_draw_frame, :4])
            tmp_box[0] = tmp_box[0] / draw_raw_imgs.shape[2] * width
            tmp_box[1] = tmp_box[1] / draw_raw_imgs.shape[1] * height
            tmp_box[2] = tmp_box[2] / draw_raw_imgs.shape[2] * width
            tmp_box[3] = tmp_box[3] / draw_raw_imgs.shape[1] * height

            draw_gt_box.append(tmp_box)
            count_draw += 1

        idx_draw_frame += 1

    return draw_imgs, draw_gt_box


def draw_sequence(idx, draw_img_name, data, tracking_model, sess, seq_length, height, width):

    draw_data = data[idx]
    draw_raw_imgs = draw_data['images_0']
    # gt_bbox = [left top right bottom flag]
    draw_raw_gt_bbox = draw_data['gt_bbox']

    draw_imgs, draw_gt_box = collect_draw_sequence(
        draw_raw_imgs, draw_raw_gt_bbox, seq_length, height, width)

    feed_data = {tracking_model['imgs']: np.expand_dims(draw_imgs, 0),
                 tracking_model['init_bbox']: np.expand_dims(draw_gt_box[0], 0),
                 tracking_model['gt_bbox']: np.expand_dims(draw_gt_box, 0),
                 tracking_model['anneal_threshold']: [1.0],
                 tracking_model['phase_train']: False}

    draw_pred_bbox, draw_IOU_score = sess.run(
        [tracking_model['predict_bbox'], tracking_model['IOU_score']], feed_dict=feed_data)

    draw_pred_bbox = np.squeeze(draw_pred_bbox)
    draw_IOU_score = np.squeeze(draw_IOU_score)

    num_col = 2
    num_row = seq_length / num_col
    plot_frame_with_bbox(draw_img_name, draw_imgs[1:], draw_pred_bbox, draw_gt_box[
                         1:], draw_IOU_score, num_row, num_col)


def parse_args():
    parser = argparse.ArgumentParser(description='Deep conv tracker')
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--logs', default='../logs')
    return parser.parse_args()


if __name__ == "__main__":
    folder = '/ais/gobi4/mren/data/kitti/tracking/'
    args = parse_args()
    if args.gpu == -1:
        device = '/cpu:0'
    else:
        device = '/gpu:{}'.format(args.gpu)

    max_iter = 100000
    batch_size = 10
    display_iter = 10
    draw_iter = 50
    seq_length = 10     # sequence length for training
    snapshot_iter = 500
    anneal_iter = 1000
    height = 128
    width = 448
    feat_map_height = 16
    feat_map_width = 56
    img_channel = 3
    num_train_seq = 16

    # read data
    train_video_seq = []
    valid_video_seq = []
    num_valid_seq = 0
    train_data_full = get_dataset(folder, 'train')

    with sh.ShardedFileReader(train_data_full) as reader:
        num_seq = len(reader)

        for idx_seq, seq_data in enumerate(pb.get_iter(reader)):
            if idx_seq < num_train_seq:
                train_video_seq.append(seq_data)
            else:
                if seq_data['gt_bbox'].shape[0] > 0:
                    valid_video_seq.append(seq_data)
                    num_valid_seq += 1

    # logger for saving intermediate output
    model_id = 'deep-tracker-003'
    logs_folder = args.logs
    logs_folder = os.path.join(logs_folder, model_id)

    logp_logger_IOU = TimeSeriesLogger(
        os.path.join(logs_folder, 'IOU_loss.csv'),
        labels=['IOU loss'],
        name='Traning IOU Loss of BBox',
        buffer_size=1)

    logp_logger_CE = TimeSeriesLogger(
        os.path.join(logs_folder, 'CE_loss.csv'),
        labels=['CE loss'],
        name='Traning CE Loss',
        buffer_size=1)

    draw_img_name = [os.path.join(
        logs_folder, 'draw_bbox_{}.png'.format(ii)) for ii in xrange(3)]
    [log_register(draw_img_name[ii], 'image', 'box') for ii in xrange(3)]

    # for i in xrange(num_valid_seq):
    #     draw_img_name.append(os.path.join(
    #         logs_folder, 'draw_bbox_{}.png'.format(i)))

    #     if not os.path.exists(draw_img_name[i]):
    #         log_register(draw_img_name[i], 'image',
    #                      'Tracking Bounding Box {}'.format(i))

    # setting model
    opt_tracking = {}

    # data parameters
    opt_tracking['img_channel'] = img_channel
    opt_tracking['img_height'] = height
    opt_tracking['img_width'] = width

    # segmentation CNN
    opt_tracking['seg_cnn_filter_size'] = [3, 3, 3, 3, 3, 3]
    opt_tracking['seg_cnn_num_filter'] = [8, 8, 16, 16, 32, 32]
    opt_tracking['seg_cnn_pool_size'] = [1, 2, 1, 2, 1, 2]
    opt_tracking['seg_cnn_use_bn'] = True

    # convolutional LSTM
    opt_tracking['conv_lstm_seq_len'] = seq_length
    opt_tracking['conv_lstm_filter_size'] = 3
    opt_tracking['conv_lstm_hidden_depth'] = 64

    # optimization parameters
    opt_tracking['weight_decay'] = 1.0e-7
    opt_tracking['base_learn_rate'] = 1.0e-3
    opt_tracking['learn_rate_decay_step'] = 1000
    opt_tracking['learn_rate_decay_rate'] = 0.96

    opt_tracking[
        'pretrain_model_filename'] = \
        "/ais/gobi4/mren/results/img-count/fg_segm-20160419004323/weights.h5"
    opt_tracking['is_pretrain'] = True

    tracking_model = build_tracking_model(opt_tracking, device)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    nodes_run = ['train_step', 'IOU_loss', 'predict_heat_map']
    node_list = [tracking_model[i] for i in nodes_run]

    # compute sampling distribution
    cdf_seq = np.zeros(num_train_seq)
    total_count = 0
    for idx_seq, seq_data in enumerate(train_video_seq):
        if idx_seq == 0:
            cdf_seq[idx_seq] = seq_data['images_0'].shape[0]
        else:
            cdf_seq[idx_seq] = cdf_seq[idx_seq - 1] + \
                seq_data['images_0'].shape[0]

        total_count += seq_data['images_0'].shape[0]

    cdf_seq /= total_count

    # training loop
    step = 0

    while step < max_iter:
        idx_sample = 0
        anneal_prob = 0
        batch_img = np.zeros(
            [batch_size, seq_length + 1, height, width, img_channel])
        gt_raw_heat_map = np.zeros([batch_size, seq_length + 1, height, width])
        gt_heat_map = np.zeros(
            [batch_size, seq_length + 1, feat_map_height, feat_map_width])

        while idx_sample < batch_size:
            # sample sequence based on the proportion of its length
            rand_val = np.random.rand()
            idx_boolean = np.logical_and(
                rand_val < cdf_seq, rand_val > np.concatenate(([0], cdf_seq[:-1])))
            idx_video = [i for i, elem in enumerate(idx_boolean) if elem]

            seq_data = train_video_seq[idx_video[0]]

            raw_imgs = seq_data['images_0']
            # gt_bbox = [left top right bottom flag]
            gt_bbox = seq_data['gt_bbox']
            num_obj = gt_bbox.shape[0]
            num_imgs = raw_imgs.shape[0]

            if num_obj < 1:
                continue

            keep_sampling = True
            idx_obj = np.random.randint(num_obj)
            idx_frame = np.random.randint(num_imgs - seq_length)

            while keep_sampling:
                if gt_bbox[idx_obj, idx_frame, 4] == 1:
                    keep_sampling = False
                else:
                    idx_obj = np.random.randint(num_obj)
                    idx_frame = np.random.randint(num_imgs - seq_length)

            tmp_bbox = np.array(
                gt_bbox[idx_obj, idx_frame: idx_frame + seq_length + 1, :4])
            tmp_bbox[:, 0] = (
                tmp_bbox[:, 0] / raw_imgs.shape[2] * width).astype(int)
            tmp_bbox[:, 1] = (
                tmp_bbox[:, 1] / raw_imgs.shape[1] * height).astype(int)
            tmp_bbox[:, 2] = (
                tmp_bbox[:, 2] / raw_imgs.shape[2] * width).astype(int)
            tmp_bbox[:, 3] = (
                tmp_bbox[:, 3] / raw_imgs.shape[1] * height).astype(int)

            for ii in xrange(seq_length + 1):
                batch_img[idx_sample, ii] = cv2.resize(
                    raw_imgs[idx_frame + ii], (width, height),
                    interpolation=cv2.INTER_CUBIC)

                gt_raw_heat_map[idx_sample, ii, tmp_bbox[ii, 1]: tmp_bbox[
                    ii, 3], tmp_bbox[ii, 0]: tmp_bbox[ii, 2]] = 1
                gt_heat_map[idx_sample, ii] = cv2.resize(
                    gt_raw_heat_map[idx_sample, ii], 
                    (feat_map_width, feat_map_height),
                    interpolation=cv2.INTER_NEAREST)
            idx_sample += 1

        # training for current batch
        feed_data = {tracking_model['imgs']: batch_img,
                     tracking_model['init_heat_map']: gt_heat_map[:, 0, :, :],
                     tracking_model['gt_heat_map']: gt_heat_map,
                     tracking_model['anneal_threshold']: [anneal_prob],
                     tracking_model['phase_train']: True}

        results = sess.run(node_list, feed_dict=feed_data)

        results_dict = {}
        for rr, name in zip(results, nodes_run):
            results_dict[name] = rr

        logp_logger_IOU.add(step + 1, results_dict['IOU_loss'])

        # display training statistics
        if (step + 1) % display_iter == 0:
            print "Train Step = {:06d} || IOU Loss = {}".format(
                step + 1, results_dict['IOU_loss'])

        # save model
        if (step + 1) % anneal_iter == 0:
            anneal_prob += 0.1
            anneal_prob = min(anneal_prob, 1.0)

        # save model
        if (step + 1) % snapshot_iter == 0:
            saver.save(sess, 'my_conv_lstm_tracker.ckpt')

        # draw bbox on selected data
        if (step + 1) % draw_iter == 0:
            log.error(batch_img.shape)
            pu.plot_thumbnails(
                draw_img_name[0], batch_img, axis=1, max_items_per_row=4)
            log.error(results_dict['predict_heat_map'].shape)
            pu.plot_thumbnails(
                draw_img_name[1], results_dict['predict_heat_map'], axis=1,
                max_items_per_row=4)
            pu.plot_thumbnails(
                draw_img_name[2], gt_heat_map, axis=1,
                max_items_per_row=4)
            log.error(('GT max min', gt_heat_map.max(), gt_heat_map.min()))

            # for i in xrange(num_valid_seq):
            #     ut.draw_sequence(i, draw_img_name[i], valid_video_seq, tracking_model, sess, seq_length, height, width)
            pass

        step += 1

    sess.close()
