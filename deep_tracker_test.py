import cslab_environ

import tensorflow as tf
import numpy as np
import sharded_hdf5 as sh
import progress_bar as pb

import os
import cv2
import math
import logger

import deep_tracker_utils as ut
import build_deep_tracker as dt

import progress_bar as pb
from deep_dashboard_utils import log_register, TimeSeriesLogger

from kitti import get_dataset

if __name__ == "__main__":
    folder = '/ais/gobi3/u/mren/data/kitti/tracking/'
    save_path = '/ais/gobi4/rjliao/Projects/Kitti/tracking_res'
    device = '/gpu:2'

    max_iter = 100000
    batch_size = 1
    display_iter = 10
    draw_iter = 50
    seq_length = 40     # sequence length for training
    snapshot_iter = 500
    anneal_prob = 1     # set to 1 for testing
    valid_iou_iter = 500
    height = 128
    width = 448
    img_channel = 3
    num_train_seq = 16
    rnn_hidden_dim = 128

    # read data
    valid_video_seq = []
    num_valid_seq = 0
    train_data_full = get_dataset(folder, 'train')

    with sh.ShardedFileReader(train_data_full) as reader:
        num_seq = len(reader)

        for idx_seq, seq_data in enumerate(pb.get_iter(reader)):
            if idx_seq >= num_train_seq:
                if seq_data['gt_bbox'].shape[0] > 0:
                    valid_video_seq.append(seq_data)
                    num_valid_seq += 1

    # setting model
    opt_tracking = {}
    opt_tracking['rnn_seq_len'] = seq_length
    # opt_tracking['cnn_filter_size'] = [3, 3, 3, 3, 3, 3, 3, 3]
    # opt_tracking['cnn_num_filter'] = [16, 16, 32, 32, 64, 64, 96, 96]
    # opt_tracking['cnn_pool_size'] = [1, 2, 1, 2, 1, 2, 1, 2]

    opt_tracking['cnn_filter_size'] = [3, 3, 3, 3, 3, 3]
    opt_tracking['cnn_num_filter'] = [8, 8, 16, 16, 32, 32]
    opt_tracking['cnn_pool_size'] = [1, 2, 1, 2, 1, 2]

    opt_tracking['img_channel'] = img_channel
    opt_tracking['use_batch_norm'] = True
    opt_tracking['img_height'] = height
    opt_tracking['img_width'] = width
    opt_tracking['weight_decay'] = 1.0e-7
    opt_tracking['rnn_hidden_dim'] = rnn_hidden_dim
    opt_tracking['base_learn_rate'] = 1.0e-3
    opt_tracking['learn_rate_decay_step'] = 1000
    opt_tracking['learn_rate_decay_rate'] = 0.96
    # opt_tracking['pretrain_model_filename'] = "/ais/gobi3/u/mren/results/deep-tracker/detector-20160417231457/weights.h5"
    opt_tracking[
        'pretrain_model_filename'] = "/ais/gobi3/u/mren/results/img-count/fg_segm-20160419004323/weights.h5"
    opt_tracking['is_pretrain'] = True

    tracking_model = dt.build_tracking_model(opt_tracking, device)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    saver.restore(
        sess, "/ais/gobi4/rjliao/Projects/Kitti/tracking_models/deep_tracker_0018000.ckpt")

    nodes_run = ['IOU_score', 'predict_bbox',
                 'predict_score', 'final_rnn_state']
    node_list = [tracking_model[i] for i in nodes_run]

    # testing loop
    for idx_seq, seq_data in enumerate(valid_video_seq):
        batch_img = np.zeros(
            [batch_size, seq_length + 1, height, width, img_channel])
        init_box = np.zeros([batch_size, 4])
        batch_box = np.zeros([batch_size, seq_length + 1, 4])
        batch_score = np.zeros([batch_size, seq_length + 1])

        raw_imgs = seq_data['images_0']
        gt_bbox = seq_data['gt_bbox']
        num_obj = gt_bbox.shape[0]
        num_frames = raw_imgs.shape[0]

        for idx_obj in xrange(num_obj):
            start_idx_frame = 0
            last_rnn_state = []
            pred_bbox = []
            pred_score = []
            IOU_score = []

            # find the first frame
            for ii in xrange(num_frames):
                if gt_bbox[idx_obj, ii, 4] == 1:
                    start_idx_frame = ii
                    break

            idx_frame = start_idx_frame
            num_test_pass = int(
                np.ceil((num_frames - start_idx_frame) / (seq_length + 1)))

            for ii in xrange(num_test_pass):
                for jj in xrange(seq_length + 1):
                    if idx_frame + jj < num_frames:
                        batch_img[0, jj] = cv2.resize(
                            raw_imgs[idx_frame + jj], (width, height), interpolation=cv2.INTER_CUBIC)
                    else:
                        batch_img[0, jj] = cv2.resize(
                            raw_imgs[num_frames - 1], (width, height), interpolation=cv2.INTER_CUBIC)

                tmp_box = np.array(
                    gt_bbox[idx_obj, idx_frame: idx_frame + seq_length + 1, :4])
                tmp_box[:, 0] = tmp_box[:, 0] / raw_imgs.shape[2] * width
                tmp_box[:, 1] = tmp_box[:, 1] / raw_imgs.shape[1] * height
                tmp_box[:, 2] = tmp_box[:, 2] / raw_imgs.shape[2] * width
                tmp_box[:, 3] = tmp_box[:, 3] / raw_imgs.shape[1] * height

                batch_box[0] = tmp_box
                init_box[0] = tmp_box[0, :]
                batch_score[0] = gt_bbox[
                    idx_obj, idx_frame: idx_frame + seq_length + 1, 4]

                if ii == 0:
                    init_rnn_state = np.concatenate(1, [dt.inverse_transform_box(
                        init_box, height, width), np.zeros([batch_size, rnn_hidden_dim * 2 - 4])])
                else:
                    init_rnn_state = last_rnn_state

                # test a sequence
                feed_data = {tracking_model['imgs']: batch_img,
                             tracking_model['init_bbox']: init_box,
                             tracking_model['gt_bbox']: batch_box,
                             tracking_model['gt_score']: batch_score,
                             tracking_model['anneal_threshold']: [anneal_prob],
                             tracking_model['init_rnn_state']: init_rnn_state,
                             tracking_model['phase_train']: True}

                results = sess.run(node_list, feed_dict=feed_data)

                results_dict = {}
                for rr, name in zip(results, nodes_run):
                    results_dict[name] = rr

                last_rnn_state = results_dict['final_rnn_state']
                pred_bbox.append(results_dict['predict_bbox'])
                pred_score.append(results_dict['predict_score'])
                IOU_score.append(results_dict['IOU_score'])

                # save results
                print IOU_score

                idx_frame += (seq_length + 1)

            # print image
            pred_bbox = np.concatenate(0, pred_bbox)
            pred_score = np.concatenate(0, pred_score)

            plot_batch_frame_with_bbox(("valid_seq_%03d_obj_%03d" % (idx_seq, idx_obj)),
                                       raw_imgs[start_idx_frame:], pred_bbox, gt_bbox[start_idx_frame:], pred_score)

    sess.close()
