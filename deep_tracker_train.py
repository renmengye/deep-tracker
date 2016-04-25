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
    save_path = '/ais/gobi4/rjliao/Projects/Kitti/tracking_models'
    device = '/gpu:2'

    max_iter = 100000
    batch_size = 10
    display_iter = 10
    draw_iter = 50
    seq_length = 40     # sequence length for training
    snapshot_iter = 500
    anneal_iter = 1000
    valid_iou_iter = 500
    height = 128
    width = 448
    img_channel = 3
    resume_training = False
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
    model_id = 'deep-tracker-002'
    logs_folder = '/u/rjliao/public_html/results'
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
    
    draw_img_name = []

    for i in xrange(num_valid_seq):
        draw_img_name.append(os.path.join(logs_folder, 'draw_bbox_{}.png'.format(i)))

        if not os.path.exists(draw_img_name[i]):
            log_register(draw_img_name[i], 'image', 'Tracking Bounding Box {}'.format(i))

    # setting model
    opt_tracking = {}
    opt_tracking['rnn_seq_len'] = seq_length
    # opt_tracking['cnn_filter_size'] = [3, 3, 3, 3, 3, 3, 3, 3]
    # opt_tracking['cnn_num_filter'] = [16, 16, 32, 32, 64, 64, 96, 96]
    # opt_tracking['cnn_pool_size'] = [1, 2, 1, 2, 1, 2, 1, 2]

    opt_tracking['cnn_filter_size'] = [3,3,3,3,3,3]
    opt_tracking['cnn_num_filter'] = [8,8,16,16,32,32]
    opt_tracking['cnn_pool_size'] = [1, 2, 1, 2, 1, 2]

    opt_tracking['img_channel'] = img_channel
    opt_tracking['use_batch_norm'] = True
    opt_tracking['img_height'] = height
    opt_tracking['img_width'] = width
    opt_tracking['weight_decay'] = 1.0e-7
    opt_tracking['rnn_hidden_dim'] = 128
    opt_tracking['base_learn_rate'] = 1.0e-3
    opt_tracking['learn_rate_decay_step'] = 1000
    opt_tracking['learn_rate_decay_rate'] = 0.96
    # opt_tracking['pretrain_model_filename'] = "/ais/gobi3/u/mren/results/deep-tracker/detector-20160417231457/weights.h5"
    opt_tracking['pretrain_model_filename'] = "/ais/gobi3/u/mren/results/img-count/fg_segm-20160419004323/weights.h5"
    opt_tracking['is_pretrain'] = True

    tracking_model = dt.build_tracking_model(opt_tracking, device)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    if resume_training:
        saver.restore(sess, "my_deep_tracker.ckpt")

    nodes_run = ['train_step', 'IOU_loss', 'IOU_score', 'CE_loss', 'predict_bbox', 'predict_score']
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
        init_box = np.zeros([batch_size, 4])
        batch_box = np.zeros([batch_size, seq_length + 1, 4])
        batch_score = np.zeros([batch_size, seq_length + 1])

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

            for ii in xrange(seq_length + 1):
                batch_img[idx_sample, ii] = cv2.resize(
                    raw_imgs[idx_frame + ii], (width, height), interpolation=cv2.INTER_CUBIC)

            tmp_box = np.array(
                gt_bbox[idx_obj, idx_frame: idx_frame + seq_length + 1, :4])
            tmp_box[:, 0] = tmp_box[:, 0] / raw_imgs.shape[2] * width
            tmp_box[:, 1] = tmp_box[:, 1] / raw_imgs.shape[1] * height
            tmp_box[:, 2] = tmp_box[:, 2] / raw_imgs.shape[2] * width
            tmp_box[:, 3] = tmp_box[:, 3] / raw_imgs.shape[1] * height

            batch_box[idx_sample] = tmp_box
            init_box[idx_sample] = tmp_box[0, :]
            batch_score[idx_sample] = gt_bbox[
                idx_obj, idx_frame: idx_frame + seq_length + 1, 4]

            idx_sample += 1

        # training for current batch
        feed_data = {tracking_model['imgs']: batch_img,
                     tracking_model['init_bbox']: init_box,
                     tracking_model['gt_bbox']: batch_box,
                     tracking_model['gt_score']: batch_score,
                     tracking_model['anneal_threshold']: [anneal_prob],
                     tracking_model['phase_train']: True}

        results = sess.run(node_list, feed_dict=feed_data)

        results_dict = {}
        for rr, name in zip(results, nodes_run):
            results_dict[name] = rr

        logp_logger_IOU.add(step + 1, results_dict['IOU_loss'])        
        logp_logger_CE.add(step + 1, results_dict['CE_loss'])

        # display training statistics
        if (step + 1) % display_iter == 0:
            print "Train Step = %06d || IOU Loss = %e || CE loss = %e" % (step + 1, results_dict['IOU_loss'], results_dict['CE_loss'])

        # save model
        if (step + 1) % anneal_iter == 0:
            anneal_prob += 0.1
            anneal_prob = min(anneal_prob, 1.0)

        # save model
        if (step + 1) % snapshot_iter == 0:
            saver.save(sess, os.path.join(save_path, ("deep_tracker_%07d.ckpt" % (step + 1))))

        # draw bbox on selected data
        if (step + 1) % draw_iter == 0:
            for i in xrange(num_valid_seq):            
                ut.draw_sequence(i, draw_img_name[i], valid_video_seq, tracking_model, sess, seq_length, height, width)

        step += 1

    sess.close()
