import cslab_environ

import tensorflow as tf
import numpy as np
import sharded_hdf5 as sh
import progress_bar as pb

import os
import cv2
import math
import logger

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from deep_dashboard_utils import log_register, TimeSeriesLogger

from build_deep_tracker import build_tracking_model
from build_deep_tracker import compute_IOU

# from tud import get_dataset
from kitti import get_dataset

def next_batch(imgs, labels, scores, idx_sample, batch_size, num_train):
    
    if idx_sample + batch_size > num_train:
        raise Exception('Incorrect index of sample')
    
    current_batch_img = imgs[idx_sample : idx_sample + batch_size]
    current_batch_label = labels[idx_sample : idx_sample + batch_size]
    current_batch_score = scores[idx_sample : idx_sample + batch_size]

    return current_batch_img, current_batch_label, current_batch_score

if __name__ == "__main__":

    # folder = '/ais/gobi4/rjliao/Projects/CSC2541/data/TUD/cvpr10_tud_stadtmitte'
    folder = '/ais/gobi3/u/mren/data/kitti/tracking/'
    device = '/gpu:3'
    
    max_iter = 1000 
    batch_size = 10     # sequence length for training
    display_iter = 10
    snapshot_iter = 1000
    height = 128
    width = 448

    # logger for saving intermediate output
    model_id = 'deep-tracker-001'
    logs_folder = '/u/rjliao/public_html/results'
    logs_folder = os.path.join(logs_folder, model_id)

    logp_logger = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'),
        labels=['IOU loss', 'CE loss'],
        name='Traning Loss',
        buffer_size=1)

    # read data
    # dataset = get_dataset(folder)
    dataset = get_dataset(folder, 'train')

    # setting model
    opt_tracking = {}
    opt_tracking['batch_size'] = batch_size
    opt_tracking['cnn_filter_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    opt_tracking['cnn_num_channel'] = [8, 8, 16, 16, 32, 32, 32, 32, 64, 64]
    opt_tracking['cnn_pool_size'] = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    opt_tracking['img_channel'] = 3
    opt_tracking['use_batch_norm']= True
    opt_tracking['img_height'] = height
    opt_tracking['img_width'] = width
    opt_tracking['weight_decay'] = 1.0e-7
    opt_tracking['rnn_hidden_dim'] = 100
    opt_tracking['mlp_hidden_dim'] = [100, 100]
    opt_tracking['base_learn_rate'] = 1.0e-4
    opt_tracking['learn_rate_decay_step'] = 10000
    opt_tracking['learn_rate_decay_rate'] = 0.96
    
    tracking_model = build_tracking_model(opt_tracking, device)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    nodes_run = ['train_step', 'IOU_loss', 'CE_loss', 'predict_bbox', 'predict_score']
    # nodes_run = ['predict_bbox', 'predict_score']

    with sh.ShardedFileReader(dataset) as reader:       
        # training 
        step = 0

        while step < max_iter:

            # for seq_num in xrange(len(reader)):
            for seq_num in xrange(1):
                seq_data = reader[seq_num]
                raw_imgs = seq_data['images_0']
                gt_bbox = seq_data['gt_bbox']    # gt_bbox = [left top right bottom flag]
                num_obj = gt_bbox.shape[0]
                num_imgs = raw_imgs.shape[0]

                if num_obj < 1:
                    continue

                # for obj in xrange(num_obj):
                for obj in xrange(1):
                    # prepare input and output
                    train_imgs = []
                    train_gt_box = []
                    train_gt_score = []
                    
                    idx_visible = np.nonzero(gt_bbox[obj, :, 4])
                    idx_visible_start = np.min(idx_visible)
                    idx_visible_end = np.max(idx_visible)
                    num_train_imgs = idx_visible_end - idx_visible_start

                    for ii in xrange(idx_visible_start, idx_visible_end):
                        # extract raw image as input
                        train_imgs.append(cv2.resize(raw_imgs[ii, :, :], (width, height), interpolation = cv2.INTER_CUBIC))                        
                        
                        # extract bbox and score as output
                        tmp_box = gt_bbox[obj, ii, :4]
                        tmp_box[0] = tmp_box[0] / raw_imgs.shape[2] * width
                        tmp_box[1] = tmp_box[1] / raw_imgs.shape[1] * height
                        tmp_box[2] = tmp_box[2] / raw_imgs.shape[2] * width
                        tmp_box[3] = tmp_box[3] / raw_imgs.shape[1] * height
                        train_gt_box.append(tmp_box)

                        train_gt_score.append(gt_bbox[obj, ii, 4])

                    # training for current sequence                
                    for idx_start in xrange(num_train_imgs - batch_size):
                        batch_img, batch_box, batch_score = next_batch(train_imgs, train_gt_box, train_gt_score, idx_start, batch_size, num_train_imgs)

                        node_list = [tracking_model[i] for i in nodes_run]
                        feed_data = {tracking_model['imgs']: batch_img, 
                                     tracking_model['init_bbox']: np.expand_dims(batch_box[0], 0), 
                                     tracking_model['gt_bbox']: batch_box, 
                                     tracking_model['gt_score']: np.expand_dims(batch_score, 1), 
                                     tracking_model['phase_train']: True}

                        results = sess.run(node_list, feed_dict=feed_data)

                        results_dict = {}
                        for rr, name in zip(results, nodes_run):
                            results_dict[name] = rr

                        logp_logger.add(step, [results_dict['IOU_loss'], results_dict['CE_loss']])

                        if (step+1) % display_iter == 0:
                            print "Train Step = %06d || IOU Loss = %e || CE loss = %e" % (step+1, results_dict['IOU_loss'], results_dict['CE_loss'])

                        if (step+1) % snapshot_iter == 0:
                            saver.save(sess, 'my_deep_tracker.ckpt')

                        step += 1

    sess.close()
