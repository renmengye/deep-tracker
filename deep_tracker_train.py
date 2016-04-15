import cslab_environ

import tensorflow as tf
import numpy as np
import sharded_hdf5 as sh
import progress_bar as pb

import os
import cv2
import math
import logger

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from deep_dashboard_utils import log_register, TimeSeriesLogger

from build_deep_tracker import build_tracking_model
from build_deep_tracker import compute_IOU

# from tud import get_dataset
from kitti import get_dataset

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
                    color='b'))

            axarr[ii, jj].add_patch(patches.Rectangle(
                    (gt_bbox[idx][0], gt_bbox[idx][1]),
                    gt_bbox[idx][2] - gt_bbox[idx][0],
                    gt_bbox[idx][3] - gt_bbox[idx][1],
                    fill=False,
                    color='r'))

            axarr[ii, jj].text(0, 0, ("%5.2f" % iou[idx]),
               color=(0, 0, 0), size=8)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=80)
    plt.close('all')

def next_batch(imgs, labels, scores, idx_sample, batch_size, num_train):
    
    if idx_sample + batch_size > num_train:
        raise Exception('Incorrect index of sample')
    
    current_batch_img = imgs[idx_sample : idx_sample + batch_size]
    current_batch_label = labels[idx_sample : idx_sample + batch_size]
    current_batch_score = scores[idx_sample : idx_sample + batch_size]

    return current_batch_img, current_batch_label, current_batch_score

def collect_draw_sequence(draw_raw_imgs, draw_raw_gt_bbox):

    count_draw = 0
    idx_draw_frame = 0
    skip_empty = True
    draw_imgs = []
    draw_gt_box = []

    while count_draw < batch_size:
        if draw_raw_gt_bbox[0, idx_draw_frame, 4] == 1:
            skip_empty = False

        if not skip_empty:
            draw_imgs.append(cv2.resize(draw_raw_imgs[idx_draw_frame, :, :], (width, height), interpolation = cv2.INTER_CUBIC))

            tmp_box = draw_raw_gt_bbox[0, idx_draw_frame, :4]
            tmp_box[0] = tmp_box[0] / draw_raw_imgs.shape[2] * width
            tmp_box[1] = tmp_box[1] / draw_raw_imgs.shape[1] * height
            tmp_box[2] = tmp_box[2] / draw_raw_imgs.shape[2] * width
            tmp_box[3] = tmp_box[3] / draw_raw_imgs.shape[1] * height

            draw_gt_box.append(tmp_box)
            count_draw += 1

        idx_draw_frame += 1

    return draw_imgs, draw_gt_box

def draw_sequence(idx, draw_img_name, reader, tracking_model, sess):
    
    draw_data = reader[idx]
    draw_raw_imgs = draw_data['images_0']
    draw_raw_gt_bbox = draw_data['gt_bbox']    # gt_bbox = [left top right bottom flag]

    draw_imgs, draw_gt_box = collect_draw_sequence(draw_raw_imgs, draw_raw_gt_bbox)

    feed_data = {tracking_model['imgs']: draw_imgs, 
             tracking_model['init_bbox']: np.expand_dims(draw_gt_box[0], 0), 
             tracking_model['gt_bbox']: draw_gt_box,                                     
             tracking_model['phase_train']: False}

    draw_pred_bbox, draw_IOU_score = sess.run([tracking_model['predict_bbox'], tracking_model['IOU_score']], feed_dict=feed_data)

    plot_frame_with_bbox(draw_img_name, draw_imgs, draw_pred_bbox, draw_gt_box, draw_IOU_score, 8, 2)


if __name__ == "__main__":

    # folder = '/ais/gobi4/rjliao/Projects/CSC2541/data/TUD/cvpr10_tud_stadtmitte'
    folder = '/ais/gobi3/u/mren/data/kitti/tracking/'
    device = '/gpu:3'
    
    max_iter = 1000 
    batch_size = 16     # sequence length for training
    display_iter = 10
    draw_iter = 50
    snapshot_iter = 1000
    height = 128
    width = 448

    # logger for saving intermediate output
    model_id = 'deep-tracker-001'
    logs_folder = '/u/rjliao/public_html/results'
    logs_folder = os.path.join(logs_folder, model_id)

    logp_logger_IOU = TimeSeriesLogger(
        os.path.join(logs_folder, 'IOU_loss.csv'),
        labels=['IOU loss'],
        name='Traning IOU Loss',
        buffer_size=1)

    logp_logger_CE = TimeSeriesLogger(
        os.path.join(logs_folder, 'CE_loss.csv'),
        labels=['CE loss'],
        name='Traning CE Loss',
        buffer_size=1)    

    draw_img_name_0 = os.path.join(logs_folder, 'draw_bbox_0.png')
    draw_img_name_1 = os.path.join(logs_folder, 'draw_bbox_1.png')
    draw_img_name_2 = os.path.join(logs_folder, 'draw_bbox_2.png')
    draw_img_name_3 = os.path.join(logs_folder, 'draw_bbox_3.png')
    draw_img_name_4 = os.path.join(logs_folder, 'draw_bbox_4.png')

    if not os.path.exists(draw_img_name_0):
        log_register(draw_img_name_0, 'image', 'Tracking Bounding Box 1')

    if not os.path.exists(draw_img_name_1):
        log_register(draw_img_name_1, 'image', 'Tracking Bounding Box 2')

    if not os.path.exists(draw_img_name_2):
        log_register(draw_img_name_2, 'image', 'Tracking Bounding Box 3')  

    if not os.path.exists(draw_img_name_3):
        log_register(draw_img_name_3, 'image', 'Tracking Bounding Box 4')  

    if not os.path.exists(draw_img_name_4):
        log_register(draw_img_name_4, 'image', 'Tracking Bounding Box 5')                          

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
    opt_tracking['base_learn_rate'] = 1.0e-3
    opt_tracking['learn_rate_decay_step'] = 5000
    opt_tracking['learn_rate_decay_rate'] = 0.96
    
    tracking_model = build_tracking_model(opt_tracking, device)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    nodes_run = ['train_step', 'IOU_loss', 'CE_loss', 'predict_bbox', 'predict_score']
    # nodes_run = ['predict_bbox', 'predict_score']

    # training 
    with sh.ShardedFileReader(dataset) as reader:
        step = 0

        while step < max_iter:
            for seq_num in xrange(len(reader)):
                seq_data = reader[seq_num]
                raw_imgs = seq_data['images_0']
                gt_bbox = seq_data['gt_bbox']    # gt_bbox = [left top right bottom flag]
                num_obj = gt_bbox.shape[0]
                num_imgs = raw_imgs.shape[0]

                if num_obj < 1:
                    continue

                for obj in xrange(num_obj):
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

                        logp_logger_IOU.add(step, results_dict['IOU_loss'])
                        logp_logger_CE.add(step, results_dict['CE_loss'])

                        # display training statistics
                        if (step+1) % display_iter == 0:
                            print "Train Step = %06d || IOU Loss = %e || CE loss = %e" % (step+1, results_dict['IOU_loss'], results_dict['CE_loss'])

                        # save model
                        if (step+1) % snapshot_iter == 0:
                            saver.save(sess, 'my_deep_tracker.ckpt')

                        # draw bbox on selected data
                        if (step+1) % draw_iter == 0:
                            draw_sequence(0, draw_img_name_0, reader, tracking_model, sess)
                            draw_sequence(3, draw_img_name_1, reader, tracking_model, sess)
                            draw_sequence(11, draw_img_name_2, reader, tracking_model, sess)
                            draw_sequence(14, draw_img_name_3, reader, tracking_model, sess)
                            draw_sequence(20, draw_img_name_4, reader, tracking_model, sess)

                        step += 1

    sess.close()
