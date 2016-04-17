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
import progress_bar as pb

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

def collect_draw_sequence(draw_raw_imgs, draw_raw_gt_bbox, seq_length, height, width):

    count_draw = 0
    idx_draw_frame = 0
    skip_empty = True
    draw_imgs = []
    draw_gt_box = []

    while count_draw < seq_length:
        if draw_raw_gt_bbox[0, idx_draw_frame, 4] == 1:
            skip_empty = False

        if not skip_empty:
            draw_imgs.append(cv2.resize(draw_raw_imgs[idx_draw_frame], (width, height), interpolation = cv2.INTER_CUBIC))

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
    draw_raw_gt_bbox = draw_data['gt_bbox']    # gt_bbox = [left top right bottom flag]

    draw_imgs, draw_gt_box = collect_draw_sequence(draw_raw_imgs, draw_raw_gt_bbox, seq_length, height, width)

    feed_data = {tracking_model['imgs']: np.expand_dims(draw_imgs, 0), 
                 tracking_model['init_bbox']: np.expand_dims(draw_gt_box[0], 0), 
                 tracking_model['gt_bbox']: np.expand_dims(draw_gt_box, 0),                                     
                 tracking_model['phase_train']: False}

    draw_pred_bbox, draw_IOU_score = sess.run([tracking_model['predict_bbox'], tracking_model['IOU_score']], feed_dict=feed_data)

    draw_pred_bbox = np.squeeze(draw_pred_bbox)
    draw_IOU_score = np.squeeze(draw_IOU_score)

    num_col = 2
    num_row = seq_length / num_col
    plot_frame_with_bbox(draw_img_name, draw_imgs, draw_pred_bbox, draw_gt_box, draw_IOU_score, num_row, num_col)


if __name__ == "__main__":

    # folder = '/ais/gobi4/rjliao/Projects/CSC2541/data/TUD/cvpr10_tud_stadtmitte'
    folder = '/ais/gobi3/u/mren/data/kitti/tracking/'
    device = '/gpu:2'
    
    max_iter = 100000 
    batch_size = 20     
    display_iter = 10
    draw_iter = 20
    seq_length = 30     # sequence length for training
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
    opt_tracking['rnn_seq_len'] = seq_length
    opt_tracking['cnn_filter_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    opt_tracking['cnn_num_filter'] = [8, 8, 16, 16, 32, 32, 32, 32, 64, 64]
    opt_tracking['cnn_pool_size'] = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    opt_tracking['img_channel'] = 3
    opt_tracking['use_batch_norm']= True
    opt_tracking['img_height'] = height
    opt_tracking['img_width'] = width
    opt_tracking['weight_decay'] = 1.0e-7
    opt_tracking['rnn_hidden_dim'] = 100
    opt_tracking['base_learn_rate'] = 1.0e-3
    opt_tracking['learn_rate_decay_step'] = 5000
    opt_tracking['learn_rate_decay_rate'] = 0.96
    
    tracking_model = build_tracking_model(opt_tracking, device)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    nodes_run = ['train_step', 'IOU_loss', 'CE_loss', 'predict_bbox', 'predict_score']
    node_list = [tracking_model[i] for i in nodes_run]

    # training 
    with sh.ShardedFileReader(dataset) as reader:
        step = 0
        num_seq = len(reader)
        video_seq = []
        
        # read data
        cdf_seq = np.zeros(num_seq)
        total_count = 0     
        for idx_seq, seq_data in enumerate(pb.get_iter(reader)):
            video_seq.append(seq_data)

            if idx_seq == 0:
                cdf_seq[idx_seq] = seq_data['images_0'].shape[0]
            else:
                cdf_seq[idx_seq] = cdf_seq[idx_seq-1] + seq_data['images_0'].shape[0]

            total_count += seq_data['images_0'].shape[0]

        cdf_seq /= total_count

        # training loop
        while step < max_iter:
            idx_sample = 0
            batch_img = []
            init_box = []
            batch_box = []
            batch_score = []

            while idx_sample < batch_size:
                # sample sequence based on the proportion of its length    
                rand_val = np.random.rand()
                idx_boolean = np.logical_and(rand_val < cdf_seq, rand_val > np.concatenate(([0], cdf_seq[:-1])))
                idx_video = [i for i, elem in enumerate(idx_boolean) if elem]
                seq_data = video_seq[idx_video[0]]

                raw_imgs = seq_data['images_0']
                gt_bbox = seq_data['gt_bbox']    # gt_bbox = [left top right bottom flag]
                num_obj = gt_bbox.shape[0]
                num_imgs = raw_imgs.shape[0]

                if num_obj < 1:
                    continue

                idx_obj = np.random.randint(num_obj)
                idx_frame = np.random.randint(num_imgs - seq_length + 1)

                current_seq = []
                for ii in xrange(seq_length):
                    current_seq.append(cv2.resize(raw_imgs[idx_frame + ii], (width, height), interpolation = cv2.INTER_CUBIC))

                batch_img.append(current_seq)

                tmp_box = np.array(gt_bbox[idx_obj, idx_frame : idx_frame + seq_length, :4])
                tmp_box[:, 0] = tmp_box[:, 0] / raw_imgs.shape[2] * width
                tmp_box[:, 1] = tmp_box[:, 1] / raw_imgs.shape[1] * height
                tmp_box[:, 2] = tmp_box[:, 2] / raw_imgs.shape[2] * width
                tmp_box[:, 3] = tmp_box[:, 3] / raw_imgs.shape[1] * height
                    
                batch_box.append(tmp_box)
                init_box.append(tmp_box[0, :4])
                batch_score.append(gt_bbox[idx_obj, idx_frame : idx_frame + seq_length, 4])

                idx_sample += 1

            # training for current batch            
            feed_data = {tracking_model['imgs']: batch_img,
                         tracking_model['init_bbox']: init_box,
                         tracking_model['gt_bbox']: batch_box,
                         tracking_model['gt_score']: batch_score,
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
                draw_sequence(0, draw_img_name_0, video_seq, tracking_model, sess, seq_length, height, width)
                draw_sequence(3, draw_img_name_1, video_seq, tracking_model, sess, seq_length, height, width)
                draw_sequence(11, draw_img_name_2, video_seq, tracking_model, sess, seq_length, height, width)
                draw_sequence(14, draw_img_name_3, video_seq, tracking_model, sess, seq_length, height, width)
                draw_sequence(20, draw_img_name_4, video_seq, tracking_model, sess, seq_length, height, width)

            step += 1

    sess.close()
