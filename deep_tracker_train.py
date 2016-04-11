import cslab_environ

import tensorflow as tf
import numpy as np
import sharded_hdf5 as sh
import progress_bar as pb

import cv2
import math
import logger

from copy import deepcopy
from build_deep_tracker import build_tracking_model
from build_deep_tracker import compute_IOU

# from tud import get_dataset
from kitti import get_dataset

def next_batch(imgs, labels, scores, idx_sample, batch_size, num_train):
	if idx_sample >= num_train:
		raise NameError('Incorrect index of sample')
	
	current_batch_img = []
	current_batch_label = []
	current_batch_score = []

	if idx_sample + batch_size > num_train:
		current_batch_img = [ imgs[idx_sample : num_train], imgs[0 : batch_size - (num_train - idx_sample)] ]
		current_batch_label = [ labels[idx_sample : num_train], labels[0 : batch_size - (num_train - idx_sample)] ]
		current_batch_score = [ scores[idx_sample : num_train], scores[0 : batch_size - (num_train - idx_sample)] ]
	else:
		current_batch_img = imgs[idx_sample : idx_sample + batch_size]
		current_batch_label = labels[idx_sample : idx_sample + batch_size]
		current_batch_score = scores[idx_sample : idx_sample + batch_size]

	return current_batch_img, current_batch_label, current_batch_score

if __name__ == "__main__":

	# folder = '/ais/gobi4/rjliao/Projects/CSC2541/data/TUD/cvpr10_tud_stadtmitte'
	folder = '/ais/gobi3/u/mren/data/kitti/tracking/'
	device = '/gpu:3'
	
	max_iter = 1000	
	batch_size = 10  	# sequence length for training
	display_iter = 10
	snapshot_iter = 1000
	height = 128
	width = 448

	# read data
	# dataset = get_dataset(folder)
	dataset = get_dataset(folder, 'train')

	# setting model
	opt_tracking = {}
	opt_tracking['batch_size'] = batch_size
	opt_tracking['cnn_filter_size'] = [5, 5, 5]
	opt_tracking['cnn_num_channel'] = [64, 64, 32]
	opt_tracking['cnn_pool_size'] = [2, 2, 2]
	opt_tracking['img_channel'] = 3
	opt_tracking['use_batch_norm']= True
	opt_tracking['img_height'] = height
	opt_tracking['img_width'] = width
	opt_tracking['weight_decay'] = 1.0e-5
	opt_tracking['rnn_hidden_dim'] = 100
	opt_tracking['mlp_hidden_dim'] = [100, 100]
	opt_tracking['base_learn_rate'] = 1.0e-3
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
		global_train_step = 0

		for epoch_step in xrange(max_iter):

			for seq_num in xrange(len(reader)):

				seq_data = reader[seq_num]
				raw_imgs = seq_data['images_0']
				gt_bbox = seq_data['gt_bbox']
				
				num_obj = gt_bbox.shape[0]
				num_imgs = raw_imgs.shape[0]

				if num_obj < 1:
					continue

				for obj in xrange(num_obj):
					print "Epoch = %04d || Training %04d-th object in %04d-th sequence" % (epoch_step, obj+1, seq_num+1)

					# prepare input and output
					train_imgs = []
					train_gt_box = []
					train_gt_score = []
					skip_flag = True
					
					for ii in xrange(num_imgs):				
						# gt_bbox = [left top right bottom flag]
						if gt_bbox[obj, ii, 4] == 1:
							# if object disappears, then using the last visible bbox for training
							skip_flag = False

						if skip_flag == False:
							# extract raw image as input
							train_imgs.append(cv2.resize(raw_imgs[ii, :, :], (width, height), interpolation = cv2.INTER_CUBIC))

							# extract bbox and score as output
							tmp_box = deepcopy(gt_bbox[obj, ii, 0:4])
							tmp_box[0] = tmp_box[0] / raw_imgs.shape[2]
							tmp_box[1] = tmp_box[1] / raw_imgs.shape[1]
							tmp_box[2] = tmp_box[2] / raw_imgs.shape[2]
							tmp_box[3] = tmp_box[3] / raw_imgs.shape[1]
							train_gt_box.append(tmp_box)

							train_gt_score.append(gt_bbox[obj, ii, 4])

					# training for current sequence
					num_train_imgs = len(train_imgs)
					inner_max_iter = int(np.floor(num_train_imgs / batch_size))

					for step in xrange(inner_max_iter):
						idx_start = (step * batch_size) % num_train_imgs
						batch_img, batch_box, batch_score = next_batch(train_imgs, train_gt_box, train_gt_score, idx_start, batch_size, num_train_imgs)

						node_list = [tracking_model[i] for i in nodes_run]
						feed_data = {tracking_model['imgs']: batch_img, 
									 tracking_model['init_bbox']: [batch_box[0]], 
									 tracking_model['gt_bbox']: batch_box, 
									 tracking_model['gt_score']: zip(*[batch_score]), 
									 tracking_model['phase_train']: True}

						results = sess.run(node_list, feed_dict=feed_data)

						results_dict = {}
						for rr, name in zip(results, nodes_run):
							results_dict[name] = rr

						# print results_dict['predict_bbox']
						# print results_dict['predict_score']

						if step % display_iter == 0:
							print "Train Iter = %06d || Loss = %e" % (global_train_step+1, results_dict['IOU_loss'])

						if (global_train_step+1) % snapshot_iter == 0:
							saver.save(sess, 'my_deep_tracker.ckpt', global_step=step)

						global_train_step += 1

	sess.close()
