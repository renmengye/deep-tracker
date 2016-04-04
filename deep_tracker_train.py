import cslab_environ

import tensorflow as tf
import nnlib as nn
import numpy as np
import logger

from grad_clip_optim import GradientClipOptimizer
from tud import get_dataset

def get_device_fn(device):
	"""Choose device for different ops."""
	OPS_ON_CPU = set(['ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'CumMin', 'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense', 'BatchMatMul'])

	def _device_fn(op):
		if op.type in OPS_ON_CPU:
			return "/cpu:0"
		else:
			# Other ops will be placed on GPU if available, otherwise CPU.
			return device

	return _device_fn

def build_tracking_model(opt, device='/cpu:0'):
	model = {}

	batch_size = opt['batch_size']
	cnn_filter_size = opt['cnn_filter_size']
	cnn_num_channel = opt['cnn_num_channel']
	cnn_pool_size = opt['cnn_pool_size']
	img_num_channel = opt['img_channel']
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
		imgs = tf.placeholder('float', [batch_size, height, width, img_num_channel])
		gt_bbox = tf.placeholder('float', [batch_size, 4])
		# img_shape = tf.shape(imgs)
		# batch_size = img_shape[0]
		model['imgs'] = imgs
		model['gt_bbox'] = gt_bbox
		model['phase_train'] = phase_train

		# define a CNN model
		cnn_filter = cnn_filter_size
		cnn_nlayer = len(cnn_filter)
		cnn_channel = [img_num_channel] + cnn_num_channel
		cnn_pool = cnn_pool_size
		cnn_act = [tf.nn.relu] * cnn_nlayer
		cnn_use_bn = [use_bn] * cnn_nlayer

		# if pretrain_ccnn:
		#     h5f = h5py.File(pretrain_ccnn, 'r')
		#     ccnn_init_w = [{'w': h5f['cnn_w_{}'.format(ii)][:],
		#                     'b': h5f['cnn_b_{}'.format(ii)][:]}
		#                    for ii in xrange(ccnn_nlayers)]
		#     ccnn_frozen = True
		# else:
		#     ccnn_init_w = None
		#     ccnn_frozen = False

		cnn_model = nn.cnn(cnn_filter, cnn_channel, cnn_pool, cnn_act,
		              cnn_use_bn, phase_train=phase_train, wd=weight_decay)
		
		h_cnn = cnn_model(imgs)	# h_cnn is a list and stores the output of every layer
		cnn_output = h_cnn[-1]
		model['cnn_output'] = cnn_output

		# define a RNN(LSTM) model
		cnn_subsample = np.array(cnn_pool).prod()
		rnn_h = height / cnn_subsample
		rnn_w = width / cnn_subsample
		rnn_dim = cnn_channel[-1]	    
		rnn_inp_dim = rnn_h * rnn_w * rnn_dim	

		rnn_state = [None] * (batch_size + 1)
		rnn_state[-1] = tf.zeros([1, rnn_hidden_dim * 2])
		rnn_hidden_feat = [None] * batch_size

		rnn_cell = nn.lstm(rnn_inp_dim, rnn_hidden_dim, wd=weight_decay)

		cnn_feat = tf.split(0, batch_size, cnn_output)

		for tt in xrange(batch_size):
			cnn_feat[tt] = tf.reshape(cnn_feat[tt], [1, rnn_inp_dim])
			rnn_state[tt], _, _, _ = rnn_cell(cnn_feat[tt], rnn_state[tt - 1])
			rnn_hidden_feat[tt] = tf.slice(rnn_state[tt], [0, rnn_hidden_dim], [-1, rnn_hidden_dim])

		# define a MLP predicting the bounding box
		num_mlp_layers = len(mlp_hidden_dim)
		mlp_dims = mlp_hidden_dim + [4]
		mlp_act = [tf.nn.relu] * num_mlp_layers + [None]
		mlp_dropout = None
		# mlp_dropout = [1.0 - mlp_dropout_ratio] * num_ctrl_mlp_layers

		mlp = nn.mlp(mlp_dims, mlp_act, add_bias=True, dropout_keep=mlp_dropout, phase_train=phase_train, wd=weight_decay)		 

		rnn_hidden_feat = tf.concat(0, rnn_hidden_feat)
		predict_bbox = mlp(rnn_hidden_feat)
		predict_bbox = predict_bbox[-1]
		model['predict_bbox'] = predict_bbox

		# we need to try different loss functions here
		l2_loss = tf.nn.l2_loss(predict_bbox - gt_bbox)
		model['loss'] = l2_loss
		
		global_step = tf.Variable(0.0)
		eps = 1e-7

		learn_rate = tf.train.exponential_decay(
			base_learn_rate, global_step, learn_rate_decay_step,
			learn_rate_decay_rate, staircase=True)
		model['learn_rate'] = learn_rate

		train_step = GradientClipOptimizer(
			tf.train.AdamOptimizer(learn_rate, epsilon=eps),
			clip=1.0).minimize(l2_loss, global_step=global_step)
		model['train_step'] = train_step

	return model

def next_batch(imgs, labels, idx_sample, batch_size, num_train):
	if idx_sample >= num_train:
		raise NameError('Incorrect index of sample')
	
	current_batch_img = []
	current_batch_label = []

	if idx_sample + batch_size > num_train:
		current_batch_img = [ imgs[idx_sample : num_train], imgs[0 : batch_size - (num_train - idx_sample)] ]
		current_batch_label = [ labels[idx_sample : num_train], labels[0 : batch_size - (num_train - idx_sample)] ]	
	else:
		current_batch_img = imgs[idx_sample : idx_sample + batch_size]
		current_batch_label = labels[idx_sample : idx_sample + batch_size]

	return current_batch_img, current_batch_label

if __name__ == "__main__":

	folder = '/ais/gobi4/rjliao/Projects/CSC2541/data/TUD/cvpr10_tud_stadtmitte'
	device = '/gpu:0'
	num_train = 100
	max_iter = 1000	
	batch_size = 10
	test_iter = 100
	display_iter = 10
	snapshot_iter = 1000

	# read data
	dataset = get_dataset(folder)

	# we are now focusing on the 3rd person
	gt_bbox = dataset['gt_bbox'][2, :, :]
	imgs = dataset['images']
	
	train_imgs = imgs[0:num_train]
	test_imgs = imgs[num_train:]
	train_gt_box = gt_bbox[0:num_train]
	test_gt_box = gt_bbox[num_train:]

	# normalize ground truth bbox [h w x y]
	gt_bbox[:, 0] = gt_bbox[:, 0] / imgs.shape[1]
	gt_bbox[:, 1] = gt_bbox[:, 1] / imgs.shape[2]
	gt_bbox[:, 2] = gt_bbox[:, 2] / imgs.shape[2]
	gt_bbox[:, 3] = gt_bbox[:, 3] / imgs.shape[1]

	# setting model
	opt_tracking = {}
	opt_tracking['batch_size'] = batch_size
	opt_tracking['cnn_filter_size'] = [5, 5, 5]
	opt_tracking['cnn_num_channel'] = [64, 64, 32]
	opt_tracking['cnn_pool_size'] = [2, 2, 2]
	opt_tracking['img_channel'] = 3
	opt_tracking['use_batch_norm']= True
	opt_tracking['img_height'] = imgs.shape[1]
	opt_tracking['img_width'] = imgs.shape[2]
	opt_tracking['weight_decay'] = 1.0e-5
	opt_tracking['rnn_hidden_dim'] = 100
	opt_tracking['mlp_hidden_dim'] = [100, 100]
	opt_tracking['base_learn_rate'] = 1.0e-2
	opt_tracking['learn_rate_decay_step'] = 10000
	opt_tracking['learn_rate_decay_rate'] = 0.96
	
	tracking_model = build_tracking_model(opt_tracking, device)

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	saver = tf.train.Saver()

	nodes_run = ['train_step', 'loss', 'predict_bbox']
	
	# training 
	for step in xrange(max_iter):
		idx_start = (step * batch_size) % num_train
		batch_img, batch_box = next_batch(train_imgs, train_gt_box, idx_start, batch_size, num_train)

		node_list = [tracking_model[i] for i in nodes_run]
		feed_data = {tracking_model['imgs']: batch_img, tracking_model['gt_bbox']: batch_box[:, 0:4], tracking_model['phase_train']: True}

		results = sess.run(node_list, feed_dict=feed_data)

		results_dict = {}
		for rr, name in zip(results, nodes_run):
			results_dict[name] = rr

		if step % display_iter == 0:
			print 'Train Iter = %06d || Loss = %e' % (step, results_dict['loss'])

		if step % snapshot_iter == 0:
			saver.save(sess, 'my_deep_tracker', global_step=step)

		if step % test_iter == 0:
			test_loss = 0
			num_test = test_imgs.shape[0]
			num_test_iter = num_test / batch_size
			print 'Number of test samples = %d, Number of test iters = %d' % (num_test, num_test_iter)

			for tt in xrange(num_test_iter):
				idx_start = (tt * batch_size) % num_test
				batch_img, batch_box = next_batch(test_imgs, test_gt_box, idx_start, batch_size, num_test)

				test_data = {tracking_model['imgs']: batch_img, tracking_model['gt_bbox']: batch_box[:, 0:4], tracking_model['phase_train']: False}
				results = sess.run(tracking_model['loss'], feed_dict=test_data)

				test_loss += results

			print 'The test loss = %e' % (test_loss / num_test_iter)

	sess.close()
