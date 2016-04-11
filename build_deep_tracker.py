import tensorflow as tf
import nnlib as nn
import numpy as np

from grad_clip_optim import GradientClipOptimizer

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
		gt_bbox = tf.placeholder('float', [batch_size, 5])

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

		# define a MLP predicting the bounding box and flag
		num_mlp_layers = len(mlp_hidden_dim)
		mlp_dims = mlp_hidden_dim + [5]
		mlp_act = [tf.tanh] * num_mlp_layers + [None]
		mlp_dropout = None
		# mlp_dropout = [1.0 - mlp_dropout_ratio] * num_ctrl_mlp_layers

		mlp = nn.mlp(mlp_dims, mlp_act, add_bias=True, dropout_keep=mlp_dropout, phase_train=phase_train, wd=weight_decay)

		rnn_hidden_feat = tf.concat(0, rnn_hidden_feat)
		predict_bbox = mlp(rnn_hidden_feat)
		predict_bbox = predict_bbox[-1]
		model['predict_bbox'] = predict_bbox[:, 0:3]
		model['predict_score'] = predict_bbox[:, 4]

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
