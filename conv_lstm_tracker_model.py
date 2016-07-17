import cslab_environ

import numpy as np
import tfplus
import tensorflow as tf

tfplus.cmd_args.add('ct:inp_depth', 'int', 3)
tfplus.cmd_args.add('ct:timespan', 'int', 3)
tfplus.cmd_args.add('ct:weight_decay', 'float', 5e-5)
tfplus.cmd_args.add('ct:res_net_layers', 'list<int>', [3, 3, 3, 3])
tfplus.cmd_args.add('ct:res_net_strides', 'list<int>', [1, 2, 2, 2])
tfplus.cmd_args.add('ct:res_net_channels', 'list<int>', [16, 16, 32, 64, 96])
tfplus.cmd_args.add('ct:conv_lstm_hid_depth', 'int', 256)
tfplus.cmd_args.add('ct:conv_lstm_filter_size', 'int', 3)
tfplus.cmd_args.add('ct:res_net_bottleneck', 'bool', False)
tfplus.cmd_args.add('ct:res_net_shortcut', 'str', 'identity')
tfplus.cmd_args.add('ct:full_res', 'bool', False)
tfplus.cmd_args.add('ct:base_learn_rate', 'float', 1e-3)
tfplus.cmd_args.add('ct:learn_rate_decay', 'float', 1e-1)
tfplus.cmd_args.add('ct:steps_per_learn_rate_decay', 'int', 10000)
tfplus.cmd_args.add('ct:switch_offset', 'int', 1000)
tfplus.cmd_args.add('ct:steps_per_switch_decay', 'int', 1000)
tfplus.cmd_args.add('ct:switch_decay', 'float', 0.9)
tfplus.cmd_args.add('ct:clip_gradient', 'float', 1.0)


class ConvLSTMTrackerModel(tfplus.nn.Model):
    """A model for ConvLSTM tracking."""

    def __init__(self):
        super(ConvLSTMTrackerModel, self).__init__()
        self.register_option('ct:inp_depth')
        self.register_option('ct:timespan')
        self.register_option('ct:weight_decay')
        self.register_option('ct:res_net_layers')
        self.register_option('ct:res_net_strides')
        self.register_option('ct:res_net_channels')
        self.register_option('ct:res_net_bottleneck')
        self.register_option('ct:res_net_shortcut')
        self.register_option('ct:full_res')
        self.register_option('ct:conv_lstm_hid_depth')
        self.register_option('ct:conv_lstm_filter_size')
        self.register_option('ct:switch_offset')
        self.register_option('ct:steps_per_switch_decay')
        self.register_option('ct:switch_decay')
        self.register_option('ct:base_learn_rate')
        self.register_option('ct:learn_rate_decay')
        self.register_option('ct:steps_per_learn_rate_decay')
        self.register_option('ct:clip_gradient')
        pass

    def init_default_options(self):
        pass

    def build_input(self):
        self.init_default_options()
        inp_depth = self.get_option('ct:inp_depth')
        timespan = self.get_option('ct:timespan')
        x = self.add_input_var(
            'x', [None, timespan, None, None, inp_depth])
        bbox_gt = self.add_input_var(
            'bbox_gt', [None, timespan, 4])
        phase_train = self.add_input_var('phase_train', None, 'bool')
        results = {
            'x': x,
            'bbox_gt': bbox_gt,
            'phase_train': phase_train
        }
        return results

    def init_var(self):
        inp_depth = self.get_option('ct:inp_depth')
        wd = self.get_option('ct:weight_decay')
        res_net_layers = self.get_option('ct:res_net_layers')
        res_net_channels = self.get_option('ct:res_net_channels')
        res_net_bottleneck = self.get_option('ct:res_net_bottleneck')
        res_net_shortcut = self.get_option('ct:res_net_shortcut')
        res_net_strides = self.get_option('ct:res_net_strides')

        self.conv1 = tfplus.nn.Conv2DW(
            f=7, ch_in=inp_depth * 2 + 1, ch_out=res_net_channels[0], stride=2,
            wd=wd, scope='conv', bias=False)

        self.res_net = tfplus.nn.ResNet(layers=res_net_layers,
                                        bottleneck=res_net_bottleneck,
                                        shortcut=res_net_shortcut,
                                        channels=res_net_channels,
                                        strides=res_net_strides,
                                        wd=wd)

        conv_lstm_filter_size = self.get_option('ct:conv_lstm_filter_size')
        conv_lstm_hid_depth = self.get_option('ct:conv_lstm_hid_depth')
        self.conv_lstm = tfplus.nn.ConvLSTM(filter_size=conv_lstm_filter_size,
                                            inp_depth=res_net_channels[-1],
                                            hid_depth=conv_lstm_hid_depth,
                                            wd=wd)
        self.post_cnn = tfplus.nn.CNN([1], [conv_lstm_hid_depth, 1], [1],
                                      [tf.sigmoid], [False], wd=wd,
                                      scope='post_cnn')
        pass

    def get_idx_map(self, shape):
        """Get index map for a image.

        Args:
            shape: [B, T, H, W] or [B, H, W]
        Returns:
            idx: [B, T, H, W, 2], or [B, H, W, 2] stores (x, y)
        """
        s = shape
        ndims = tf.shape(s)
        wdim = ndims - 1
        hdim = ndims - 2
        idx_shape = tf.concat(0, [s, tf.constant([1])])
        ones_h = tf.ones(hdim - 1, dtype='int32')
        ones_w = tf.ones(wdim - 1, dtype='int32')
        h_shape = tf.concat(
            0, [ones_h, tf.constant([-1]), tf.constant([1, 1])])
        w_shape = tf.concat(0, [ones_w, tf.constant([-1]), tf.constant([1])])

        idx_y = tf.zeros(idx_shape, dtype='float')
        idx_x = tf.zeros(idx_shape, dtype='float')

        h = tf.slice(s, ndims - 2, [1])
        w = tf.slice(s, ndims - 1, [1])
        idx_y += tf.reshape(tf.to_float(tf.range(h[0])), h_shape)
        idx_x += tf.reshape(tf.to_float(tf.range(w[0])), w_shape)
        # idx = tf.concat(ndims[0], [idx_y, idx_x])
        idx = tf.concat(ndims[0], [idx_x, idx_y])
        # self.log.fatal(idx.get_shape())
        return idx

    def get_filled_box_idx(self, idx, left_top, right_bot):
        """Fill a box with top left and bottom right coordinates.

        Args:
            idx: [B, T, H, W, 2] or [B, H, W, 2] or [H, W, 2]
            left_top: [B, T, 2] or [B, 2] or [2]
            right_bot: [B, T, 2] or [B, 2] or [2]
        """
        ss = tf.shape(idx)
        ndims = tf.shape(ss)
        # coord_shape = tf.pack([ss[0], 1, 1, 2])
        batch = tf.slice(ss, [0], ndims - 3)
        coord_shape = tf.concat(0, [batch, tf.constant([1, 1, 2])])
        left_top = tf.reshape(left_top, coord_shape)
        right_bot = tf.reshape(right_bot, coord_shape)
        lower = tf.reduce_prod(tf.to_float(idx >= left_top), ndims - 1)
        upper = tf.reduce_prod(tf.to_float(idx <= right_bot), ndims - 1)
        box = lower * upper
        return box

    def build(self, inp):
        self.lazy_init_var()
        x = inp['x']
        bbox_gt = inp['bbox_gt']
        phase_train = inp['phase_train']

        x_shape = tf.shape(x)
        num_ex = x_shape[0]
        inp_height = x_shape[2]
        inp_width = x_shape[3]
        res_net_strides = self.get_option('ct:res_net_strides')
        stride_prod = np.prod(res_net_strides) * 4
        results = {}

        timespan = self.get_option('ct:timespan')
        conv_lstm_hid_depth = self.get_option('ct:conv_lstm_hid_depth')

        conv_lstm_state = tf.zeros(
            tf.pack([num_ex, inp_height / stride_prod,
                     inp_width / stride_prod,
                     2 * conv_lstm_hid_depth]))

        switch_offset = self.get_option('ct:switch_offset')
        steps_per_switch_decay = self.get_option('ct:steps_per_switch_decay')
        switch_decay = self.get_option('ct:switch_decay')
        step_offset = tf.maximum(0.0, self.global_step - switch_offset)
        gt_switch = tf.train.exponential_decay(
            1.0, step_offset, steps_per_switch_decay, switch_decay,
            staircase=True)
        gt_prob_switch = tf.to_float(tf.random_uniform(
            tf.pack([num_ex, timespan, 1, 1]), 0, 1.0) <= gt_switch)
        phase_train_f = tf.to_float(phase_train)
        idx_map = self.get_idx_map(tf.pack([num_ex, inp_height, inp_width]))
        self.register_var('idx_map', idx_map)
        bbox_gt_dense = [None] * timespan
        bbox_out_dense = [None] * timespan

        # Annealing idea of sending back the previously output bbox.
        for tt in xrange(1, timespan):
            img_prev = x[:, tt - 1, :, :, :]
            img_now = x[:, tt, :, :, :]
            # bbox_gt_prev = x[:, tt, :, :, :]
            bbox_gt_prev = bbox_gt[:, tt - 1, :]
            # Paint the previous bounding box into a dense image.
            bbox_gt_prev = self.get_filled_box_idx(
                idx_map, bbox_gt_prev[:, :2], bbox_gt_prev[:, 2:])
            # Store it for later.
            bbox_gt_dense[tt - 1] = bbox_gt_prev

            if tt > 1:
                bbox_out_prev = bbox_out_dense[tt - 1]
            else:
                bbox_out_prev = bbox_gt_prev

            bbox_prev = bbox_gt_prev * phase_train_f * \
                gt_prob_switch[:, tt, :, :] + \
                bbox_out_prev * (1 - phase_train_f *
                                 gt_prob_switch[:, tt, :, :])
            bbox_prev = tf.expand_dims(bbox_prev, 3)
            joint_inp = tf.concat(3, [img_prev, img_now, bbox_prev])

            h = self.conv1(joint_inp)
            self.bn1 = tfplus.nn.BatchNorm(h.get_shape()[-1])
            h = tf.nn.relu(h)
            h = tfplus.nn.MaxPool(3, stride=2)(h)

            conv_feat = self.res_net({'input': h, 'phase_train': phase_train})
            conv_lstm_state = self.conv_lstm(
                {'input': conv_feat, 'state': conv_lstm_state})['state']

            # slice the hidden state out
            h_lstm = tf.slice(conv_lstm_state, [0, 0, 0, conv_lstm_hid_depth],
                              [-1, -1, -1, conv_lstm_hid_depth])

            bbox_out_dense[tt] = self.post_cnn({'input': h_lstm,
                                                'phase_train': phase_train})

            # Need to regress score? Not for now maybe...

        bbox_gt_dense[tt] = self.get_filled_box_idx(
            idx_map, bbox_gt[:, tt, :2], bbox_gt[:, tt, 2:])
        return {
            'bbox_out_dense': bbox_out_dense,
            'bbox_gt_dense': bbox_gt_dense
        }

    def build_loss(self, inp, output):
        bbox_out_dense = tf.concat(
            1, [tf.expand_dims(xx, 1) for xx in output['bbox_out_dense'][1:]])
        bbox_gt_dense = tf.concat(
            1, [tf.expand_dims(xx, 1) for xx in output['bbox_gt_dense'][1:]])
        ce = tfplus.nn.CE()({'y_out': bbox_out_dense, 'y_gt': bbox_gt_dense})
        x = inp['x']
        x_shape = tf.shape(x)
        num_ex_f = tf.to_float(x_shape[0])
        inp_height_f = tf.to_float(x_shape[1])
        inp_width_f = tf.to_float(x_shape[2])
        timespan = self.get_option('ct:timespan')

        # Need to mask the loss with s_gt.
        ce = tf.reduce_sum(ce) / num_ex_f / timespan / \
            inp_height_f / inp_width_f
        self.add_loss(ce)
        loss = self.get_loss()
        self.register_var('loss', loss)
        return loss

    def build_optim(self, loss):
        base_learn_rate = self.get_option('ct:base_learn_rate')
        steps_per_learn_rate_decay = self.get_option(
            'ct:steps_per_learn_rate_decay')
        learn_rate_decay = self.get_option('ct:learn_rate_decay')
        clip_gradient = self.get_option('ct:clip_gradient')
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, self.global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        self.register_var('learn_rate', learn_rate)
        eps = 1e-7
        train_step = tfplus.utils.GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=clip_gradient).minimize(loss, global_step=self.global_step)
        return train_step
    pass

tfplus.nn.model.register('conv_lstm_tracker', ConvLSTMTrackerModel)

if __name__ == '__main__':
    m = tfplus.nn.model.create_from_main('conv_lstm_tracker').build_all()
    with tf.Session() as sess:
        idx = sess.run(m.get_var('idx_map'), feed_dict={
            m.get_var('x'): np.zeros([10, 3, 128, 448, 3]),
            m.get_var('phase_train'): False})
        print idx
        print idx.shape
