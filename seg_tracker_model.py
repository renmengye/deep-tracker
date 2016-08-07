import cslab_environ

import numpy as np
import tfplus
import tensorflow as tf

tfplus.cmd_args.add('st:timespan', 'int', 20)
tfplus.cmd_args.add('st:weight_decay', 'float', 5e-5)
tfplus.cmd_args.add('st:conv_lstm_hid_depth', 'int', 16)
tfplus.cmd_args.add('st:conv_lstm_filter_size', 'int', 3)
tfplus.cmd_args.add('st:base_learn_rate', 'float', 1e-3)
tfplus.cmd_args.add('st:learn_rate_decay', 'float', 0.5)
tfplus.cmd_args.add('st:steps_per_learn_rate_decay', 'int', 5000)
tfplus.cmd_args.add('st:switch_offset', 'int', 1000)
tfplus.cmd_args.add('st:steps_per_switch_decay', 'int', 1000)
tfplus.cmd_args.add('st:switch_decay', 'float', 0.9)
tfplus.cmd_args.add('st:clip_gradient', 'float', 1.0)


class SegTrackerModel(tfplus.nn.Model):
    """A model for ConvLSTM tracking."""

    def __init__(self):
        super(SegTrackerModel, self).__init__(name='seg_tracker')
        self.register_option('st:timespan')
        self.register_option('st:weight_decay')
        self.register_option('st:conv_lstm_hid_depth')
        self.register_option('st:conv_lstm_filter_size')
        self.register_option('st:switch_offset')
        self.register_option('st:steps_per_switch_decay')
        self.register_option('st:switch_decay')
        self.register_option('st:base_learn_rate')
        self.register_option('st:learn_rate_decay')
        self.register_option('st:steps_per_learn_rate_decay')
        self.register_option('st:clip_gradient')
        pass

    def init_default_options(self):
        pass

    def build_input(self):
        self.init_default_options()
        timespan = self.get_option('st:timespan')
        x = self.add_input_var(
            'x', [None, timespan, None, None, 3])
        bbox_gt = self.add_input_var(
            'bbox_gt', [None, timespan, 4])
        fg = self.add_input_var('fg', [None, timespan, None, None, 1])
        angle = self.add_input_var('angle', [None, timespan, None, None, 8])
        phase_train = self.add_input_var('phase_train', None, 'bool')
        s_gt = self.add_input_var('s_gt', [None, timespan])
        results = {
            'x': x,
            'fg': fg,
            'angle': angle,
            's_gt': s_gt,
            'bbox_gt': bbox_gt,
            'phase_train': phase_train
        }
        return results

    def init_var(self):
        with tf.device(self.get_device_fn()):
            wd = self.get_option('st:weight_decay')
            conv_lstm_f_size = self.get_option('st:conv_lstm_filter_size')
            conv_lstm_depth = self.get_option('st:conv_lstm_hid_depth')
            self.conv_lstm = tfplus.nn.ConvLSTM(filter_size=conv_lstm_f_size,
                                                inp_depth=14,
                                                hid_depth=conv_lstm_depth,
                                                wd=wd)
            self.conv2 = tfplus.nn.Conv2DW(f=1, ch_in=conv_lstm_depth,
                                           ch_out=1, stride=1, wd=wd,
                                           scope='conv2', bias=True)
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
        idx = tf.concat(ndims[0], [idx_x, idx_y])
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
        fg = inp['fg']
        angle = inp['angle']
        bbox_gt = inp['bbox_gt']
        phase_train = inp['phase_train']

        x_shape = tf.shape(x)
        num_ex = x_shape[0]
        self.register_var('num_ex', num_ex)
        inp_height = x_shape[2]
        self.register_var('inp_height', inp_height)
        inp_width = x_shape[3]
        self.register_var('inp_width', inp_width)
        results = {}

        timespan = self.get_option('st:timespan')
        conv_lstm_hid_depth = self.get_option('st:conv_lstm_hid_depth')

        # Concatenate the first bounding box +
        conv_lstm_state = tf.zeros(
            tf.pack([num_ex, inp_height, inp_width, 2 * conv_lstm_hid_depth]))

        switch_offset = self.get_option('st:switch_offset')
        steps_per_switch_decay = self.get_option(
            'st:steps_per_switch_decay')
        switch_decay = self.get_option('st:switch_decay')
        step_offset = tf.maximum(0.0, self.global_step - switch_offset)
        gt_switch = tf.train.exponential_decay(
            1.0, step_offset, steps_per_switch_decay, switch_decay,
            staircase=True)
        self.register_var('gt_switch', gt_switch)
        gt_prob_switch = tf.to_float(tf.random_uniform(
            tf.pack([num_ex, timespan, 1, 1, 1]), 0, 1.0) <= gt_switch)
        phase_train_f = tf.to_float(phase_train)

        idx_map_hi_res = self.get_idx_map(
            tf.pack([num_ex, inp_height, inp_width]))
        bbox_gt_dense = [None] * timespan
        bbox_out_dense = [None] * timespan

        # Annealing idea of sending back the previously output bbox.
        for tt in xrange(1, timespan):
            img_now = x[:, tt, :, :, :]
            fg_now = fg[:, tt, :, :, :]
            fg_prev = fg[:, tt - 1, :, :, :]
            angle_now = angle[:, tt, :, :, :]
            bbox_gt_prev_coord = bbox_gt[:, tt - 1, :]

            # Paint the previous bounding box into a dense image.
            bbox_gt_prev_hi = self.get_filled_box_idx(
                idx_map_hi_res, bbox_gt_prev_coord[:, :2],
                bbox_gt_prev_coord[:, 2:])

            # [B, H, W] => [B, H, W, 1]
            bbox_gt_prev_hi = tf.expand_dims(bbox_gt_prev_hi, 3)
            # Store it for later.
            bbox_gt_dense[tt - 1] = bbox_gt_prev_hi
            if tt == 1:
                bbox_out_dense[0] = bbox_gt_prev_hi

            bbox_out_prev = bbox_out_dense[tt - 1]
            switch = gt_prob_switch[:, tt, :, :, :] * phase_train_f
            bbox_prev = bbox_gt_prev_hi * switch + bbox_out_prev * (1 - switch)

            # 3 + 1 + 1 + 8 + 1
            joint_inp = tf.concat(
                3, [img_now, fg_prev, fg_now, angle_now, bbox_prev])
            conv_lstm_state = self.conv_lstm(
                {'input': joint_inp, 'state': conv_lstm_state})

            # slice the hidden state out
            h_lstm = tf.slice(conv_lstm_state, [0, 0, 0, conv_lstm_hid_depth],
                              [-1, -1, -1, conv_lstm_hid_depth])
            bbox_out_dense[tt] = tf.sigmoid(self.conv2(h_lstm))

            # Need to regress score? Not for now maybe...

        # [B, H, W] => [B, H, W, 1]
        bbox_gt_dense[tt] = tf.expand_dims(self.get_filled_box_idx(
            idx_map_hi_res, bbox_gt[:, tt, :2], bbox_gt[:, tt, 2:]), 3)
        bbox_out_dense = tf.concat(
            1, [tf.expand_dims(xx, 1) for xx in bbox_out_dense])
        bbox_gt_dense = tf.concat(
            1, [tf.expand_dims(xx, 1) for xx in bbox_gt_dense])
        self.register_var('bbox_out_dense', bbox_out_dense)
        self.register_var('bbox_gt_dense', bbox_gt_dense)
        return {
            'bbox_out_dense': bbox_out_dense,
            'bbox_gt_dense': bbox_gt_dense
        }

    def build_loss(self, inp, output):
        s_gt = inp['s_gt']
        bbox_out_dense = output['bbox_out_dense']
        bbox_gt_dense = output['bbox_gt_dense']
        ce = tfplus.nn.CE()(
            {'y_out': bbox_out_dense, 'y_gt': bbox_gt_dense})
        num_ex_f = tf.to_float(self.get_var('num_ex'))
        inp_height_f = tf.to_float(self.get_var('inp_height'))
        inp_width_f = tf.to_float(self.get_var('inp_width'))
        timespan = self.get_option('st:timespan')

        # Need to mask the loss with s_gt.
        # [B, T, H, W, 1] = > [B, T]
        ce = tf.reduce_sum(ce, [2, 3, 4])
        ce = tf.reduce_sum(ce * s_gt) / tf.reduce_sum(s_gt)
            # / inp_height_f / inp_width_f
        self.add_loss(ce)
        loss = self.get_loss()
        self.register_var('loss', loss)
        return loss

    def build_optim(self, loss):
        base_learn_rate = self.get_option('st:base_learn_rate')
        steps_per_learn_rate_decay = self.get_option(
            'st:steps_per_learn_rate_decay')
        learn_rate_decay = self.get_option('st:learn_rate_decay')
        clip_gradient = self.get_option('st:clip_gradient')
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, self.global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        self.register_var('learn_rate', learn_rate)
        eps = 1e-7
        train_step = tfplus.utils.GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=clip_gradient).minimize(loss, global_step=self.global_step)
        return train_step

    def get_save_var_dict(self):
        results = {}
        if self.has_var('step'):
            results['step'] = self.get_var('step')
        self.add_prefix_to(
            'conv_lstm', self.conv_lstm.get_save_var_dict(), results)
        self.add_prefix_to('conv2', self.conv2.get_save_var_dict(), results)
    pass

tfplus.nn.model.register('seg_tracker', SegTrackerModel)

if __name__ == '__main__':
    m = tfplus.nn.model.create_from_main('seg_tracker').build_all()
    pass
