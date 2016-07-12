import tfplus
import tensorflow as tf


class ConvLSTMTrackerModel(tfplus.nn.Model):
    """A model for ConvLSTM tracking."""

    def build_input(self):
        inp_height = self.get_option('ct:inp_height')
        inp_width = self.get_option('ct:inp_width')
        inp_depth = self.get_option('ct:inp_depth')
        timespan = self.get_option('ct:timespan')
        x = self.add_input_var(
            'x', [None, timespan, inp_height, inp_width, inp_depth])
        bbox_gt = self.add_input_var(
            'bbox_gt', [None, timespan, 4])
        phase_train = self.add_input_var('phase_train', None, 'bool')
        results = {
            'x': x,
            'bbox_gt': bbox_gt,
            'phase_train': phase_train
        }
        pass

    def init_var(self):
        inp_depth = self.get_option('ct:inp_depth')
        wd = self.get_option('ct:weight_decay')
        res_net_layers = self.get_option('ct:res_net_layers')
        res_net_channels = self.get_option('ct:res_net_channels')
        res_net_bottleneck = self.get_option('ct:res_net_bottleneck')
        res_net_strides = self.get_option('ct:res_net_strides')

        self.pre_cnn = tfplus.nn.CNN([3], [inp_depth, channels[0]], [1],
                                     [None], [True], wd=wd, scope='pre_cnn')

        self.res_net = tfplus.nn.ResNet(layers=res_net_layers,
                                        bottleneck=res_net_bottleneck,
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

    def build(self, inp):
        x = inp['x']
        phase_train = inp['phase_train']

        x_shape = tf.shape(x)
        num_ex = x_shape[0]

        results = {}

        timespan = self.get_option('ct:timespan')

        conv_lstm_state = tf.zeros(
            [inp_height, inp_width, 2 * conv_lstm_hid_depth])

        bbox_out = [None] * timespan

        switch_offset = self.get_option('ct:switch_offset')
        steps_per_switch_decay = self.get_option('ct:steps_per_switch_decay')
        switch_decay = self.get_option('ct:switch_decay')
        step_offset = tf.maximum(0.0, self.global_step - switch_offset)
        gt_switch = tf.train.exponential_decay(
            1.0, step_offset, steps_per_switch_decay, switch_decay,
            staircase=True)
        gt_prob_switch = tf.to_float(tf.random_uniform(
            tf.pack([num_ex, timespan, 1]), 0, 1.0) <= gt_switch)
        phase_train_f = tf.to_float(phase_train)

        # Annealing idea of sending back the previously output bbox.
        for tt in xrange(1, timespan):
            img_prev = x[:, tt - 1, :, :, :]
            img_now = x[:, tt, :, :, :]
            bbox_gt_prev = x[:, tt, :, :, :]
            if tt > 1:
                bbox_out_prev = bbox_out[tt - 1]
            else:
                bbox_out_prev = bbox_gt_prev

            # Paint the previous bounding box into a dense image.
            bbox_gt_prev = get_box_img(bbox_gt_prev)

            # Leave the probability part for now
            bbox_prev = bbox_gt_prev * phase_train_f * gt_prob_switch[tt] + \
                bbox_out_prev * (1 - phase_train_f * gt_prob_switch[tt])

            joint_inp = tf.concat(3, [img_prev, img_now, bbox_prev_img])

            # Could really be fully convolutional.
            # So we don't have to do pooling here.
            conv_feat = self.res_net(
                {'input': self.pre_cnn({
                    'input': joint_inp,
                    'phase_train': phase_train}),
                    'phase_train': phase_train})

            conv_lstm_state = self.conv_lstm(
                {'input': conv_feat, 'state': conv_lstm_state})['state']

            # slice the hidden state out
            h_lstm = tf.slice(conv_lstm_state, [0, 0, conv_lstm_hid_depth],
                              [inp_height, inp_width, conv_lstm_hid_depth])

            bbox_out[tt] = self.post_cnn({'input': h_lstm,
                                          'phase_train': phase_train})
            pass
        return results

    def build_loss(self, inp, output):
        bbox_gt = inp['bbox_gt'][:, 1:, :]
        bbox_gt = build_img(bbox_gt)
        ce = CE()({'y_out': bbox_out, 'y_gt': bbox_gt})
        self.add_loss(ce)
        loss = self.get_loss()
        self.register_var('loss', loss)
        return loss
        pass

    def build_optim(self, loss):
        pass
    pass
