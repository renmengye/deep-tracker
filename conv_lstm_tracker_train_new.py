"""
Train a 50-layer ResNet on ImageNet.
Usage: python res_net_example.py --help
"""

import cslab_environ

import numpy as np
import os
import tensorflow as tf
import tfplus
import kitti_new as kitti
import conv_lstm_tracker_model

tfplus.init('Train a Conv-LSTM tracker on KITTI')
UID_PREFIX = 'conv_lstm_tracker'
DATASET = 'kitti_track'
MODEL_NAME = 'conv_lstm_tracker'

# Main options
tfplus.cmd_args.add('gpu', 'int', -1)
tfplus.cmd_args.add('results', 'str', '../results')
tfplus.cmd_args.add('logs', 'str', '../logs')
tfplus.cmd_args.add('localhost', 'str', 'http://localhost')
tfplus.cmd_args.add('restore_model', 'str', None)
tfplus.cmd_args.add('restore_logs', 'str', None)
tfplus.cmd_args.add('batch_size', 'int', 4)
tfplus.cmd_args.add('prefetch', 'bool', False)


if __name__ == '__main__':
    opt = tfplus.cmd_args.make()

    # Initialize logging/saving folder.
    uid = tfplus.nn.model.gen_id(UID_PREFIX)
    logs_folder = os.path.join(opt['logs'], uid)
    log = tfplus.utils.logger.get(os.path.join(logs_folder, 'raw'))
    tfplus.utils.LogManager(logs_folder).register('raw', 'plain', 'Raw Logs')
    results_folder = os.path.join(opt['results'], uid)

    # Initialize session.
    sess = tf.Session()
    tf.set_random_seed(1234)

    # Initialize model.
    model = (tfplus.nn.model.create_from_main(MODEL_NAME)
             .set_gpu(opt['gpu'])
             .set_folder(results_folder)
             .restore_options_from(opt['restore_model'])
             .build_all()
             )

    if opt['restore_model'] is not None:
        model.restore_weights_aux_from(sess, opt['restore_model'])
    else:
        model.init(sess)

    # Initialize data.
    def get_data(split, batch_size=4, cycle=True, max_queue_size=10,
                 num_threads=10):
        dp = tfplus.data.create_from_main(
            DATASET, split=split).set_iter(
            batch_size=batch_size, cycle=cycle)
        if opt['prefetch']:
            return tfplus.data.ConcurrentDataProvider(
                dp, max_queue_size=max_queue_size, num_threads=num_threads)
        else:
            return dp

    # Initialize experiment.
    (tfplus.experiment.create_from_main('train')
     .set_session(sess)
     .set_model(model)
     .set_logs_folder(os.path.join(opt['logs'], uid))
     .set_localhost(opt['localhost'])
     .restore_logs(opt['restore_logs'])
     .add_csv_output('Loss', ['train'])
     # .add_csv_output('Top 1 Accuracy', ['train', 'valid'])
     # .add_csv_output('Top 5 Accuracy', ['train', 'valid'])
     .add_csv_output('Step Time', ['train'])
     .add_csv_output('Learning Rate', ['train'])
     .add_plot_output('Input (Train)', 'video', max_num_frame=10,
                      max_num_col=5)
     .add_plot_output('GT (Train)', 'video', max_num_frame=10,
                      max_num_col=5, cmap='jet')
     .add_plot_output('Output (Train)', 'video', max_num_frame=10,
                      max_num_col=5, cmap='jet')
     # .add_plot_output('Input (Valid)', 'thumbnail', max_num_col=5)
     .add_runner(
        tfplus.runner.create_from_main('basic')
        .set_name('plotter_train')
        .set_outputs(['x_id', 'bbox_gt_dense', 'bbox_out_dense'])
        .add_plot_listener('Input (Train)', {'x_id': 'images'})
        .add_plot_listener('GT (Train)', {'bbox_gt_dense': 'images'})
        .add_plot_listener('Output (Train)', {'bbox_out_dense', : 'images'})
        .set_data_provider(get_data('train', batch_size=10, cycle=True,
                                    max_queue_size=10, num_threads=5))
        .set_phase_train(True)
        .set_offset(0)       # Every 500 steps (10 min)
        .set_interval(10))
     # .add_runner(
     #    tfplus.runner.create_from_main('basic')
     #    .set_name('plotter_valid')
     #    .set_outputs(['x_trans'])
     #    .add_plot_listener('Input (Valid)', {'x_trans': 'images'})
     #    .set_data_provider(get_data('valid', batch_size=10, cycle=True,
     #                                max_queue_size=10, num_threads=5))
     #    .set_phase_train(False)
     #    .set_offset(0)       # Every 500 steps (10 min)
     #    .set_interval(50))
     .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('train')
        .set_outputs(['loss', 'train_step'])
        .add_csv_listener('Loss', 'loss', 'train')
        .add_csv_listener('Step Time', 'step_time', 'train')
        .add_cmd_listener('Step', 'step')
        .add_cmd_listener('Loss', 'loss')
        .add_cmd_listener('Step Time', 'step_time')
        .set_data_provider(get_data('train', batch_size=opt['batch_size'],
                                    cycle=True))
        .set_phase_train(True)
        .set_num_batch(10)
        .set_interval(1))
     #  .add_runner(
     #     tfplus.runner.create_from_main('saver')
     #     .set_name('saver')
     #     .set_interval(100))    # Every 1000 steps (20 min)
     #  .add_runner(
     #     tfplus.runner.create_from_main('average')
     #     .set_name('trainval')
     #     .set_outputs(['acc', 'top5_acc', 'learn_rate'])
     #     .add_csv_listener('Top 1 Accuracy', 'acc', 'train')
     #     .add_cmd_listener('Top 1 Accuracy', 'acc')
     #     .add_csv_listener('Top 5 Accuracy', 'top5_acc', 'train')
     #     .add_cmd_listener('Top 5 Accuracy', 'top5_acc')
     #     .add_csv_listener('Learning Rate', 'learn_rate', 'train')
     #     .set_data_provider(get_data('train', batch_size=opt['batch_size'],
     #                                 cycle=True))
     #     .set_phase_train(False)
     #     .set_num_batch(10)
     #     .set_offset(100)
     #     .set_interval(20))     # Every 200 steps (4 min)
     # .add_runner(  # Full epoch evaluation on validation set.
     #    tfplus.runner.create_from_main('average')
     #    .set_name('valid')
     #    .set_outputs(['acc', 'top5_acc'])
     #    .add_csv_listener('Top 1 Accuracy', 'acc', 'valid')
     #    .add_cmd_listener('Top 1 Accuracy', 'acc')
     #    .add_csv_listener('Top 5 Accuracy', 'top5_acc', 'valid')
     #    .add_cmd_listener('Top 5 Accuracy', 'top5_acc')
     #    .set_data_provider(get_data('valid', batch_size=opt['batch_size'],
     #                                cycle=True))
     #    .set_phase_train(False)
     #    .set_num_batch(50000 / opt['batch_size'])
     #    .set_offset(100)
     #    .set_interval(1000))    # Every 10000 steps (200 min)
     ).run()
    pass
