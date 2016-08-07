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
import seg_tracker_model
import orientation_plotter
from tfplus.utils import BatchIterator, ConcurrentBatchIterator

tfplus.init('Train a Conv-LSTM tracker on KITTI')

# Main options
tfplus.cmd_args.add('gpu', 'int', -1)
tfplus.cmd_args.add('model', 'str', 'seg_tracker')
tfplus.cmd_args.add('results', 'str', '../results')
tfplus.cmd_args.add('logs', 'str', '../logs')
tfplus.cmd_args.add('localhost', 'str', 'http://localhost')
tfplus.cmd_args.add('restore_model', 'str', None)
tfplus.cmd_args.add('restore_logs', 'str', None)
tfplus.cmd_args.add('batch_size', 'int', 8)
tfplus.cmd_args.add('prefetch', 'bool', False)
opt = tfplus.cmd_args.make()

DATASET = 'kitti_track'

# Initialize logging/saving folder.
uid = tfplus.nn.model.gen_id(opt['model'])
logs_folder = os.path.join(opt['logs'], uid)
log = tfplus.utils.logger.get(os.path.join(logs_folder, 'raw'))
tfplus.utils.LogManager(logs_folder).register('raw', 'plain', 'Raw Logs')
results_folder = os.path.join(opt['results'], uid)

# Initialize session.
sess = tf.Session()
tf.set_random_seed(1234)

# Initialize model.
model = (
    tfplus.nn.model.create_from_main(opt['model'])
    .set_gpu(opt['gpu'])
    .set_folder(results_folder)
    .restore_options_from(opt['restore_model'])
    .build_all()
)

if opt['restore_model'] is not None:
    model.restore_weights_aux_from(sess, opt['restore_model'])
else:
    model.init(sess)

# Intialize data.
data = {}
for split in ['train', 'valid']:
    data[split] = tfplus.data.create_from_main(DATASET, split=split)


def get_data(split, batch_size=4, cycle=True, max_queue_size=10,
             num_threads=10):
    batch_iter = BatchIterator(
        num=data[split].get_size(), progress_bar=False, shuffle=True,
        batch_size=batch_size, cycle=cycle,
        get_fn=data[split].get_batch_idx)
    if opt['prefetch']:
        batch_iter = ConcurrentBatchIterator(
            batch_iter, max_queue_size=max_queue_size,
            num_threads=num_threads)
    return batch_iter

# Initialize experiment.
exp = (
    tfplus.experiment.create_from_main('train')
    .set_session(sess)
    .set_model(model)
    .set_logs_folder(os.path.join(opt['logs'], uid))
    .set_localhost(opt['localhost'])
    .restore_logs(opt['restore_logs'])

    .add_csv_output('Loss', ['train'])

    .add_csv_output('Step Time', ['train'])
    .add_csv_output('Learning Rate', ['train'])
    .add_csv_output('GT Switch', ['train'])

    .add_plot_output('Input (Train)', 'video', max_num_frame=10,
                     max_num_col=5)
    .add_plot_output('GT (Train)', 'video', max_num_frame=10,
                     max_num_col=5, cmap='jet')
    .add_plot_output('Output (Train)', 'video', max_num_frame=10,
                     max_num_col=25, cmap='jet')

    .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('train')
        .add_output('loss')
        .add_output('train_step')
        .add_output('step_time')
        .add_csv_listener('Loss', 'loss', 'train')
        .add_csv_listener('Step Time', 'step_time', 'train')
        .add_cmd_listener('Step', 'step')
        .add_cmd_listener('Loss', 'loss')
        .add_cmd_listener('Step Time', 'step_time')
        .set_iter(get_data('train', batch_size=opt['batch_size'],
                           cycle=True))
        .set_phase_train(True)
        .set_num_batch(10)
        .set_interval(1))

    .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('trainval')
        .add_output('gt_switch')
        .add_csv_listener('GT Switch', 'gt_switch', 'train')
        .add_cmd_listener('GT Switch', 'gt_switch')
        .set_iter(get_data('train', batch_size=1, cycle=True))
        .set_phase_train(False)
        .set_num_batch(1)
        .set_interval(10)
        )
)

runner_plot = (
    tfplus.runner.create_from_main('basic')
    .set_name('plotter_train')
    .set_outputs(['bbox_gt_dense', 'bbox_out_dense'])
    .add_plot_listener('Input (Train)', {'x': 'images'})
    .add_plot_listener('GT (Train)', {'bbox_gt_dense': 'images'})
    .add_plot_listener('Output (Train)', {'bbox_out_dense': 'images'})
    .set_iter(get_data('train', batch_size=2, cycle=True,
                       max_queue_size=10, num_threads=1))
    .set_phase_train(False)
    .set_offset(0)       # Every 500 steps (10 min)
    .set_interval(20)
)

if opt['model'] == 'seg_tracker':
    (
        exp
        .add_plot_output('Input FG (Train)', 'video', max_num_frame=2,
                         max_num_col=10, cmap='jet')
        .add_plot_output('Input Angle (Train)', 'orientation_video',
                         max_num_frame=10, max_num_col=5)
    )
    (
        runner_plot
        .add_plot_listener('Input FG (Train)', {'fg': 'images'})
        .add_plot_listener('Input Angle (Train)', {'fg': 'foreground',
                                                   'angle': 'orientation'})
    )

exp.add_runner(runner_plot)

exp.run()
