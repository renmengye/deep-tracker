import cslab_environ

import tensorflow as tf
import saver
import logger

log = logger.get()


def get_model(opt, device='/cpu:0'):
    return model.get_model(opt, device)


def get_dataset(opt):
    dataset = {}
    folder = '/ais/gobi3/u/mren/data/kitti/tracking/training'
    dataset = data.get_dataset(folder, opt, seqs=[21])

    return dataset


def get_batch_fn(dataset):
    def get_batch(idx):
        x1_bat = dataset['images_0'][idx]
        x2_bat = dataset['images_1'][idx]
        y_bat = dataset['labels'][idx]
        x1_bat, x2_bat, y_bat = preprocess(x1_bat, x2_bat, y_bat)

        return x1_bat, x2_bat, y_bat

    return get_batch


if __name__ == '__main__
    restore_folder = sys.argv[1]
    saver = Saver(restore_folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    data_opt = ckpt_info['data_opt']
    ckpt_fname = ckpt_info['ckpt_fname']
    step = ckpt_info['step']
    model_id = ckpt_info['model_id']

    log.info('Building model')
    m = get_model(model_opt, device=device)

    log.info('Loading dataset')
    dataset = get_dataset(data_opt)

    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    idx = np.arange(10)
    get_batch = get_batch_fn(dataset)
    x1, x2, y = get_batch(idx)

    y = sess.run(m['y'], feed_dict={m['x1']: x1, m[
                 'x2']: x2, m['phase_train']: False})
    print y
