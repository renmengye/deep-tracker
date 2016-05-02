import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_frame_with_bbox(fname, data, pred_bbox, gt_bbox, iou, predict_score, num_row, num_col):
    f, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))

    for ii in xrange(num_row):
        for jj in xrange(num_col):
            axarr[ii, jj].set_axis_off()
            idx = ii * num_col + jj
            axarr[ii, jj].imshow(data[idx], cmap=cm.Greys_r)

            if predict_score[idx] > 0.5:
                axarr[ii, jj].add_patch(patches.Rectangle(
                    (pred_bbox[idx][0], pred_bbox[idx][1]),
                    pred_bbox[idx][2] - pred_bbox[idx][0],
                    pred_bbox[idx][3] - pred_bbox[idx][1],
                    fill=False,
                    color='r'))

            axarr[ii, jj].add_patch(patches.Rectangle(
                (gt_bbox[idx][0], gt_bbox[idx][1]),
                gt_bbox[idx][2] - gt_bbox[idx][0],
                gt_bbox[idx][3] - gt_bbox[idx][1],
                fill=False,
                color='b'))

            axarr[ii, jj].text(0, 0, ("%5.2f" % iou[idx]),
                               color=(0, 0, 0), size=8)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=80)
    plt.close('all')

def plot_batch_frame_with_bbox(fname, data, pred_bbox, gt_bbox, predict_score):
    
    for idx, img in enumerate(data):
        f, axarr = plt.subplots(1, 1)

        axarr[0, 0].set_axis_off()        
        axarr[0, 0].imshow(img, cmap=cm.Greys_r)

        if predict_score[idx] > 0.5:
            axarr[0, 0].add_patch(patches.Rectangle(
                (pred_bbox[idx][0], pred_bbox[idx][1]),
                pred_bbox[idx][2] - pred_bbox[idx][0],
                pred_bbox[idx][3] - pred_bbox[idx][1],
                fill=False,
                color='r'))

        axarr[0, 0].add_patch(patches.Rectangle(
            (gt_bbox[idx][0], gt_bbox[idx][1]),
            gt_bbox[idx][2] - gt_bbox[idx][0],
            gt_bbox[idx][3] - gt_bbox[idx][1],
            fill=False,
            color='b'))

        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(fname + ("%05d" % (idx)), dpi=80)
        plt.close('all')

def collect_draw_sequence(draw_raw_imgs, draw_raw_gt_bbox, seq_length, height, width):

    count_draw = 0
    idx_draw_frame = 0
    skip_empty = True
    draw_imgs = []
    draw_gt_box = []
    draw_idx_obj = 2

    while count_draw <= seq_length:
        if draw_raw_gt_bbox[draw_idx_obj, idx_draw_frame, 4] == 1:
            skip_empty = False

        if not skip_empty:
            draw_imgs.append(cv2.resize(
                draw_raw_imgs[idx_draw_frame], (width, height), interpolation=cv2.INTER_CUBIC))

            # draw 3-th object in the sequence
            tmp_box = np.array(draw_raw_gt_bbox[draw_idx_obj, idx_draw_frame, :4])
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
    # gt_bbox = [left top right bottom flag]
    draw_raw_gt_bbox = draw_data['gt_bbox']

    draw_imgs, draw_gt_box = collect_draw_sequence(
        draw_raw_imgs, draw_raw_gt_bbox, seq_length, height, width)

    feed_data = {tracking_model['imgs']: np.expand_dims(draw_imgs, 0),
                 tracking_model['init_bbox']: np.expand_dims(draw_gt_box[0], 0),
                 tracking_model['gt_bbox']: np.expand_dims(draw_gt_box, 0),
                 tracking_model['anneal_threshold']: [1.0],
                 tracking_model['phase_train']: False}

    draw_pred_bbox, draw_IOU_score, predict_score = sess.run(
        [tracking_model['predict_bbox'], tracking_model['IOU_score'], tracking_model['predict_score']], feed_dict=feed_data)
    
    draw_pred_bbox = np.squeeze(draw_pred_bbox)
    draw_IOU_score = np.squeeze(draw_IOU_score)
    predict_score = np.squeeze(predict_score)

    num_col = 2
    num_row = seq_length / num_col
    plot_frame_with_bbox(draw_img_name, draw_imgs[1:], draw_pred_bbox, draw_gt_box[
                         1:], draw_IOU_score, predict_score, num_row, num_col)
