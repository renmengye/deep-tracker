import cv2
import data_utils
import logger
import numpy as np
import os
import progress_bar as pb
import sharded_hdf5 as sh

log = logger.get()


class KITTIPatchData(object):

    def __init__(self, folder, opt, split='train', seqs=None, usage='match'):
        """
        Args:

            folder: folder where the dataset is

            opt: dict
                patch_height: height of the extracted patch
                patch_width: width of the extracted patch
                center_noise: +/- noise of center shift (uniform), relative to size
                padding_noise: +/- noise of padding (uniform), relative to size
                padding_mean: mean of padding
                num_ex_pos: number of positive examples per object
                num_ex_neg: number of negative examples per object
                shuffle: shuffle the final dataset

            split: string, 'train': sequences 0 - 12, 'valid': sequences 13 - 20

            seqs: list of sequences.
        """
        self.folder = folder
        self.opt = opt
        self.split = split
        self.seqs = seqs
        self.usage = usage
        self.random = np.random.RandomState(2)
        num_ex_pos = self.opt['num_ex_pos']
        num_ex_neg = self.opt['num_ex_neg']
        if split is not None:
            self.h5_fname = os.path.join(
                folder,
                '{}_{}_{}_{}.h5'.format(split, usage, num_ex_pos, num_ex_neg))
        else:
            self.h5_fname = None
        self.dataset = None
        pass

    def get_dataset(self):
        """Get matching dataset. 

        Returns:
            dataset: dict
                images_0: [B, H, W, 3], first instance patches
                images_1: [B, H, W, 3], second instance patches
                label: [B], 1/0, whether they are the same instance. 
        """
        if self.dataset is not None:
            return self.dataset

        if self.h5_fname is not None:
            cache = data_utils.read_h5_data(self.h5_fname)
            if cache:
                return cache
        patch_height = self.opt['patch_height']
        patch_width = self.opt['patch_width']
        center_noise = self.opt['center_noise']
        padding_noise = self.opt['padding_noise']
        padding_mean = self.opt['padding_mean']
        num_ex_pos = self.opt['num_ex_pos']
        num_ex_neg = self.opt['num_ex_neg']
        shuffle = self.opt['shuffle']
        folder = self.folder
        split = self.split
        seqs = self.seqs
        usage = self.usage
        random = self.random

        dataset_pattern = os.path.join(folder, 'dataset-*')
        dataset_file = sh.ShardedFile.from_pattern_read(dataset_pattern)
        dataset_images = []
        dataset_labels = []

        if split is not None:
            if split == 'train':
                seqs = range(13)
                self.seqs = seqs
            elif split == 'valid':
                seqs = range(13, 21)
                self.seqs = seqs
            else:
                raise Exception('Unknown split: {}'.format(split))
            pass

        with sh.ShardedFileReader(dataset_file) as reader:
            for seq_num in pb.get_iter(seqs):
                seq_data = reader[seq_num]
                images = seq_data['images_0']
                gt_bbox = seq_data['gt_bbox']
                num_obj = gt_bbox.shape[0]
                num_frames = gt_bbox.shape[1]
                nneg = num_ex_neg * num_obj
                npos = num_ex_pos * num_obj

                if usage == 'match':
                    output_images = np.zeros(
                        [nneg + npos, 2, patch_height, patch_width, 3],
                        dtype='uint8')
                elif usage == 'detect' or usage == 'detect_multiscale':
                    output_images = np.zeros(
                        [nneg + npos, patch_height, patch_width, 3],
                        dtype='uint8')

                output_labels = np.zeros([nneg + npos], dtype='uint8')
                dataset_images.append(output_images)
                dataset_labels.append(output_labels)

                if num_obj < 2:
                    continue

                if usage == 'match':
                    output_images[: nneg], output_labels[: nneg] = \
                        self.get_neg_pair(nneg, images, gt_bbox)

                    output_images[nneg:], output_labels[nneg:] = \
                        self.get_pos_pair(npos, images, gt_bbox)
                elif usage == 'detect':
                    output_images[: nneg], output_labels[: nneg] = \
                        self.get_neg_patch(nneg, images, gt_bbox)

                    output_images[nneg:], output_labels[nneg:] = \
                        self.get_pos_patch(npos, images, gt_bbox)
                elif usage == 'detect_multiscale':
                    output_images[: nneg], output_labels[: nneg] = \
                        self.get_neg_patch_multiscale(nneg, images, gt_bbox)

                    output_images[nneg:], output_labels[nneg:] = \
                        self.get_pos_patch_multiscale(npos, images, gt_bbox)
                pass
            pass

        dataset = self.assemble_dataset(dataset_images, dataset_labels)
        self.dataset = dataset

        if self.h5_fname is not None:
            data_utils.write_h5_data(self.h5_fname, dataset)

        return dataset

    def crop_patch(self, image, bbox):
        """Get a crop of the image.

        Args:
            image: [H, W, 3]
            bbox: [left, top, right, bottom] 
        """
        patch_height = self.opt['patch_height']
        patch_width = self.opt['patch_width']
        center_noise = self.opt['center_noise']
        padding_noise = self.opt['padding_noise']
        padding_mean = self.opt['padding_mean']
        padding = padding_mean
        patch_size = [patch_height, patch_width]
        random = self.random

        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]
        size_x = right - left
        size_y = bottom - top
        im_height = image.shape[0]
        im_width = image.shape[1]

        pn_x = random.uniform(padding - padding_noise, padding + padding_noise)
        pn_y = random.uniform(padding - padding_noise, padding + padding_noise)
        cn_x = random.uniform(-center_noise, center_noise)
        cn_y = random.uniform(-center_noise, center_noise)

        left = left + (cn_x - pn_x) * size_x
        right = right + (cn_x + pn_x) * size_x
        top = top + (cn_y - pn_y) * size_y
        bottom = bottom + (cn_y + pn_y) * size_y

        left = max(0, left)
        right = min(right, im_width)
        top = max(0, top)
        bottom = min(bottom, im_height)
        image_crop = image[top: bottom + 1, left: right + 1, :]
        image_resize = cv2.resize(image_crop, (patch_size[1], patch_size[0]))

        return image_resize

    def get_neg_pair(self, num, images, gt_bbox):
        """Get negative pair."""
        patch_height = self.opt['patch_height']
        patch_width = self.opt['patch_width']
        center_noise = self.opt['center_noise']
        padding_noise = self.opt['padding_noise']
        padding_mean = self.opt['padding_mean']
        random = self.random

        patch_size = [patch_height, patch_width]
        output_images = np.zeros(
            [num, 2, patch_height, patch_width, 3], dtype='uint8')
        output_labels = np.zeros([num], dtype='uint8')
        num_obj = gt_bbox.shape[0]

        for ii in xrange(num):
            obj_id1 = 0
            obj_id2 = 0

            while obj_id1 == obj_id2:
                obj_id1 = int(np.floor(random.uniform(0, num_obj)))
                obj_id2 = int(np.floor(random.uniform(0, num_obj)))
                pass

            non_zero_frames1 = gt_bbox[obj_id1, :, 4].nonzero()[0]
            idx1 = np.floor(random.uniform(0,
                                           non_zero_frames1.shape[0]))
            frm1 = non_zero_frames1[idx1]

            non_zero_frames2 = gt_bbox[obj_id2, :, 4].nonzero()[0]
            idx2 = np.floor(random.uniform(0,
                                           non_zero_frames2.shape[0]))
            frm2 = non_zero_frames2[idx2]

            image1 = images[frm1]
            image2 = images[frm2]
            bbox1 = gt_bbox[obj_id1, frm1, :4]
            bbox2 = gt_bbox[obj_id2, frm2, :4]
            patch_size = [patch_height, patch_width]
            output_images[ii, 0] = self.crop_patch(image1, bbox1)
            output_images[ii, 1] = self.crop_patch(image2, bbox2)
            output_labels[ii] = 0
            pass

        return output_images, output_labels

    def get_pos_pair(self, num, images, gt_bbox):
        """Get positive pair."""
        patch_height = self.opt['patch_height']
        patch_width = self.opt['patch_width']
        center_noise = self.opt['center_noise']
        padding_noise = self.opt['padding_noise']
        padding_mean = self.opt['padding_mean']
        random = self.random

        patch_size = [patch_height, patch_width]
        output_images = np.zeros(
            [num, 2, patch_height, patch_width, 3], dtype='uint8')
        output_labels = np.zeros([num], dtype='uint8')
        num_obj = gt_bbox.shape[0]

        for ii in xrange(num):
            frames = np.array([0])
            while frames.shape[0] <= 1:
                obj_id = int(np.floor(random.uniform(0, num_obj)))
                frames = gt_bbox[obj_id, :, 4].nonzero()[0]
                pass

            idx1 = 0
            idx2 = 0
            while idx1 == idx2:
                idx1 = int(np.floor(random.uniform(0, frames.shape[0])))
                idx2 = int(np.floor(random.uniform(0, frames.shape[0])))
                pass

            frm1 = frames[idx1]
            frm2 = frames[idx2]
            image1 = images[frm1]
            image2 = images[frm2]
            bbox1 = gt_bbox[obj_id, frm1, :4]
            bbox2 = gt_bbox[obj_id, frm2, :4]
            output_images[ii, 0] = self.crop_patch(image1, bbox1)
            output_images[ii, 1] = self.crop_patch(image2, bbox2)
            output_labels[ii] = 1
            pass
        return output_images, output_labels

    def get_neg_patch(self, num, images, gt_bbox):
        """Extract negative patches randomly.

        Args:
            num: number of patches.
            images: [T, H, W, 3]
            gt_bbox: [N, T, 5]
        """
        patch_height = self.opt['patch_height']
        patch_width = self.opt['patch_width']
        center_noise = self.opt['center_noise']
        padding_noise = self.opt['padding_noise']
        padding_mean = self.opt['padding_mean']
        random = self.random

        patch_size = [patch_height, patch_width]
        output_images = np.zeros(
            [num, patch_height, patch_width, 3], dtype='uint8')
        output_labels = np.zeros([num], dtype='uint8')
        num_obj = gt_bbox.shape[0]
        im_height = images.shape[-3]
        im_width = images.shape[-2]

        top_left = gt_bbox[:, :, :2]
        bot_right = gt_bbox[:, :, 2: 4]
        box_size = bot_right - top_left
        box_width = box_size[:, :, 0]
        box_height = box_size[:, :, 1]
        num_boxes = gt_bbox[:, :, 4].sum()
        mean_box_width = box_width.sum() / num_boxes
        mean_box_height = box_height.sum() / num_boxes
        std_box_width = np.sqrt(
            ((box_width - mean_box_width) * (box_width - mean_box_width) *
             gt_bbox[:, :, 4]).sum() / num_boxes)
        std_box_height = np.sqrt(
            ((box_height - mean_box_height) * (box_height - mean_box_height) *
             gt_bbox[:, :, 4]).sum() / num_boxes)

        for ii in xrange(num):
            frm = int(np.floor(random.uniform(0, images.shape[0])))
            image = images[frm]
            bbox_height = int(random.normal(mean_box_height, std_box_height))
            bbox_height = max(20, bbox_height)
            bbox_width = int(random.normal(mean_box_width, std_box_width))
            bbox_width = max(20, bbox_width)
            bbox_y = int(random.uniform(0, im_height - bbox_height))
            bbox_x = int(random.uniform(0, im_width - bbox_width))
            bbox = [bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height]
            output_images[ii] = self.crop_patch(image, bbox)
            output_labels[ii] = 0
            pass

        return output_images, output_labels

    def get_pos_patch(self, num, images, gt_bbox):
        """Extract positive patches.

        Args:
            num: number of patches.
            images: [T, H, W, 3]
            gt_bbox: [N, T, 5]
        """
        patch_height = self.opt['patch_height']
        patch_width = self.opt['patch_width']
        center_noise = self.opt['center_noise']
        padding_noise = self.opt['padding_noise']
        padding_mean = self.opt['padding_mean']
        random = self.random

        patch_size = [patch_height, patch_width]
        output_images = np.zeros(
            [num, patch_height, patch_width, 3], dtype='uint8')
        output_labels = np.zeros([num], dtype='uint8')
        num_obj = gt_bbox.shape[0]

        for ii in xrange(num):
            frames = np.array([0])
            while frames.shape[0] <= 1:
                obj_id = int(np.floor(random.uniform(0, num_obj)))
                frames = gt_bbox[obj_id, :, 4].nonzero()[0]
                pass

            idx = int(np.floor(random.uniform(0, frames.shape[0])))
            frm = frames[idx]
            image = images[frm]
            bbox = gt_bbox[obj_id, frm, :4]
            output_images[ii] = self.crop_patch(image, bbox)
            output_labels[ii] = 1
            pass

        return output_images, output_labels

    def get_neg_patch_multiscale(self, num, images, gt_bbox):
        """Get multiscale patches."""
        patch_height = self.opt['patch_height']
        patch_width = self.opt['patch_width']
        center_noise = self.opt['center_noise']
        padding_noise = self.opt['padding_noise']
        padding_mean = self.opt['padding_mean']
        random = self.random
        nimages = images.shape[0]
        im_height = images.shape[1]
        im_width = images.shape[2]
        num_obj = gt_bbox.shape[0]

        output_images = np.zeros(
            [num, patch_height, patch_width, 3], dtype='uint8')
        output_labels = np.zeros([num], dtype='uint8')
        base_size = np.array([128, 448])
        scale_list = np.array([0.5, 1.0, 1.5, 2.0, 4.0])
        orig_size = np.array([im_height, im_width])
        base_ratio = orig_size / base_size.astype('float32')

        for ii in xrange(num):
            # scale = 1.0
            scale = scale_list[ii % len(scale_list)]
            ratio = base_ratio / scale
            im_size = (base_size * scale).astype('int32')
            frm = int(np.floor(random.uniform(0, images.shape[0])))
            image = images[frm]
            image = cv2.resize(image, (im_size[1], im_size[0]))
            bbox_y = int(random.uniform(0, im_size[0] - patch_height))
            bbox_x = int(random.uniform(0, im_size[1] - patch_width))
            output_images[ii] = image[bbox_y: bbox_y + patch_height,
                                      bbox_x: bbox_x + patch_width, :]
            output_labels[ii] = 0.0

        return output_images, output_labels

    def get_pos_patch_multiscale(self, num, images, gt_bbox):
        """Get multiscale patches."""
        patch_height = self.opt['patch_height']
        patch_width = self.opt['patch_width']
        center_noise = self.opt['center_noise']
        padding_noise = self.opt['padding_noise']
        padding_mean = self.opt['padding_mean']
        random = self.random
        nimages = images.shape[0]
        im_height = images.shape[1]
        im_width = images.shape[2]
        num_obj = gt_bbox.shape[0]

        output_images = np.zeros(
            [num, patch_height, patch_width, 3], dtype='uint8')
        output_labels = np.zeros([num], dtype='uint8')
        base_size = np.array([128, 448])
        scale_list = np.array([0.5, 1.0, 1.5, 2.0, 4.0])
        stride_list = np.array([1, 1, 1, 1, 1])
        orig_size = np.array([im_height, im_width])
        base_ratio = orig_size / base_size.astype('float32')

        for ii in xrange(num):
            found_box = False
            jj = ii
            while not found_box and jj - ii < 1000:
                scale = scale_list[jj % len(scale_list)]
                ratio = base_ratio / scale
                im_size = (base_size * scale).astype('int32')
                frames = np.array([0])
                while frames.shape[0] <= 1:
                    obj_id = int(np.floor(random.uniform(0, num_obj)))
                    frames = gt_bbox[obj_id, :, 4].nonzero()[0]
                idx = int(np.floor(random.uniform(0, frames.shape[0])))
                frm = frames[idx]
                image = images[frm]
                bbox_gt = gt_bbox[obj_id, frm, :4]

                bbox_width = bbox_gt[2] - bbox_gt[0]
                bbox_height = bbox_gt[3] - bbox_gt[1]
                bbox_size = max(bbox_width, bbox_height)
                ratio_list = base_ratio.mean() / scale_list
                bbox_size_list = bbox_size / ratio_list
                bbox_size_list /= float(max(patch_height, patch_width))
                bbox_size_list = (bbox_size_list - 1) ** 2
                scale_id = np.argmin(bbox_size_list)
                scale = scale_list[scale_id]
                stride = stride_list[scale_id]
                # print scale
                ratio = base_ratio / scale
                im_size = (base_size * scale).astype('int32')
                image = cv2.resize(image, (im_size[1], im_size[0]))
                bbox_gt_rescale = np.zeros(4)
                bbox_gt_rescale[0] = bbox_gt[0] / ratio[0]
                bbox_gt_rescale[2] = bbox_gt[2] / ratio[0]
                bbox_gt_rescale[1] = bbox_gt[1] / ratio[1]
                bbox_gt_rescale[3] = bbox_gt[3] / ratio[1]
                bbox_prop, max_iou = self.find_all_overlap_bbox(
                    im_size[0], im_size[1],
                    patch_height, patch_width, bbox_gt_rescale,
                    stride=5, thresh=0.6)
                if len(bbox_prop) > 0:
                    found_box = True
                    box_id = int(
                        np.floor(random.uniform(0, len(bbox_prop))))
                    bbox = bbox_prop[box_id]
                    output_images[ii] = image[
                        bbox[1]: bbox[3], bbox[0]: bbox[2], :]
                    output_labels[ii] = 1.0
                jj += 1

        return output_images, output_labels

    @staticmethod
    def find_all_overlap_bbox(im_height, im_width, patch_height, patch_width, bbox, stride, thresh):
        bleft = max(int(bbox[0]), patch_width / 2)
        btop = max(int(bbox[1]), patch_height / 2)
        bright = min(int(bbox[2]), im_width - patch_width / 2)
        bbot = min(int(bbox[3]), im_height - patch_height / 2)
        box_out = []
        max_iou = 0.0

        for x in xrange(bleft, bright, stride):
            for y in xrange(btop, bbot, stride):
                box_ = (x - patch_width / 2, y - patch_height / 2,
                        x + patch_width / 2, y + patch_width / 2)
                iou = KITTIPatchData.box_iou(box_, bbox)
                max_iou = max(iou, max_iou)
                if iou > thresh:
                    box_out.append(box_)

        return box_out, max_iou

    @staticmethod
    def box_iou(box1, box2):
        # Not IOU here, it is max(Precision, Recall)
        left1 = box1[0]
        left2 = box2[0]
        top1 = box1[1]
        top2 = box2[1]
        right1 = box1[2]
        right2 = box2[2]
        bot1 = box1[3]
        bot2 = box2[3]
        left_ = max(left1, left2)
        top_ = max(top1, top2)
        right_ = min(right1, right2)
        bot_ = min(bot1, bot2)
        inter = (right_ - left_) * (bot_ - top_)
        # union = (right1 - left1) * (bot1 - top1) + \
        #     (right1 - left1) * (bot1 - top1) - inter
        union = min((right1 - left1) * (bot1 - top1),
                    (right1 - left1) * (bot1 - top1))
        return inter / union

    def assemble_dataset(self, dataset_images, dataset_labels):
        seqs = self.seqs
        random = self.random
        usage = self.usage
        shuffle = self.opt['shuffle']
        num_ex = 0
        for ss in xrange(len(seqs)):
            num_ex += dataset_images[ss].shape[0]
            pass

        patch_height = dataset_images[0].shape[-3]
        patch_width = dataset_images[0].shape[-2]

        if usage == 'match':
            final_images = np.zeros([num_ex, 2, patch_height, patch_width, 3],
                                    dtype='uint8')
        elif usage == 'detect' or usage == 'detect_multiscale':
            final_images = np.zeros([num_ex, patch_height, patch_width, 3],
                                    dtype='uint8')
        final_labels = np.zeros([num_ex], dtype='uint8')
        log.info('Image shape: {}'.format(final_images.shape))
        log.info('Label shape: {}'.format(final_labels.shape))

        counter = 0
        for ss in xrange(len(seqs)):
            _num_ex = dataset_images[ss].shape[0]
            final_images[counter: counter + _num_ex] = dataset_images[ss]
            final_labels[counter: counter + _num_ex] = dataset_labels[ss]
            counter += _num_ex
            pass

        if shuffle:
            idx = np.arange(num_ex)
            random.shuffle(idx)
            final_images = final_images[idx]
            final_labels = final_labels[idx]
            pass

        if usage == 'match':
            dataset = {
                'images_0': final_images[:, 0],
                'images_1': final_images[:, 1],
                'labels': final_labels
            }
        elif usage == 'detect' or usage == 'detect_multiscale':
            dataset = {
                'images': final_images,
                'labels': final_labels
            }

        return dataset


if __name__ == '__main__':
    opt = {
        'patch_height': 32,
        'patch_width': 32,
        'center_noise': 0.2,
        'padding_noise': 0.2,
        'padding_mean': 0.2,
        'num_ex_pos': 10,
        'num_ex_neg': 10,
        'shuffle': True
    }

    d = KITTIPatchData(
        '/ais/gobi3/u/mren/data/kitti/tracking/training', opt, split='train',
        usage='detect_multiscale').get_dataset()
