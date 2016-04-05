import cv2
import h5py
import logger
import numpy as np
import os

def get_dataset(folder, split):
    """Get KITTI dataset.

    Args:
        folder: root directory.
        split: train or test.

    Returns:
        dataset:
            images: list of [T, H, W, 3], video sequence.
            gt_bbox: list of [T, 4], groundtruth bounding box.
            
    """
    # 'train' => 'training', 'test' => 'testing'
    split += 'ing'
    left_folder = os.path.join(folder, split, 'image_02')
    right_folder = os.path.join(folder, split, 'image_03')
    if split == 'training':
        label_folder = os.path.join(folder, split, 'label_02')

    for camera, camera_folder in enumerate([left_folder, right_folder]):
        for seq_num in os.listdir(camera_folder):
            if seq_num.startswith('0'):
                seq_folder = os.path.join(camera_folder, seq_num)
            else:
                continue
            for fname in os.listdir(seq_folder):
                img_fname = os.path.join(seq_folder, fname)
                print img_fname
    pass


if __name__ == '__main__':
    folder = '/ais/gobi3/u/mren/data/kitti/tracking'
    get_dataset(folder, 'train')
