import cv2
import h5py
import logger
import numpy as np
import os
import progress_bar as pb
import sharded_hdf5 as sh

log = logger.get()

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
    h5_fname = os.path.join(folder, split, 'dataset-*')
    try:
        h5_f = sh.ShardedFile.from_pattern_read(h5_fname)
    except:
        h5_f = None

    if h5_f:
        return h5_f

    left_folder = os.path.join(folder, split, 'image_02')
    right_folder = os.path.join(folder, split, 'image_03')
    if split == 'training':
        label_folder = os.path.join(folder, split, 'label_02')

    # List the sequences
    seq_list = [] 
    for seq_num in os.listdir('image_02'):
        if seq_num.startswith('0'):
            seq_list.append(seq_num)
            pass
        pass

    # Prepare output file
    fname_out = os.path.join(folder, split, 'dataset')
    f_out = sh.ShardedFile(fname_out, num_shards=len(seq_list))
    
    with sh.ShardedFileWriter(f_out, num_objects=1) as writer:
        for seq_num in pb.get_iter(seq_list):
            
            seq_data = {}
            
            for camera, camera_folder in enumerate([left_folder, right_folder]):
                if seq_num.startswith('0'):
                    seq_folder = os.path.join(camera_folder, seq_num)
                else:
                    continue
                seq_folder = os.path.join(folder, split, camera_folder)
                image_list = os.listdir(seq_folder)
                im_height = None
                im_width = None
                images = None
                num_images = len(image_list)
                for ii, fname in enumerate(image_list):
                    img_fname = os.path.join(seq_folder, fname)
                    img = cv2.imread(img_fname)
                    if im_height is None:
                        im_height = img.shape[0]
                        im_widht = img.shape[1]
                        images = np.zeros([num_images, im_height, im_width, 3],
                                dtype='uint8')
                    images[ii] = img

                seq_data['images_{}'.format(camera)] = images
            
            label_file = None 
            # Exist state
            seq_data['exist'] = None
            # Occlusion state
            seq_data['occlusion'] = None
            seq_data['gt_bbox'] = None
            writer.write(seq_data)

    return f_out
    
    pass


if __name__ == '__main__':
    folder = '/ais/gobi3/u/mren/data/kitti/tracking'
    get_dataset(folder, 'train')
