import cv2
import h5py
import logger
import numpy as np
import os
import progress_bar as pb
import sharded_hdf5 as sh

log = logger.get()


def add_3d_bbox(folder, split):
    split += 'ing'
    h5_fname = os.path.join(folder, split, 'dataset-{}'.format(seq))
    _3d_bbox_folder = os.path.join(folder, split, '3d_bbox')


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
    for seq_num in os.listdir(left_folder):
        if seq_num.startswith('0'):
            seq_list.append(seq_num)
            pass
        pass

    # Prepare output file
    fname_out = os.path.join(folder, split, 'dataset')
    f_out = sh.ShardedFile(fname_out, num_shards=len(seq_list))
    target_types = set(['Van', 'Car', 'Truck'])

    with sh.ShardedFileWriter(f_out, num_objects=len(seq_list)) as writer:
        for seq_num in pb.get_iter(seq_list):
            
            seq_data = {}
            
            if split == 'training':
                label_fname = os.path.join(label_folder, seq_num + '.txt')
                obj_data = {}
                idx_map = []
                frame_start = None
                frame_end = None
                with open(label_fname) as label_f:
                    lines = label_f.readlines()
                    for ll in lines:
                        parts = ll.split(' ')
                        frame_no = int(parts[0])
                        ins_no = int(parts[1])
                        typ = parts[2]
                        truncated = int(parts[3])
                        occluded = int(parts[4])
                        bleft = float(parts[6])
                        btop = float(parts[7])
                        bright = float(parts[8])
                        bbot = float(parts[9])
                        if frame_start is None:
                            frame_start = frame_no
                            frame_end = frame_no
                        else:
                            frame_start = min(frame_start, frame_no)
                            frame_end = max(frame_start, frame_no)
                        
                        raw_data = {
                            'frame_no': frame_no,
                            'ins_no': ins_no,
                            'typ': typ,
                            'truncated': truncated,
                            'occluded': occluded,
                            'bbox': (bleft, btop, bright, bbot)
                        }
                        if ins_no != -1 and typ in target_types:
                            if ins_no in obj_data:
                                obj_data[ins_no].append(raw_data)
                            else:
                                obj_data[ins_no] = [raw_data]

                num_ins = len(obj_data.keys())
                num_frames = frame_end - frame_start + 1
                bbox = np.zeros([num_ins, num_frames, 5], dtype='float32')
                idx_map = []

                for idx in obj_data.iterkeys():
                    new_idx = len(idx_map)
                    for dd in obj_data[idx]:
                        new_frame = dd['frame_no'] - frame_start
                        bbox[new_idx, new_frame, 4] = 1.0
                        bbox[new_idx, new_frame, 0: 4] = dd['bbox']
                    idx_map.append(idx)
                idx_map = np.array(idx_map, dtype='uint8')
                frame_map = np.arange(frame_start, frame_end + 1)

                seq_data['gt_bbox'] = bbox
                seq_data['idx_map'] = idx_map
                seq_data['frame_map'] = frame_map
            
            for camera, camera_folder in enumerate([left_folder, right_folder]):
                if seq_num.startswith('0'):
                    seq_folder = os.path.join(camera_folder, seq_num)
                else:
                    continue
                seq_folder = os.path.join(folder, split, camera_folder, seq_num)
                image_list = os.listdir(seq_folder)
                im_height = None
                im_width = None
                images = None
                for ii, fname in enumerate(image_list):
                    img_fname = os.path.join(seq_folder, fname)
                    log.info(img_fname)
                    frame_no = int(fname[: 6])
                    img = cv2.imread(img_fname)
                    if im_height is None:
                        im_height = img.shape[0]
                        im_width = img.shape[1]
                        images = np.zeros([num_frames, im_height, im_width, 3],
                                dtype='uint8')
                    images[frame_no] = img

                seq_data['images_{}'.format(camera)] = images
            
            writer.write(seq_data)

    return f_out
    
    pass


if __name__ == '__main__':
    folder = '/ais/gobi3/u/mren/data/kitti/tracking'
    get_dataset(folder, 'train')
