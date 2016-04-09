# import h5py
import logger
import numpy as np
import sharded_hdf5 as sh

log = logger.get()


def get_dataset(folder, opt):
    """Get matching dataset.

    opt:
        patch_height:
        patch_width:
        center_noise:
        padding_noise:
        padding_mean:
        num_ex_pos:
        num_ex_neg:

    """
    patch_height = opt['patch_height']
    patch_width = opt['patch_width']
    center_noise = opt['center_noise']
    padding_noise = opt['padding_noise']
    padding_mean = opt['padding_mean']

    dataset_pattern = os.path.join(folder, 'dataset-*')
    file = sh.ShardedFile.from_pattern_read(dataset_pattern)
    with sh.ShardedFileReader(file) as reader:
        for seq_num, seq_data in enumerate(reader):
            images = seq_data['images']
            gt_bbox = seq_data['gt_bbox']
            num_obj = gt_bbox.shape[0]
            num_frames = gt_bbox.shape[1]

            # n = number of objects in the sequence.
            # t = number of frames.
            # This operation is O(n^2 t^2).
            # t ~= 200, n ~= 10 => 100 * 40,000 => 4,000,000 operations.

            random = np.random.RandomState(2)
            for ii in xrange(num_ex_neg):
                obj_id_1 = 0
                obj_id_2 = 0
                while obj_id_1 != obj_id_2:
                    obj_id_1 = np.floor(random.uniform(0, num_obj))
                    obj_id_2 = np.floor(random.uniform(0, num_obj))

                non_zero_frames1 = gt_bbox[obj_id_1, :, 4].nonzeros()
                random.shuffle(non_zero_frames1)
                frm1 = non_zero_frames1[0]

                non_zero_frames2 = gt_bbox[obj_id_2, :, 4].nonzeros()
                random.shuffle(non_zero_frames2)
                frm2 = non_zero_frames2[0]

                print seq_num, obj_id_1, frm1, obj_id_2, frm2
        pass

    pass


if __name__ == '__main__':
    get_dataset('/ais/gobi3/u/mren/data/kitti/tracking/training')
