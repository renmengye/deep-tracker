from __future__ import division

import cslab_environ

import cv2
import h5py
import numpy as np
import tfplus
import tfplus.utils.progress_bar as pb
from tracking_data_assembler import TrackingDataAssembler


class KITTITrackingDataAssembler(TrackingDataAssembler):

    def __init__(self, folder, output_fname=None, split='train'):
        self.folder = folder
        if output_fname is None:
            output_fname = os.path.join(folder, '{}_{}x{}.h5'.format(split))

        if split == 'train' or 'valid':
            self.left_folder = os.path.join(folder, 'training', 'image_02')
        elif split == 'test':
            self.left_folder = os.path.join(folder, 'testing', 'image_02')
        else:
            raise Exception('Unknown split "{}"'.format(split))

        super(KITTITrackingDataAssembler, self).__init__(output_fname)
        pass

    def get_video_ids(self):
        return os.listdir(self.left_folder).filter(lambda x: x.startswith('0'))

    def get_frame_ids(self, vid_id):
        raise Exception('Not implemented')

    def get_frame_img(self, vid_id, frm_id):
        raise Exception('Not implemented')

    def get_obj_ids(self, vid_id):
        raise Exception('Not implemented')

    def get_obj_data(self, vid_id, frm_id, obj_id):
        raise Exception('Not implemented')

    pass


class KITTITrackingDataProvider(tfplus.data.DataProvider):

    def __init__(self):
        pass
    pass

tfplus.data.data_provider.register('kitti_track', KITTITrackingDataProvider)


if __name__ == '__main__':
    assembler = KITTITrackingDataAssembler(
        output_fname='/ais/gobi4/mren/data/kitti/tracking')
    print assembler.get_video_ids()
    pass
