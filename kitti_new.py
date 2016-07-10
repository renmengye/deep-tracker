from __future__ import division

import cslab_environ

import cv2
import h5py
import numpy as np
import os
import tfplus
import tfplus.utils.progress_bar as pb
from tracking_data_assembler import TrackingDataAssembler


class KITTITrackingDataAssembler(TrackingDataAssembler):

    def __init__(self, folder, output_fname=None, split='train'):
        self.folder = folder
        if output_fname is None:
            output_fname = os.path.join(folder, '{}.h5'.format(split))

        if split == 'train' or 'valid':
            self.left_folder = os.path.join(folder, 'training', 'image_02')
            self.label_folder = os.path.join(folder, 'training', 'label_02')
        elif split == 'test':
            self.left_folder = os.path.join(folder, 'testing', 'image_02')
            self.label_folder = None
        else:
            raise Exception('Unknown split "{}"'.format(split))

        super(KITTITrackingDataAssembler, self).__init__(output_fname)
        pass

    def get_video_ids(self):
        return filter(lambda x: x.startswith('0'), os.listdir(self.left_folder))

    def get_frame_ids(self, vid_id):
        vid_folder = os.path.join(self.left_folder, vid_id)
        return sorted(map(lambda x: x[:6], os.listdir(vid_folder)))

    def get_frame_img(self, vid_id, frm_id):
        fname = os.path.join(self.left_folder, vid_id, frm_id + '.png')
        return cv2.imread(fname)

    def _read_annotations(self, vid_id):

        label_fname = os.path.join(label_folder, vid_id + '.txt')
        pass
   
    def get_obj_ids(self, vid_id):
        if self.label_folder is None:
            return None
        if vid_id not in self.anns:
            self._read_annotations(vid_id)
        raise Exception('Not implemented')

    def get_obj_data(self, vid_id, frm_id, obj_id):
        if self.label_folder is None:
            return None
        raise Exception('Not implemented')

    pass


class KITTITrackingDataProvider(tfplus.data.DataProvider):

    def __init__(self):
        pass
    pass

tfplus.data.data_provider.register('kitti_track', KITTITrackingDataProvider)


if __name__ == '__main__':
    assembler = KITTITrackingDataAssembler(
            '/ais/gobi4/mren/data/kitti/tracking')
    print assembler.get_video_ids()
    print assembler.get_frame_ids('0001')
    print assembler.get_obj_ids('0001')
    print assembler.get_obj_data('0001', '000088')
    pass
