from __future__ import division

import cslab_environ
import cv2
import numpy as np
import os
import tfplus
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
        self.anns = {}

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
        label_fname = os.path.join(self.label_folder, vid_id + '.txt')
        target_types = set(['Van', 'Car', 'Truck'])
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

        for idx in obj_data.iterkeys():
            new_idx = len(idx_map)
            for dd in obj_data[idx]:
                new_frame = dd['frame_no'] - frame_start
                bbox[new_idx, new_frame, 4] = 1.0
                bbox[new_idx, new_frame, 0: 4] = dd['bbox']
            idx_map.append(idx)
        idx_map = np.array(idx_map, dtype='uint8')
        frame_map = np.arange(frame_start, frame_end + 1)

        print 'Index Map', idx_map
        print 'Frame Map', frame_map
        self.anns[vid_id] = bbox
        pass

    def get_obj_ids(self, vid_id):
        if self.label_folder is None:
            return None
        if vid_id not in self.anns:
            self._read_annotations(vid_id)
        return ['{:04d}'.format(x) for x in xrange(self.anns[vid_id].shape[0])]

    def get_obj_data(self, vid_id, obj_id):
        if self.label_folder is None:
            return None

        obj_idx = int(obj_id)
        results = {
            'bbox': self.anns[vid_id][obj_idx, :, :4],
            'presence': self.anns[vid_id][obj_idx, :, 4]
        }
        return results

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
    print assembler.get_obj_data('0001', '0001')
    assembler.assemble()
    pass
