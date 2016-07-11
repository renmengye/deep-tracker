import cv2
import h5py
import numpy as np
import tfplus
import tfplus.utils.progress_bar as pb


class TrackingDataAssembler(object):
    """
    {video_id}/video/frm_{frame_id}: PNG encoded image
    {video_id}/annotations/obj_{object_id}/bbox: {left, top, right, bottom} * num_valid_frames
    {video_id}/annotations/obj_{object_id}/presence: frame indices of appearance
    """

    def __init__(self, output_fname):
        self.log = tfplus.utils.logger.get()
        self.output_fname = output_fname
        self.log.info('Output h5 dataset: {}'.format(self.output_fname))
        self.log.info('Reading video IDs')
        self.vid_ids = self.get_video_ids()
        self.log.info('Video IDs: {}'.format(self.vid_ids))
        pass

    def save(self, key, data, group):
        if key in group:
            del group[key]
        group[key] = data
        pass

    def encode(self, img):
        return cv2.imencode('.png', img)[1]

    def get_video_ids(self):
        raise Exception('Not implemented')

    def get_frame_ids(self, vid_id):
        raise Exception('Not implemented')

    def get_frame_img(self, vid_id, frm_id):
        raise Exception('Not implemented')

    def get_obj_ids(self, vid_id):
        raise Exception('Not implemented')

    def get_obj_data(self, vid_id, obj_id):
        raise Exception('Not implemented')

    def assemble(self):
        num_vid = len(self.vid_ids)

        with h5py.File(self.output_fname, 'a') as h5f:
            for idx in pb.get(num_vid):
                vid_id = self.vid_ids[idx]
                for frm_id in self.get_frame_ids(vid_id):
                    frm_img = self.get_frame_img(vid_id, frm_id)
                    img_enc = self.encode(frm_img)
                    frm_key = '{}/video/frm_{}'.format(vid_id, frm_id)
                    self.save(frm_key, img_enc, h5f)

                obj_ids = self.get_obj_ids(vid_id)
                if obj_ids is None:
                    # Testing data do not have labels.
                    continue
                for obj_id in obj_ids:
                    obj_data = self.get_obj_data(vid_id, obj_id)
                    obj_key = '{}/annotations/obj_{}'.format(vid_id, obj_id)
                    frm_nonzero = obj_data['presence'].nonzero()[0]
                    self.save(obj_key + '/bbox',
                              obj_data['bbox'][frm_nonzero], h5f)
                    self.save(obj_key + '/frame_indices', frm_nonzero, h5f)
                    pass
                pass
            pass
        pass
