import tfplus

class TrackingDataAssembler(object):
    """
    video_id/video/frm_{frame_id}: PNG encoded image
    video_id/annotations/obj_{object_id}/bbox: {left, top, right, bottom} * num_frames
    video_id/annotations/obj_{object_id}/presence: 1/0 * num_frames
    """

    def __init__(self, output_fname):
        self.log = tfplus.utils.logger.get()
        self.output_fname = output_fname
        self.log.info('Output h5 dataset: {}'.format(self.output_fname))
        self.log.info('Reading video IDs')
        self.vid_ids = self.get_video_ids()
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

    def get_obj_data(self, vid_id, frm_id, obj_id):
        raise Exception('Not implemented')

    def assemble(self):
        num_vid = len(self.vid_ids)

        with h5py.File(self.output_fname, 'a') as h5f:
            for idx in pb.get(num_vid):
                vid_id = self.vid_ids[idx]
                obj_ids = self.get_obj_ids(vid_id)
                for frm_id in self.get_frame_ids(vid_id):
                    frm_img = self.get_frame_img(vid_id, frm_id)
                    img_enc = self.encode(frm_img)
                    frm_key = '{}/video/frm_{}'.format(vid_id, frm_id)
                    self.save(frm_key, img_enc)
                for obj_id in obj_ids:
                    obj_key =  '{}/annotations/obj_{}'.format(vid_id, obj_id)
                    obj_data = self.get_obj_data(vid_id, obj_id)
                    if obj_data is not None:
                        self.save(obj_key + '/bbox', obj_data['bbox'])
                        self.save(obj_key + '/presence',
                                  obj_data['presence'])
                    pass
                pass
            pass
        pass
