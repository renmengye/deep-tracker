import cslab_environ

import cv2
import h5py
import numpy as np
import tfplus

tfplus.cmd_args.add('td:window_size', 'int', 20)
tfplus.cmd_args.add('td:inp_height', 'int', 128)
tfplus.cmd_args.add('td:inp_width', 'int', 448)


class TrackingDataProvider(tfplus.data.DataProvider):

    def __init__(self, split='train', filename=None):
        super(TrackingDataProvider, self).__init__()
        self._windows = None
        self._filename = filename
        self._split = split
        self.log = tfplus.utils.logger.get()
        self.register_option('td:window_size')
        self.register_option('td:inp_height')
        self.register_option('td:inp_width')
        self.mode = 'train_dense'
        pass

    @property
    def filename(self):
        return self._filename

    @property
    def windows(self):
        if self._windows is None:
            self._windows = self.compute_windows()
        return self._windows

    @property
    def split(self):
        return self._split

    def get_size(self):
        if self._windows is None:
            self._windows = self.compute_windows()
        return len(self._windows)

    def compute_windows(self):
        """
        Extracts usable windows from the video sequence.

        Args:
            mode: how the windows are selected.
                "train_dense": overlapping windows (stride 1) on valid frame indices.
                "eval_no_overlap": non-overlapping windows on all frames

        Returns:
            windows: list of window metadata.
                "video_id", "object_id", "frame_start"
        """
        if self.mode != 'train_dense':
            raise Exception('Mode "{}" not supported'.format(mode))
        window_size = self.get_option('td:window_size')
        windows = []
        with h5py.File(self.filename, 'r') as f:
            video_ids = f.keys()
            window_count = 0
            for vid in video_ids:
                group = f[vid]['annotations']
                obj_list = group.keys()
                for oid in obj_list:
                    frm_idx = group[oid]['frame_indices']
                    num_val_frm = frm_idx[-1] - frm_idx[0] + 1
                    if num_val_frm < window_size:
                        windows.append({
                            'video_id': vid,
                            'object_id': oid,
                            'frame_start': frm_idx[0]
                        })
                        pass
                    else:
                        for frm_start in xrange(num_val_frm - 4):
                            windows.append({
                                'video_id': vid,
                                'object_id': oid,
                                'frame_start': frm_start
                            })
                            pass
                        pass
                    pass
                self.log.info('Vid {} Windows {}'.format(vid, len(windows) -
                                                         window_count))
                window_count = len(windows)
                pass
            pass
        return windows

    def get_batch_idx(self, idx, **kwargs):
        # Remember that the images are not resized to uniform size.
        # Remember to normalize the bounding box
        # coordinates.
        num_ex = len(idx)
        window_size = self.get_option('td:window_size')
        inp_height = self.get_option('td:inp_height')
        inp_width = self.get_option('td:inp_width')
        images = np.zeros([num_ex, window_size, inp_height, inp_width, 3],
                          dtype='float32')
        bbox = np.zeros([num_ex, window_size, 4], dtype='float32')
        presence = np.zeros([num_ex, window_size], dtype='float32')
        self.log.info(self.filename)

        with h5py.File(self.filename, 'r') as f:
            for kk, ii in enumerate(idx):
                window = self.windows[ii]
                vid = window['video_id']
                oid = window['object_id']
                frm_start = window['frame_start']
                vid_group = f[vid]
                obj_group = vid_group['annotations'][oid]
                val_frm_idx = obj_group['frame_indices'][:]
                num_frm = len(vid_group['video'].keys())
                frm_end = min(frm_start + window_size, num_frm)
                # print frm_start, frm_end
                count = 0
                for jj in xrange(frm_start, frm_end):
                    _img = vid_group['video/frm_{:06d}'.format(jj)][:]
                    _img = cv2.imdecode(_img, -1)
                    orig_height = _img.shape[0]
                    orig_width = _img.shape[1]
                    _img = cv2.resize(_img, (inp_width, inp_height),
                                      interpolation=cv2.INTER_CUBIC)
                    images[kk, jj - frm_start, :, :] = _img
                    val_frm = set(val_frm_idx).intersection(
                        set(range(frm_start, frm_end)))
                    if jj in val_frm_idx:
                        presence[kk, jj - frm_start] = 1.0
                        bbox_ = obj_group['bbox'][count]
                        # Resize boxes.
                        bbox_[0] = bbox_[0] / orig_width * inp_width
                        bbox_[1] = bbox_[1] / orig_height * inp_height
                        bbox_[2] = bbox_[2] / orig_width * inp_width
                        bbox_[3] = bbox_[3] / orig_height * inp_height
                        bbox[kk, jj - frm_start] = bbox_
                        count += 1
                    pass
                pass
            pass
        results = {
            'x': images / 255.0,
            'bbox_gt': bbox,
            's_gt': presence
        }
        return results

if __name__ == '__main__':
    dp = TrackingDataProvider(
        filename='/ais/gobi4/mren/data/kitti/tracking/train.h5').init_from_main()
    size = dp.get_size()
    print 'Number of windows', size
