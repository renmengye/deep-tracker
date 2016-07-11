import h5py
import tfplus

tfplus.cmd_args.add('window_size', 'int', 30)


class TrackingDataProvider(tfplus.data.DataProvider):

    def __init__(self, split='train', filename=None):
        self._windows = None
        self.register_option('window_size')
        pass

    @property
    def filename(self):
        return self._filename

    @property
    def window(self):
        return self._window

    @property
    def split(self):
        return self._split

    def get_size(self):
        if self._windows is None:
            self._windows = self.compute_windows()
        return len(self._windows)

    def compute_windows(self, mode='train_dense'):
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
        if mode != 'train_dense':
            raise Exception('Mode "{}" not supported'.format(mode))
        window_size = self.get_option('window_size')
        windows = []
        with h5py.File(self.filename, 'r') as f:
            video_ids = f.keys()
            for vid in video_ids:
                group = f[vid]['annotations']
                obj_list = group.keys()
                for oid in obj_list:
                    frm_idx = group[oid]['frame_indices']
                    num_val_frm = frm_idx[-1] - frm_idx[0] + 1
                    if num_val_frm < window_size:
                        window.append({
                            'video_id': vid,
                            'object_id': oid,
                            'frame_start': frm_idx[0]
                        })
                        pass
                    else:
                        for frm_start in xrange(num_val_frm - window_size + 1):
                            window.append({
                                'video_id': vid,
                                'object_id': oid,
                                'frame_start': frm_start
                            })
                            pass
                        pass
                    pass
                pass
            pass
        pass

    def get_batch(self, idx, **kwargs):
        # Remember that the images are not resized to uniform size.
        # Remember to normalize the bounding box coordinates.
        with h5py.File(self.filename, 'r') as f:
            for ii in idx:
                window = self.window[ii]

        pass
    pass
