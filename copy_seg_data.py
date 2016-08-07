import cslab_environ

import cv2
import os
import h5py
import tfplus.utils.progress_bar as pb

# seg_data_folder = '/ais/gobi4/mren/results/insseg_tracking'
seg_data_folder = '/ais/gobi4/mren/data/kitti/tracking'
track_data_folder = '/ais/gobi4/mren/data/kitti/tracking'
# for split in ['train', 'valid', 'test']:
# for split in ['valid']:
for split in ['train', 'test']:
    track_h5_fname = os.path.join(track_data_folder, split + '.h5')
    print 'Writing to', track_h5_fname
    with h5py.File(track_h5_fname, 'a') as track_file:
        seqs = track_file.keys()
        print seqs
        for seq in pb.get_iter(seqs):
            if split == 'valid':
                split2 = 'train'
            else:
                split2 = split
            seg_h5_fname = os.path.join(
                track_data_folder, '_'.join([split2, seq, '128x448.h5']))
            with h5py.File(seg_h5_fname, 'r') as seg_file:
                read_keys = seg_file.keys()
                for rk in read_keys:
                    if not rk.startswith('0'):
                        continue
                    wk = '/'.join([seq, 'video', 'frm_{}'.format(rk)])
                    fgk = '/foreground_pred'
                    # fg = seg_file[rk + fgk][:]
                    # img = track_file[wk + '/image'][:]
                    # img = cv2.imdecode(img, -1)
                    # fg = cv2.imdecode(fg, -1)
                    # fg = cv2.resize(fg, (img.shape[1], img.shape[0]))
                    # track_file[wk + fgk] = cv2.imencode('.png', fg)[1]
                    track_file[wk + fgk] = seg_file[rk + fgk][:]
                    for angle in xrange(8):
                        orik = '/orientation_pred/{:02d}'.format(angle)
                        # ori = seg_file[rk + orik][:]
                        # ori = cv2.imdecode(ori, -1)
                        # ori = cv2.resize(ori, (img.shape[1], img.shape[0]))
                        # track_file[wk + orik] = cv2.imencode('.png', ori)[1]
                        track_file[wk + orik] = seg_file[rk + orik][:]
