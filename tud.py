import cv2
import h5py
import logger
import numpy as np
import os
import xml.etree.ElementTree

log = logger.get()


def read_h5_data(h5_fname):
    """Read a dataset stored in H5."""
    if os.path.exists(h5_fname):
        log.info('Reading dataset from {}'.format(h5_fname))
        h5f = h5py.File(h5_fname, 'r')
        dataset = {}
        for key in h5f.keys():
            dataset[key] = h5f[key][:]
            pass

        return dataset

    else:
        return None


def write_h5_data(h5_fname, dataset):
    log.info('Writing dataset to {}'.format(h5_fname))
    h5f = h5py.File(h5_fname, 'w')
    for key in dataset.iterkeys():
        h5f[key] = dataset[key]

        pass


def get_dataset(folder):
    h5_fname = os.path.join(folder, 'dataset.h5')
    cache = read_h5_data(h5_fname)
    if cache:
        return cache
    xml_fname = os.path.join(folder, 'TUD-Stadtmitte.xml')
    tree = xml.etree.ElementTree.parse(xml_fname).getroot()
    obj_data = {}
    frame_start = None
    frame_end = None
    for frame in tree.findall('frame'):
        nframe = int(frame.attrib['number'])
        if frame_start is None:
            frame_start = nframe
            frame_end = nframe
        else:
            frame_start = min(frame_start, nframe)
            frame_end = max(frame_end, nframe)
        for obj_list in frame.findall('objectlist'):
            for obj in obj_list.findall('object'):
                idx = int(obj.attrib['id'])
                for box in obj.findall('box'):
                    h = float(box.attrib['h'])
                    w = float(box.attrib['h'])
                    x = float(box.attrib['xc'])
                    y = float(box.attrib['yc'])
                if idx in obj_data:
                    obj_data[idx].append((nframe, h, w, x, y))
                else:
                    obj_data[idx] = [(nframe, h, w, x, y)]

                print nframe, idx, h, w, x, y

    log.info('frame_start: {}'.format(frame_start))
    log.info('frame_end: {}'.format(frame_end))

    nframe = frame_end - frame_start + 1
    idx_map = []
    num_idx = len(obj_data.keys())
    data = np.zeros([num_idx, nframe, 5], dtype='float32')
    for idx in obj_data.iterkeys():
        new_idx = len(idx_map)
        for dd in obj_data[idx]:
            new_frame = dd[0] - frame_start
            data[new_idx, new_frame, 4] = 1.0
            data[new_idx, new_frame, 0] = dd[1]
            data[new_idx, new_frame, 1] = dd[2]
            data[new_idx, new_frame, 2] = dd[3]
            data[new_idx, new_frame, 3] = dd[4]
        idx_map.append(idx)

    idx_map = np.array(idx_map, dtype='uint8')
    frame_map = np.arange(frame_start, frame_end + 1)

    log.info('Reading images')
    im_height = None
    im_width = None
    im_channels = None
    imgs = []
    for frame in xrange(frame_start, frame_end + 1):
        image_fname = os.path.join(folder,
                                   'DaMultiview-seq{}.png'.format(frame))
        log.info(image_fname)
        img = cv2.imread(image_fname)
        if im_height is None:
            im_height = img.shape[0]
            im_width = img.shape[1]
            im_channels = img.shape[2]
        imgs.append(img)

    log.info('Assemble dataset')
    images = np.zeros([len(imgs), im_height, im_width, im_channels],
                      dtype='uint8')
    for ii, img in enumerate(imgs):
        images[ii] = img

    dataset = {
        'images': images,
        'idx_map': idx_map,
        'frame_map': frame_map
    }

    write_h5_data(h5_fname, dataset)

    return dataset

if __name__ == '__main__':
    folder = '/ais/gobi4/rjliao/Projects/CSC2541/data/TUD/cvpr10_tud_stadtmitte'
    get_dataset(folder)
