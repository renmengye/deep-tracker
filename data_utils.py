import h5py
import logger
import os


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
    """Write a dataset stored in H5."""
    log.info('Writing dataset to {}'.format(h5_fname))
    h5f = h5py.File(h5_fname, 'w')
    for key in dataset.iterkeys():
        h5f[key] = dataset[key]

    h5f.close()
