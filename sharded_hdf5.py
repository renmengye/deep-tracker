"""
Sharded HDF5 format
Storing of a bundle of sharded files for homogeneous data type.

==File system structure
    1. File pattern
    /path/folder/file_prefix-{index}-{num_shards}{suffix}
    {index} and {num_shards} are 5 digit 0-padded integer string.
    e.g. 
    - /path/folder/example-00000-of-00005.h5
    - /path/folder/example-00001-of-00005.h5
    - /path/folder/example-00002-of-00005.h5
    - /path/folder/example-00003-of-00005.h5
    - /path/folder/example-00004-of-00005.h5

    2. HDF5 structure
    {
        '__num_items__': 1 elem numpy.ndarray, number of items in this file.
        '__keys__': 1D numpy.ndarray, look up keys.
        '__sep_key1__': 1D int64 array storing position of each item.
        '__sep_key2__': 1D int64 array storing position of each item.
        'key1': 1-2D numpy.ndarray, 1st dimension is concatenated.
        'key2': 1-2D numpy.ndarray, 1st dimension is concatenated.
        ...
    }

    3. Object mapping
    {
        'key1': 1-2D numpy.ndarray, maps to 2D numpy.ndarray in the file.
        'key2': string, int, float, maps to 1D numpy.ndarray in the file.
        ...
    }

==Key classes
    1. ShardedFile
    2. ShardedFileReader
    3. ShardedFileWriter

==Examples
    1. Read: iterate everything
    >> f = ShardedFile('a', num_shards=100)
    >> with ShardedFileReader(f) as reader:
    >>    for item in reader:
    >>        do(items)

    2. Read: iterate everything in a batch
    >> f = ShardedFile('a', num_shards=100)
    >> with ShardedFileReader(f, batch_size=10) as reader:
    >>     for items in reader:
    >>         do(items)

    3. Read: iterate from a position
    >> f = ShardedFile('a', num_shards=100)
    >> with ShardedFileReader(f) as reader:
    >>     for item in reader.seek(50):
    >>         do(item)

    4. Read: random access with a position
    >> f = ShardedFile('a', num_shards=100)
    >> with ShardedFileReader(f) as reader:
    >>     reader.seek(pos=position)
    >>     items = reader.read(num_items=100)

    5. Read: random access with a key (string or int)
    >> f = ShardedFile('a', num_shards=100)
    >> with ShardedFileReader(f) as reader:
    >>     item = reader[key]

    6. Write a list
    >> f = ShardedFile('a', num_shards=100)
    >> with ShardedFileWriter(f, num_objects=1000) as writer:
    >>     for i in xrange(1000):
    >>         writer.write(item[i])

    7. Write a dictionary (see example 5 for reading dictionary)
    >> f = ShardedFile('a', num_shards=100)
    >> with ShardedFileWriter(f, num_objects=1000) as writer:
    >>     for i in xrange(1000):
    >>         writer.write(item, key=key)
"""


import bisect
import fnmatch
import h5py
import logger
import math
import numpy
import os
import re

log = logger.get()

ERR_MSG_IDX_TOO_LARGE = 'Shard index larger than number of shards: {}'
ERR_MSG_IDX_TOO_LARGE2 = 'Shard index larger than number of shards'
ERR_MSG_MISSING_FILE = 'Missing file for index {:d} out of {:d}'
ERR_MSG_DUPE_FILES = 'Duplicate files found for index {:d}'
ERR_MSG_MISSING_NUM_ITEMS_FIELD = 'Missing field "num_items" in file {}'
KEY_NUM_ITEM = '__num_items__'
KEY_SEPARATOR = '__sep_{}__'
KEY_SEPARATOR_RE = re.compile('^__sep_([^_]+)__$')
KEY_SEPARATOR_PREFIX = '__sep_'
KEY_KEYS = '__keys__'
FILE_PATTERN = re.compile(
    '^(?P<prefix>.*)-(?P<shard>[0-9]{5})-of-(?P<total>[0-9]{5})(?P<suffix>.*)$')


def _get_sep_from_key(key):
    """Get separator key from key name"""
    return KEY_SEPARATOR.format(key)


def _get_key_from_sep(sep):
    """Get key name from separator key"""
    match = fname_re.match(fname)
    if match is not None:
        return match.groups()[0]
    else:
        return None


class ShardedFile(object):
    """Sharded file object."""

    def __init__(self, file_prefix, num_shards, suffix='.h5'):
        """Construct a sharded file instance."""
        if not isinstance(num_shards, int):
            raise Exception('Number of shards need to be integer')
        self.file_prefix = os.path.abspath(file_prefix)
        self.basename = os.path.basename(self.file_prefix)
        self.num_shards = num_shards
        self.suffix = suffix

        pass

    def __str__(self):
        return '{}-?????-of-{:05d}{}'.format(
            self.basename, self.num_shards, self.suffix)

    @classmethod
    def from_pattern(cls, file_pattern):
        """Initialize a sharded file object with file pattern.

        Note:
            File pattern must look same as the format:
            file_pattern = {prefix}-?????-of-{num_shards}{suffix}
        Example:
            file_pattern = prefix-?????-of-00010.h5
        """
        match = FILE_PATTERN.match(file_pattern)
        if match:
            return cls(file_prefix=match.groupdict()['prefix'],
                       num_shards=int(match.groupdict()['total']),
                       suffix=match.groupdict()['suffix'])

    @classmethod
    def from_pattern_read(cls, file_pattern):
        """Initialize a sharded file object with file pattern.

        Note:
            Pattern can be general wildcard but files must already exist.
        Example:
            file_pattern = prefix*
        """
        flist = []
        dirname = os.path.dirname(file_pattern)

        for fname in os.listdir(dirname):
            fullname = os.path.join(dirname, fname)
            if fnmatch.fnmatch(fullname, file_pattern):
                flist.append(fullname)

        prefix = None
        suffix = None
        num_shards = 0

        if len(flist) == 0:
            raise Exception('No file pattern found: {}'.format(file_pattern))

        for fname in flist:
            match = FILE_PATTERN.match(fname)
            if match is not None:
                if prefix is None:
                    prefix = match.groupdict()['prefix']
                elif prefix != match.groupdict()['prefix']:
                    raise Exception(
                        'Found two different prefixes in the file pattern.')

                if suffix is None:
                    suffix = match.groupdict()['suffix']
                elif suffix != match.groupdict()['suffix']:
                    raise Exception(
                        'Found two different suffixes in the file pattern.')

                if num_shards == 0:
                    num_shards = int(match.groupdict()['total'])
                elif num_shards != int(match.groupdict()['total']):
                    raise Exception(
                        'Found two different total number of shards.')
            else:
                raise Exception('Incorrect file pattern: {}'.format(fname))

        return cls(file_prefix=prefix, num_shards=num_shards, suffix=suffix)

    def get_fname(self, shard):
        """Get the file name for a specific shard.

        Args:
            shard: int, shard index.
        """
        # log.error('current shard: {:d}'.format(shard))
        # log.error('total shard: {:d}'.format(self.num_shards))

        if shard >= self.num_shards:
            raise Exception(ERR_MSG_IDX_TOO_LARGE2)

        return '{}-{:05d}-of-{:05d}{}'.format(
            self.file_prefix, shard, self.num_shards, self.suffix)


class ShardedFileReader(object):
    """Shareded file reader.
    """

    def __init__(self, sharded_file,
                 key_name=KEY_KEYS, batch_size=1, check=True):
        """Construct a sharded file reader instance.

        Args:
            sharded_file: SharededFile instance.
            key_name: Name of key field for random access (optional).
            batch_size: number, average batch_size for each read. The actual 
            size depends on the number of items in a file so is not guaranteed 
            to be the same size.
        """
        self.file = sharded_file

        # Batch size of reading.
        self.batch_size = batch_size

        # Position index in each file for binary search.
        self._file_index = None

        # Reader position.
        self._pos = 0

        # Current file ID.
        self._cur_fid = 0

        # Current file handler.
        self._fh = None

        # Whether need to refresh file handler in the next read.
        self._need_refresh = False

        # Current file separator.
        self._cur_sep = {}

        # Index based on keys.
        self._key_index = None

        # Name of the key field.
        self._key_name = key_name

        # Check files all exist.
        if check:
            self._check_files()

        pass

    def __iter__(self):
        """Get an iterator."""
        return self

    def __getitem__(self, key):
        """Get item based on key"""
        return self.read_key(key)

    def __enter__(self):
        """Enter with clause."""
        return self

    def __exit__(self, type, value, traceback):
        """Exit with clause."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

        pass

    def __len__(self):
        """Get total number of items."""
        return self.get_num_items()

    def __contains__(self, key):
        """Check whether a key is contained in the file."""
        # Lazy build key index.
        if self._key_index is None:
            if self._key_name:
                self._build_key(self._key_name)
            else:
                raise Exception(
                    'You need to specify key field in the constructor.')

        return key in self._key_index

    def _check_files(self):
        """Check existing files"""

        fname_re = re.compile(
            '({})-([0-9]{{5}})-of-{:05d}{}'.format(
                self.file.basename,
                self.file.num_shards,
                self.file.suffix))
        dirname = os.path.dirname(os.path.abspath(self.file.file_prefix))
        files_in_dir = os.listdir(dirname)
        found_files = {}

        # Look for files in the expected pattern.
        for fname in files_in_dir:
            match = fname_re.match(fname)
            if match is not None:
                index_str = match.groups()[1]
                index_int = int(index_str)
                if index_int >= self.file.num_shards:
                    raise Exception(ERR_MSG_IDX_TOO_LARGE.format(fname))
                if index_int in found_files:
                    raise Exception(ERR_MSG_DUPE_FILES.format(index_int))
                found_files[index_int] = fname

        # Check all files in the sequence exist.
        for idx in xrange(self.file.num_shards):
            if idx not in found_files:
                raise Exception(ERR_MSG_MISSING_FILE.format(
                    idx, self.file.num_shards))

        log.info('Check file success: {}'.format(self.file.file_prefix))

        pass

    def _build_index(self):
        """Build a mapping from an index to shard number.

        Returns:
            file_index: list, end element id - 1 of each shard.
        """
        log.info('Building index of file {}'.format(self.file.basename))
        file_index = []
        index = 0
        for shard_idx in xrange(self.file.num_shards):
            fname = self.file.get_fname(shard_idx)
            fh = h5py.File(fname, 'r')
            if KEY_NUM_ITEM in fh:
                num_items = fh[KEY_NUM_ITEM][0]
            else:
                raise Exception(ERR_MSG_MISSING_NUM_ITEMS_FIELD.format(fname))
            index += num_items
            file_index.append(index)

        return file_index

    def _build_key(self, key_name):
        """Build a mapping from a key to shard number and position.

        Args:
            key_name: string, name of the key field.
        """
        log.info('Building key index of file {}'.format(self.file.basename))
        self._key_index = {}
        for shard_idx in xrange(self.file.num_shards):
            fname = self.file.get_fname(shard_idx)
            fh = h5py.File(fname, 'r')
            if key_name in fh:
                key_index_i = fh[key_name][:]
                num_keys = key_index_i.shape[0]

                if KEY_NUM_ITEM in fh:
                    num_items = fh[KEY_NUM_ITEM][0]
                else:
                    raise Exception(
                        ERR_MSG_MISSING_NUM_ITEMS_FIELD.format(fname))

                if num_keys != num_items:
                    raise Exception(
                        'Number of keys not equal to number of items')

                for key_idx in xrange(key_index_i.shape[0]):
                    k = key_index_i[key_idx]
                    self._key_index[k] = (shard_idx, key_idx)
            else:
                raise Exception(
                    'Key "{}" not found in the file {}'.format(
                        key_name, fname))

        pass

    def find(self, index):
        """Find the file id.

        Args:
            index: number, item index.

        """
        # Lazy build file index.
        if self._file_index is None:
            self._file_index = self._build_index()

        return bisect.bisect_left(self._file_index, index)

    def _renew(self):
        """Move to next file."""
        self._cur_fid += 1
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._fh = h5py.File(self.file.get_fname(self._cur_fid), 'r')
        self._need_refresh = False

        pass

    def _build_sep(self):
        """Build separators."""
        num_items = self._fh[KEY_NUM_ITEM][0]
        for key in self._fh.keys():
            if not key.startswith('__'):
                sepname = _get_sep_from_key(key)
                if sepname in self._fh:
                    self._cur_sep[key] = self._fh[sepname][:]
                else:
                    if self._fh[key].shape[0] != num_items:
                        raise Exception('Unknown sep {}'.format(key))
                    else:
                        self._cur_sep[key] = numpy.arange(num_items)

        pass

    def _read_item(self, idx):
        result = {}
        for key in self._fh.keys():
            if not key.startswith('__'):
                # Compute line start and end.
                if idx == 0:
                    line_start = 0
                else:
                    line_start = self._cur_sep[key][idx - 1]
                line_end = self._cur_sep[key][idx]
                if line_start == line_end - 1:
                    result[key] = self._fh[key][line_start]
                else:
                    result[key] = self._fh[key][line_start: line_end]

        return result

    def read(self, num_items=1):
        """Read from the current position.

        Args:
            num_items: number, number of desired items to read. It is not 
            guaranteed to return the exact same number of items.
        Returns:
            results: list of dict, keys are same with the keys defined in the 
            file, values are numpy.ndarray.
        """
        # Lazy build file index.
        if self._file_index is None:
            self._file_index = self._build_index()

        # Renew file ID.
        if self._need_refresh:
            self._renew()

        # Open a file.
        if self._fh is None:
            self._fh = h5py.File(self.file.get_fname(self._cur_fid), 'r')
            self._build_sep()

        # Compute file_start and file_end (absolute cursor) and
        # item_start and item_end (relative cursor).
        if self._cur_fid == 0:
            file_start = 0
        else:
            file_start = self._file_index[self._cur_fid - 1]
        file_end = self._file_index[self._cur_fid]

        item_start = self._pos - file_start
        item_end = min(self._pos + num_items, file_end) - file_start

        # Refresh next time if reached the end.
        if item_end == file_end - file_start:
            self._need_refresh = True

        # log.error('fn: {}'.format(self.file))
        # log.error('fs: {:d}'.format(file_start))
        # log.error('fe: {:d}'.format(file_end))
        # log.error('is: {:d}'.format(item_start))
        # log.error('ie: {:d}'.format(item_end))

        # Read data.
        results = []
        for i, idx in enumerate(xrange(item_start, item_end)):
            results.append(self._read_item(idx))

        self._pos += num_items

        if num_items == 1:
            return results[0]
        else:
            return results

    def read_key(self, key):
        """Read an item based on key.

        Args:
            key: string, key of the item.
        Returns:
            results: dict.
        """
        # Lazy build file index.
        if self._file_index is None:
            self._file_index = self._build_index()

        # Lazy build key index.
        if self._key_index is None:
            if self._key_name:
                self._build_key(self._key_name)
            else:
                raise Exception(
                    'You need to specify key field in the constructor.')

        if key not in self._key_index:
            log.warning('Key {} not found in file {}'.format(key, self.file))
            return None
        else:
            location = self._key_index[key]
            fid = location[0]
            pos = location[1]
            # log.error('fid: {:d} pos: {:d}'.format(fid, pos))
            if fid != self._cur_fid:
                self._cur_fid = fid
                self._fh = None
            if fid != 0:
                file_start = self._file_index[fid - 1]
                pos = pos + file_start
            self._pos = pos

        # Disable refresh in key reading mode.
        self._need_refresh = False

        return self.read(num_items=1)

    def keys(self):
        """Get a list of keys."""
        # Lazy build key index.
        if self._key_index is None:
            if self._key_name:
                self._build_key(self._key_name)

        return self._key_index.keys()

    def iterkeys(self):
        """Get an iterable of keys."""
        # Lazy build key index.
        if self._key_index is None:
            if self._key_name:
                self._build_key(self._key_name)

        return self._key_index.iterkeys()

    def seek(self, pos):
        """Seek to specific position.

        Args:
            pos: number, position in terms of number of items.
        Returns:
            A SharededReader instance.
        """
        self._pos = pos
        fid = self.find(self._pos)
        if fid != self._cur_fid or self._fh is None:
            self._cur_fid = fid
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self._fh = h5py.File(self.file.get_fname(fid), 'r')

        return self

    def next(self):
        """Iterate to next batch to read.
        """
        # Lazy build file index.
        if self._file_index is None:
            self._file_index = self._build_index()

        if self._pos < self._file_index[-1]:
            return self.read(self.batch_size)
        else:
            raise StopIteration()

        pass

    def get_num_items(self):
        """Get total number of items.
        """
        # Lazy build file index.
        if self._file_index is None:
            self._file_index = self._build_index()

        return self._file_index[-1]

    def close(self):
        self.__exit__(None, None, None)

        pass


class ShardedFileWriter(object):
    """Sharded file writer."""

    def __init__(self, sharded_file, num_objects):
        """Construct a sharded file writer instance.

        Args:
            sharded_file: ShardedFile instance.
            num_objects: number, total number of objects to write.
        """
        self.file = sharded_file

        # Total number of items to write.
        self._num_objects = num_objects

        # Total number of shards.
        self._num_shards = self.file.num_shards

        # Number of items per shard.
        self._num_objects_per_shard = int(
            math.ceil(num_objects / float(self._num_shards)))

        # Current file handler.
        self._fh = None

        # Current item index.
        self._pos = 0

        # Current shard index.
        self._shard = 0

        # File buffer for current shard.
        self._buffer = {}

        # Number of items for current shard.
        self._cur_num_items = 0

        # Index separator for current shard.
        self._cur_sep = {}

        # Set of keys used.
        self._keys = set()

        pass

    def __enter__(self):
        """Enter with clause."""
        return self

    def __exit__(self, type, value, traceback):
        """Exit with clause."""
        self._flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

        pass

    def write(self, data, key=None):
        """Write a single entry into buffer.

        Args:
            data: numpy.ndarray or int or string, data entry.
            key: (optional), int or string, key for data entry, default is the 
            0-based index.
        """
        if self._fh is None:
            self._fh = h5py.File(self.file.get_fname(self._shard), 'w')

        # Assign numerical key.
        if key is None:
            key = self._pos

        # Check key exists.
        if key in self._keys:
            raise Exception('Key already exists: {}.'.format(key))

        # Check data format.
        for kkey in data.iterkeys():
            if kkey.startswith('__'):
                raise Exception(
                    'Keys must not start with "__": {}'.format(kkey))
            if len(self._buffer) > 0:
                if kkey not in self._buffer:
                    raise Exception('Unknown key: {}'.format(kkey))

        # Assign key.
        if KEY_KEYS in self._buffer:
            self._buffer[KEY_KEYS].append(key)
        else:
            self._buffer[KEY_KEYS] = [key]

        for kkey in data.iterkeys():
            if kkey in self._buffer:
                self._buffer[kkey].append(data[kkey])
            else:
                self._buffer[kkey] = [data[kkey]]

            if kkey not in self._cur_sep:
                self._cur_sep[kkey] = []

            shape0 = data[kkey].shape[0] if isinstance(
                data[kkey], numpy.ndarray) else 1
            if len(self._cur_sep[kkey]) > 0:
                last = self._cur_sep[kkey][-1]
                self._cur_sep[kkey].append(last + shape0)
            else:
                self._cur_sep[kkey].append(shape0)

        # Increment counter.
        self._cur_num_items += 1
        self.next()

        pass

    def _flush(self):
        """Flush the buffer into the current shard."""
        if len(self._buffer) > 0:
            for key in self._buffer.iterkeys():
                if isinstance(self._buffer[key][0], numpy.ndarray):
                    value = numpy.concatenate(self._buffer[key], axis=0)
                elif isinstance(self._buffer[key][0], str):
                    value = numpy.array(self._buffer[key], dtype='string')
                elif isinstance(self._buffer[key][0], int):
                    value = numpy.array(self._buffer[key])
                elif isinstance(self._buffer[key][0], float):
                    value = numpy.array(self._buffer[key])
                else:
                    raise Exception('Unknown type: {}'.format(
                        type(self._buffer[key][0])))
                self._fh[key] = value
            self._fh[KEY_NUM_ITEM] = numpy.array([self._cur_num_items])
            for key in self._cur_sep.iterkeys():
                sepname = _get_sep_from_key(key)
                self._fh[sepname] = numpy.array(
                    self._cur_sep[key], dtype='int64')
            self._cur_num_items = 0
            self._cur_sep = {}
            self._buffer = {}

        pass

    def seek(self, pos, shard):
        """Seek to a position."""
        self._shard = shard
        self._pos = pos
        if self._fh is not None:
            self._fh.close()
            self._fh = None

        pass

    def next_file(self):
        """Move to writing the next shard."""
        if self._cur_num_items > 0:
            self._flush()

        self._shard += 1
        if self._shard >= self.file.num_shards:
            raise Exception(ERR_MSG_IDX_TOO_LARGE2)
        if self._fh is not None:
            self._fh.close()
            self._fh = None

        pass

    def next(self):
        """Move to writing the next object."""

        if self._pos < self._num_objects:
            r = self._pos - self._shard * self._num_objects_per_shard
            if r == self._num_objects_per_shard:
                self.next_file()
            i = self._pos
            self._pos += 1
            return i
        else:
            raise Exception(
                'Exceeded initialized capacity {}'.format(self._num_objects))
            # raise StopIteration()

        pass

    def close(self):
        """Close the opened file."""
        self.__exit__(None, None, None)

        pass
