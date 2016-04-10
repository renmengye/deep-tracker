"""
A batch iterator.

Usage:
    for idx in BatchIterator(num=1000, batch_size=25):
        inp_batch = inp_all[idx]
        labels_batch = labels_all[idx]
        train(inp_batch, labels_batch)
"""

import numpy as np
import progress_bar as pb


class BatchIterator(object):

    def __init__(self, num, batch_size=1, progress_bar=False, get_fn=None, cycle=False):
        """Construct a batch iterator.

        Args:
            data: numpy.ndarray, (N, D), N is the number of examples, D is the
            feature dimension.
            labels: numpy.ndarray, (N), N is the number of examples.
            batch_size: int, batch size.
        """

        self._num = num
        self._batch_size = batch_size
        self._step = 0
        self._num_steps = int(np.ceil(self._num / float(batch_size)))
        self._pb = None
        self._get_fn = get_fn
        self._cycle = cycle
        if progress_bar:
            self._pb = pb.get(self._num_steps)

        pass

    def __iter__(self):
        """Get iterable."""
        return self

    def __len__(self):
        """Get iterable length."""
        return self._num_steps

    def next(self):
        """Iterate next element."""
        if self._pb:
            self._pb.increment()
        if not self._cycle:
            if self._step < self._num_steps:
                start = self._batch_size * self._step
                end = min(self._num, self._batch_size * (self._step + 1))
                self._step += 1
                idx = np.arange(start, end)
                if self._get_fn:
                    return self._get_fn(idx)
                else:
                    return idx
            else:
                raise StopIteration()
        else:
            start = (self._batch_size * self._step) % self._num
            end = (self._batch_size * (self._step + 1)) % self._num
            self._step += 1

            if end > start:
                idx = np.arange(start, end)
            else:
                idx = np.array(range(start, self._num) + range(0, end))

            if self._get_fn:
                return self._get_fn(idx)
            else:
                return idx

        pass
