import os
import datetime
import logger

log = logger.get()

def log_register(filename, typ, name):
    """Register a new file in the catalog

    Args:
        filename: string, path to the log file.
        typ: string, file type, "csv" or "plain" or "image".
        name: string, name of the visualization.
    """
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    catalog = os.path.join(folder, 'catalog')
    basename = os.path.basename(filename)
    if not os.path.exists(catalog):
        with open(catalog, 'w') as f:
            f.write('filename,type,name\n')
            f.write('{},{},{}\n'.format(basename, typ, name))
    else:
        with open(catalog, 'a') as f:
            f.write('{},{},{}\n'.format(basename, typ, name))


class TimeSeriesLogger():
    """Log time series data to CSV file."""

    def __init__(self, filename, labels, name=None, buffer_size=100):
        """
        Args:
            label: list of string
            name: string
        """
        self.filename = filename
        self.written_catalog = False
        if type(labels) != list:
            labels = [labels]
        if name is None:
            self.name = labels[0]
        else:
            self.name = name
        self.labels = labels
        self.buffer = []
        self.buffer.append('step,time,{}\n'.format(','.join(self.labels)))
        self.buffer_size = buffer_size
        log.info('Time series data "{}" log to "{}"'.format(labels, filename))

        if os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                f.write('')
        else:
            log_register(self.filename, 'csv', self.name)

        pass

    def add(self, step, values):
        """Add an entry.

        Args:
            step: int
            value: list of numbers
        """
        t = datetime.datetime.utcnow()
        if type(values) != list:
            values = [values]
        self.buffer.append('{:d},{},{}\n'.format(
            step, t.isoformat(), ','.join([str(v) for v in values])))
        if len(self.buffer) >= self.buffer_size:
            self.flush()
        pass

    def flush(self):
        """Write the buffer to file."""
        if not os.path.exists(self.filename):
            mode = 'w'
        else:
            mode = 'a'
        with open(self.filename, mode) as f:
            f.write(''.join(self.buffer))
        self.buffer = []
        pass

    def close(self):
        """Flush the rest."""
        self.flush()
        pass
