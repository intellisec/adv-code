import os
import json
from utils import compress_file, get_logger
from concurrent.futures import Executor
from typing import Optional, Iterable

logger = get_logger(__name__)


class SerializationBuffer(object):
    """
    Class to iteratively serialize samples to disk in a format
    that can be read by datasets.load_dataset.
    Using this as a context manager ensures that the last samples
    are written out when the context is exited.

    TODO: Enrich serialized dataset with metadata
    Currently when loading the dataset, the metadata such as column names
    might be missing. HF operations such as filter, remove_columns etc.
    thus will not work.

    Example usage:
    my_iterable = ...
    with SerializationBuffer("mydir", samples_per_file=10*5, compress=True) as buffer:
        for sample in my_iterable:
            buffer.addSample(sample)
    """

    def __init__(self,
                 directory: str,
                 samples_per_file: int = 10**5,
                 compress: bool = False,
                 pool: Optional[Executor] = None):
        self.compress = compress
        self.directory = directory
        assert samples_per_file > 0
        # If path exists and is not empty, do not accept
        if self.directory and os.path.exists(self.directory) and len(os.listdir(self.directory)) > 0:
            raise ValueError(f"Directory {self.directory} is not empty")
        self.samples_per_file = samples_per_file
        self.file_number = 0
        self.sampleBuffer = []
        self.pool = None
        self.counter = 0

        if self.directory is not None:
            # not a dummy buffer
            os.makedirs(self.directory, exist_ok=True)
            self.pool = pool

    def addSample(self, sample: any):
        # Stage a single sample for serialization
        # TODO: Ponder which types make sense for sample (except dict)
        if self.directory is None:
            # this is a null buffer, do nothing
            return
        if isinstance(sample, dict):
            self.sampleBuffer.append(sample)
        else:
            self.sampleBuffer.append({"content": sample})
        self.counter += 1
        if len(self.sampleBuffer) >= self.samples_per_file:
            self._writeout()

    def addSamples(self, samples: Iterable[any]):
        # dumb wrapper around addSample
        for sample in samples:
            self.addSample(sample)

    def _buildFileName(self, file_number: int):
        assert self.directory is not None
        return os.path.join(self.directory, f"file-{file_number+1:012}.json")

    def _writeout(self):
        if len(self.sampleBuffer) == 0:
            return
        # Write out the current buffer to disk
        filename = self._buildFileName(self.file_number)
        logger.debug(f"Writing out {len(self.sampleBuffer)} samples to {filename}")
        with open(filename, "w") as f:
            for sample in self.sampleBuffer:
                f.write(json.dumps(sample) + "\n")
        assert os.path.isfile(filename)

        # saving a raw file and then compressing proved way faster than using gzip directly
        if self.compress:
            if self.pool is not None:
                logger.debug(f"Submitting compression job for {filename}")
                self.pool.submit(compress_file, filename)
            else:
                logger.debug(f"Compressing {filename}")
                compress_file(filename)
        self.sampleBuffer.clear()
        self.file_number += 1
        assert len(self.sampleBuffer) == 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug("Writing out remaining samples")
        self._writeout()
        logger.info(f"Serialized {self.counter} samples to {self.directory}")


def serializeDataSet(dataset: Iterable[any],
                     directory: str,
                     compress: bool = False):
    with SerializationBuffer(directory,
                             samples_per_file=10**5,
                             compress=compress) as buffer:
        previousType = None
        for sample in dataset:
            # This could be streamlined, but it shouldn't be a bottleneck anyway
            if previousType is not None:
                assert previousType == type(sample), f"Expected type {previousType}, got {type(sample)}"
            else:
                previousType = type(sample)
            if isinstance(sample, dict):
                buffer.addSample(sample)
            else:
                buffer.addSample({"content": sample})
