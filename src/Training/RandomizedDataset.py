from torch.utils.data import Dataset
from typing import Optional, Union
from utils import get_logger
import random

logger = get_logger(__name__)


class RandomizedDataset(Dataset):
    """
    This is a wrapper for any torch dataset which always returns random elements.

    This is intended to be used in conjunction with the HuggingFace trainer,
    as it does not support random sampling of the evaluation dataset, but rather
    always iterates over the whole dataset.
    """

    def __init__(self, dataset: Dataset,
                 num_samples: Union[int, float],
                 allow_less: bool = True,
                 unique: bool = False,
                 seed: Optional[int] = None):
        """
        :param dataset: The dataset to wrap
        :param num_samples: Simulate a dataset of this size, or a fraction of the original dataset size
        :param allow_less: If True, the num_samples will be reduced to the actual length of the dataset if it is smaller
        :param unique: If True, each batch of num_samples items will contain only unique elements
        :param seed: The random seed to use (optional)
        """
        self.dataset = dataset
        if isinstance(num_samples, float) and num_samples > 1.0:
            logger.warning(f"num_samples is a float {num_samples} > 1.0, this is probably a mistake. Converting to int")
            num_samples = int(num_samples)

        if isinstance(num_samples, float):
            assert 0.0 < num_samples <= 1.0, f"Invalid fraction {num_samples}"
            self.num_samples = max(1, int(num_samples * len(dataset)))
        else:
            assert num_samples >= 0, f"Invalid number of samples {num_samples}"
            self.num_samples = num_samples
        self.rng = random.Random(seed)
        self.actual_len = len(dataset)
        if self.actual_len < self.num_samples:
            if allow_less:
                self.num_samples = self.actual_len
            else:
                raise ValueError("The dataset is too small to sample {} elements".format(self.num_samples))
        assert self.num_samples > 0 and self.actual_len >= self.num_samples
        self._unique = unique
        if self._unique:
            self._resampleIndices()

    def _resampleIndices(self):
        self._indices = self.rng.sample(range(self.actual_len), self.num_samples)
        assert len(self._indices) == self.num_samples
        self._counter = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self._unique:
            if self._counter >= self.num_samples:
                self._resampleIndices()
            idx = self._indices[self._counter]
            self._counter += 1
            return self.dataset[idx]
        else:
            return self.dataset[self.rng.randint(0, self.actual_len - 1)]
