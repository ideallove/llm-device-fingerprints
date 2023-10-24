from abc import abstractmethod
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import datasets as ds
import numpy as np
import sklearn.utils
import torch

from ..internetdataset import InternetDataset

# Type of datasets wrapped by BaseDataset.
RawDataset = Union[ds.Dataset, InternetDataset]


class BaseDataset(torch.utils.data.IterableDataset):
    """Wraps and generates training/evaluation data from Hugging Face datasets.

    Note that any subclass must implement the _iter method, which should yield
    output data one sample at a time. You can use `self.data` inside the _iter
    method to access the (shuffled/sorted) datasets provided to the constructor.
    """

    def __init__(self,
                 data: Union[RawDataset, Sequence[RawDataset]],
                 shuffle: bool = False,
                 seed: Optional[int] = None,
                 sort: bool = False,
                 sort_column: Optional[str] = None,
                 batch_size: int = 1000):
        """Creates a dataset.

        Args:
            data: Associated dataset(s). Must all have the same number of rows.
            shuffle: Whether to shuffle the dataset(s) (in a consistent manner).
            seed: Random seed for shuffling.
            sort: Whether to sort the dataset(s) according to the length of
                examples in `sort_column`. Can be used to speed up evaluation.
                Note that when `data` is a sequence, the order obtained from the
                first dataset is used to sort all datasets.
            sort_column: Column used for sorting the dataset(s).
            batch_size: Batch size for processing examples.

        Raises:
            ValueError: If `data` is a sequence and the number of rows in the
                enclosed datasets are not the same.
            ValueError: If both `sort` and `shuffle` are True, or if `sort` is
                True and `sort_column` is not specified.
        """
        is_sequence = not isinstance(data, RawDataset.__args__)
        if is_sequence:
            for i, dataset in enumerate(data[1:]):
                if dataset.num_rows != data[0].num_rows:
                    raise ValueError(
                        f'data[{i}].num_rows ({data[i].num_rows}) != '
                        f'data[0].num_rows ({data[0].num_rows})')
        if sort and shuffle:
            raise ValueError('Only one of sort and shuffle can be True')
        if sort:
            if sort_column is None:
                raise ValueError('sort_column is not specified')

            dataset = data[0] if is_sequence else data
            dataset = dataset.with_transform(
                lambda b: np.array([len(x) for x in b[sort_column]]),
                columns=sort_column)
            num_batches = int(np.ceil(dataset.num_rows / batch_size))
            lengths = [
                dataset[i * batch_size:(i + 1) * batch_size]
                for i in range(num_batches)
            ]
            indices = np.argsort(np.hstack(lengths))
            if is_sequence:
                data = type(data)([d.select(indices) for d in data])
            else:
                data = data.select(indices)

        self.data = data
        self.shuffle = shuffle
        self.sort = sort
        self.seed = seed
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.shuffle:
            if isinstance(self.data, RawDataset.__args__):
                self.data = self.data.shuffle(seed=self.seed)
            else:
                indices = np.arange(self.data[0].num_rows)
                indices = sklearn.utils.shuffle(indices, random_state=self.seed)
                self.data = type(
                    self.data)([d.select(indices) for d in self.data])

        yield from self._iter()

    @abstractmethod
    def _iter(self) -> Iterator[Dict[str, Any]]:
        ...
