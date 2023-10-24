import concurrent
import ipaddress
from pathlib import Path
import shutil
from typing import (Callable, Dict, Iterator, List, Optional, Sequence, Tuple,
                    Union)
import uuid

import datasets as ds
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import torch
import transformers
import tqdm
import xxhash

# Supported types for a network CIDR.
CIDR = Union[ipaddress.IPv4Address, ipaddress.IPv4Network, str]

# Supported type for a tokenizer.
Tokenizer = transformers.PreTrainedTokenizerBase

# Supported types for a model.
Model = Union[transformers.PreTrainedModel, torch.nn.DataParallel]

# Type for transform functions passed to InternetDatset.transform.
TransformFunction = Callable[[Model, transformers.BatchEncoding],
                             Union[npt.NDArray, Dict[str, npt.NDArray]]]


class InternetDataset:
    """Class for querying/manipulating Internet measurement data.

    Wraps a Hugging Face dataset containing Internet measurement data, with each
    row corresponding to data extracted from an IP address and port pair. The
    associated Hugging Face dataset must contain the following columns:

    - 'id': Contains bytes objects starting with a single byte (\x00 for IPv4
      and \x01 for IPv6 addresses), followed by the byte representation of the
      associated IP address and port. Note that currently datasets containing
      only IPv4 addresses are supported.
    - 'service_name': The associated service (network protcol) name for the
      entry, e.g., 'HTTP', 'Telnet', etc.

    Attributes:
        num_rows: Number of rows in the dataset.
        num_columns: Number of columns in the dataset.
        num_services: Number of services (length of `service_names`).
        column_names: Names of the columns in the dataset.
        hash: Hash of the dataset.
        ids: 1-D array of uint64 containing the associated IDs.
        ips: 1-D array of uint32 containing the associated IP addresses.
        ports: 1-D array of uint16 containing the associated ports.
        service_names: 1-D array of strings, with `service_names[services[i]]`
            corresponding to the service name for the `i`th row in the dataset.
        services: 1-D array of int32 containing the associated services.
    """

    def __init__(self, dataset: ds.Dataset, _cache: Optional[dict] = None):
        """Creates a new dataset.

        Args:
            dataset: The associated Huggingface dataset.
        """
        self.dataset = dataset
        self._cache = _cache or {}

    @property
    def _cache_dir(self) -> Optional[str]:
        return self._cache.get('cache_dir')

    @property
    def num_rows(self) -> int:
        return self.dataset.num_rows

    @property
    def num_columns(self) -> int:
        return self.dataset.num_columns

    @property
    def num_services(self) -> int:
        return self.service_names.shape[0]

    @property
    def column_names(self) -> List[str]:
        return self.dataset.column_names

    @property
    def hash(self) -> int:
        return hash(self.dataset._fingerprint)

    @property
    def ids(self) -> npt.NDArray[np.uint64]:
        if 'ids' not in self._cache:
            self._cache_ids()
        return self._cache['ids']

    @property
    def ips(self) -> npt.NDArray[np.uint32]:
        return np.right_shift(self.ids, 16).astype(np.uint32)

    @property
    def ports(self) -> npt.NDArray[np.uint16]:
        return (self.ids & 0xFFFF).astype(np.uint16)

    @property
    def service_names(self) -> npt.NDArray:
        if 'service_names' not in self._cache:
            self._cache_services()
        return self._cache['service_names']

    @property
    def services(self) -> npt.NDArray[np.int32]:
        if 'services' not in self._cache:
            self._cache_services()
        return self._cache['services']

    def __getitem__(self, key):
        return self.dataset[key]

    def __iter__(self) -> Iterator:
        return iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def _clear_cache(self, delete_files: bool = False):
        # Clears the cache, and (optionally) deletes all cache files.
        if delete_files and self._cache_dir is not None:
            shutil.rmtree(self._cache_dir, ignore_errors=True)

        self._cache.clear()

    def _get_subset_cache(self):
        # Returns cache for initializing a dataset that is a subset of this one.
        return {
            k: v
            for k, v in self._cache.items()
            if k not in ['ids', 'services']
        }

    def _get_with_indices(self, data: npt.NDArray) -> npt.NDArray:
        # If the Hugging Face dataset contains indices, returns the
        # corresponding data elements, otherwise returns the data unchanged.
        assert data.shape[0] == self.dataset.data.num_rows
        if self.dataset._indices is None:
            return data

        indices = self.dataset._indices.column('indices').to_numpy()
        return data[indices]

    def _store_ids(self, ids: npt.NDArray[np.uint64], batch_size: int = 1000):
        # Stores the IDs in the provided array.
        column = 'id'
        dataset = self.dataset.with_format(columns=column)
        vectorizer = np.vectorize(lambda x: int.from_bytes(x, 'big'))
        num_batches = int(np.ceil(self.dataset.data.num_rows / batch_size))
        iterator = tqdm.trange(num_batches,
                               desc='Extracting IDs from dataset',
                               unit='batch')
        for i in iterator:
            start = i * batch_size
            end = (i + 1) * batch_size
            ids[start:end] = vectorizer(dataset[start:end][column])

    def _cache_ids(self):
        # Caches the IDs in either memory or a memory mapped file.
        if 'ids' in self._cache:
            return
        if 'ids_orig' in self._cache:
            self._cache['ids'] = self._get_with_indices(self._cache['ids_orig'])
            return
        if self._cache_dir is None:
            num_rows = self.dataset.data.num_rows
            ids = np.zeros([num_rows], dtype=np.uint64)
            self._store_ids(ids)
            ids.flags.writeable = False
        else:
            cache_dir = self._cache_dir
            ids_file = cache_dir / 'ids.data'
            num_rows = self.dataset.data.num_rows
            if not ids_file.is_file():
                ids = np.memmap(ids_file,
                                dtype=np.uint64,
                                mode='w+',
                                shape=(num_rows,))
                self._store_ids(ids)

            ids = np.memmap(ids_file, dtype=np.uint64, mode='r')
            assert ids.shape == (num_rows,)

        self._cache['ids_orig'] = ids
        self._cache['ids'] = self._get_with_indices(ids)

    def _store_services(self,
                        services: npt.NDArray[np.int32],
                        batch_size: int = 1000) -> npt.NDArray:
        # Stores the services in the provided array.
        column = 'service_name'
        dataset = ds.Dataset(self.dataset.data, fingerprint=str(uuid.uuid4()))
        dataset = dataset.with_format(columns=column)
        encoder = sklearn.preprocessing.LabelEncoder()
        encoder.fit(dataset.unique(column))
        service_names = encoder.classes_
        num_batches = int(np.ceil(dataset.num_rows / batch_size))
        iterator = tqdm.trange(num_batches,
                               desc='Extracting services from dataset',
                               unit='batch')
        for i in iterator:
            start = i * batch_size
            end = (i + 1) * batch_size
            services[start:end] = encoder.transform(dataset[start:end][column])

        return service_names

    def _cache_services(self):
        # Caches the IDs in either memory or a memory mapped file.
        if 'services' in self._cache:
            return
        if 'services_orig' in self._cache:
            services = self._cache['services_orig']
            self._cache['services'] = self._get_with_indices(services)
            return
        if self._cache_dir is None:
            num_rows = self.dataset.data.num_rows
            services = np.zeros([num_rows], dtype=np.int32)
            service_names = self._store_services(services)
            services.flags.writeable = False
        else:
            cache_dir = self._cache_dir
            services_file = cache_dir / 'services.data'
            service_names_file = cache_dir / 'service_names.npy'
            num_rows = self.dataset.data.num_rows
            if not services_file.is_file():
                services = np.memmap(services_file,
                                     dtype=np.int32,
                                     mode='w+',
                                     shape=(num_rows,))
                service_names = self._store_services(services)
                np.save(service_names_file, service_names)

            services = np.memmap(services_file, dtype=np.int32, mode='r')
            service_names = np.load(service_names_file)
            assert services.shape == (num_rows,)

        self._cache['services_orig'] = services
        self._cache['services'] = self._get_with_indices(services)
        self._cache['service_names'] = service_names

    def save_to_disk(self, *args, **kwargs):
        """Wraps ds.Dataset.save_to_disk."""
        self.dataset.save_to_disk(*args, **kwargs)

    @staticmethod
    def concatenate_datasets(datasets: Sequence['InternetDataset'], *args,
                             **kwargs) -> 'InternetDataset':
        """Wraps ds.concatenate_datasets."""
        datasets = [d.dataset for d in datasets]
        return InternetDataset(
            ds.concatenate_datasets(datasets, *args, **kwargs))

    @staticmethod
    def load_from_disk(dataset_path: str, *args, **kwargs) -> 'InternetDataset':
        """Wraps ds.Dataset.load_from_disk."""
        return InternetDataset(ds.load_from_disk(dataset_path, *args, **kwargs),
                               _cache={'cache_dir': Path(dataset_path)})

    def set_format(self, *args, **kwargs):
        """Wraps ds.Dataset.set_format."""
        self.dataset.set_format(*args, **kwargs)

    def reset_format(self):
        """Wraps ds.Dataset.reset_format."""
        self.dataset.reset_format()

    def with_format(self, *args, **kwargs):
        """Wraps ds.Dataset.with_format."""
        return InternetDataset(self.dataset.with_format(*args, **kwargs),
                               self._cache)

    def with_transform(self, *args, **kwargs):
        """Wraps ds.Dataset.with_transform."""
        return InternetDataset(self.dataset.with_transform(*args, **kwargs),
                               self._cache)

    def map(self, *args, **kwargs):
        """Wraps ds.Dataset.map."""
        return InternetDataset(self.dataset.map(*args, **kwargs))

    def flatten_indices(self, *args, **kwargs):
        """Wraps ds.Dataset.flatten_indices."""
        return InternetDataset(self.dataset.flatten_indices(*args, **kwargs))

    def select(self, *args, **kwargs):
        """Wraps ds.Dataset.select."""
        return InternetDataset(self.dataset.select(*args, **kwargs),
                               self._get_subset_cache())

    def shuffle(self, *args, **kwargs):
        """Wraps ds.Dataset.shuffle."""
        return InternetDataset(self.dataset.shuffle(*args, **kwargs),
                               self._get_subset_cache())

    def add_column(self, *args, **kwargs):
        """Wraps ds.Dataset.add_column."""
        return InternetDataset(self.dataset.add_column(*args, **kwargs),
                               self._cache)

    def encode_service(self,
                       service_names: Union[str, npt.ArrayLike],
                       batch_size: int = 1000) -> npt.NDArray[np.int32]:
        """Converts service names to integer values.

        Converts a service_name to its corresponding index in
        `self.service_names`.

        Args:
            service_names: Array-like containing service name(s) to convert.

        Returns:
            services: The corresponding integer values, with the same shape as
                `service_names`. A value of -1 means that the corresponding
                element was not found in `self.service_names`.
        """
        service_names = np.asarray(service_names)
        names = service_names.reshape([-1, 1])
        services = np.full([service_names.size], -1, dtype=np.int32)
        num_batches = int(np.ceil(services.shape[0] / batch_size))
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            rows, cols = (names[start:end] == self.service_names).nonzero()
            services[start + rows] = cols

        return services.reshape(service_names.shape)

    def filter_id(self,
                  ids: npt.ArrayLike,
                  services: Optional[npt.ArrayLike] = None,
                  return_mask: bool = False,
                  **kwargs) -> Optional['InternetDataset']:
        """Filters for rows whose ID is in the provided list.

        Note that to use this function, the dataset must be sorted according to
        the IDs.

        Args:
            ids: 1-D array-like containing IDs to search for.
            services: 1-D array-like with the same shape as `ids`. If provided,
                only returns rows that also match the specified service. Note
                that elements can be integer indices (corresponding to elements
                in `self.service_names`) or strings.
            return_mask: If True, also returns a mask that specifies which
                elements in `ids` were matched to a dataset row.
            **kwargs: Keyword arguments to forward to ds.Dataset.select.

        Returns:
            dataset: Dataset containing rows with the same IDs as `ids[mask]`,
                and the same services as `service[mask]` (if `services` is not
                None. Returns None if the resulting dataset is empty.
            mask (if `return_mask` is True): 1-D boolean array with the same
                shape as `ids`, with True specifying elements for which a match
                was found in the dataset.

        Raises:
            ValueError: If `ids` is not a 1-D array, or if `services` is not
                None and its shape is not the same as `ids`.
            TypeError: If `ids` does not have an integer type, or if `services`
                is not None and does not have an integer or string type.
        """
        ids = np.asarray(ids, dtype=None if len(ids) else np.uint64)
        if ids.ndim != 1:
            raise ValueError(f'ids.ndim ({ids.ndim}) != 1')
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f'ids.type ({ids.dtype}) is not an integer type')
        if services is not None:
            dtype = None if len(services) else np.uint16
            services = np.asarray(services, dtype=dtype)
            if services.shape != ids.shape:
                raise ValueError(f'ids.shape {ids.shape} != '
                                 f'services.shape {services.shape}')
            if (not np.issubdtype(services.dtype, np.integer) and
                    not np.issubdtype(service.dtype, np.str_)):
                raise TypeError(f'services.type ({ids.dtype}) is not an '
                                'integer or string type')

        indices = np.searchsorted(self.ids, ids)
        mask = self.ids[np.minimum(indices, self.num_rows - 1)] == ids
        indices = indices[mask]
        if services is not None:
            if np.issubdtype(services.dtype, np.integer):
                m = self.services[indices] == services[mask]
            else:
                m = self.service_names[self.services[indices]] == services[mask]

            indices = indices[m]
            mask[mask] = m

        dataset = self.select(indices, **kwargs) if indices.shape[0] else None
        return (dataset, mask) if return_mask else dataset

    def filter_network(self, network: Union[CIDR, Sequence[CIDR]],
                       **kwargs) -> Optional['InternetDataset']:
        """Filters for rows that belong to the specified network.

        Note that to use this function, the dataset must be sorted according to
        the IDs.

        Args:
            network: Network CIDR(s) to filter for.
            **kwargs: Keyword arguments to forward to ds.Dataset.select.

        Returns:
            Dataset containing rows from the specified network, or None if the
                resulting dataset is empty.
        """

        def get_bounds(cidr):
            # Returns IDs corresponding to the beginning/end of the CIDR for
            # querying the dataset.
            begin = int(cidr.network_address) << 16
            try:
                end = cidr.network_address + cidr.num_addresses
            except ipaddress.AddressValueError:
                end = 1 << 48
            else:
                end = int(end) << 16

            return begin, end

        if isinstance(network, CIDR.__args__):
            network = [ipaddress.ip_network(network)]
        else:
            network = [ipaddress.ip_network(cidr) for cidr in network]
            if not network:
                return None

        ids = self.ids
        begins, ends = zip(*[get_bounds(cidr) for cidr in network])
        begins = np.searchsorted(ids, begins)
        ends = np.searchsorted(ids, ends)
        indices = np.concatenate(
            [np.arange(begin, end) for begin, end in zip(begins, ends)])
        return self.select(indices, **kwargs) if indices.shape[0] else None

    def filter_service(self,
                       service_names: Union[str, Sequence[str]],
                       invert: bool = False,
                       **kwargs) -> Optional['InternetDataset']:
        """Filters for rows whose associated service is in the provided list.

        Args:
            service_names: Service name(s) to filter for.
            invert: Whether to filter for rows whose associated service is *not*
                in `service_names`.
            **kwargs: Keyword arguments to forward to ds.Dataset.select.

        Returns:
            Dataset containing rows corresponding to `service_names`, or None if
                the resulting dataset is empty.
        """
        if isinstance(service_names, str):
            service_names = [service_names]

        service_indices = np.isin(self.service_names,
                                  service_names).nonzero()[0]
        if np.all(service_indices == np.arange(self.service_names.shape[0])):
            return self

        indices = np.isin(self.services, service_indices,
                          invert=invert).nonzero()[0]
        return self.select(indices, **kwargs) if indices.shape[0] else None

    def hash_column(self, column: str) -> npt.NDArray:
        """Computes and returns the hash of values in a column.

        Args:
            column: Name of the dataset column to hash.

        Returns:
            hashes: Hashes of values in the specified column.
        """
        hashes = self.map(
            lambda batch: {'hash': [xxhash.xxh128_digest(x) for x in batch]},
            input_columns=[column],
            remove_columns=[column],
            batched=True,
            keep_in_memory=True)
        return hashes.with_format(columns='hash', type='numpy')['hash']

    def intersect(self,
                  other: "InternetDataset",
                  match_service: bool = True,
                  return_indices: bool = False,
                  **kwargs) -> Optional[Tuple]:
        """Returns the intersection of this dataset with the provided one.

        Computes the intersection of two datasets by finding rows corresponding
        to the same IDs and (optionally) services.

        Args:
            other: Dataset to intersect with this one.
            match_service: Whether to match services in addition to IDs.
            return_indices: Whether to return the indices of the intersections'
                rows in the original datasets.
            **kwargs: Keyword arguments to forward to ds.Dataset.select.

        Returns:
            Subsets of `self` and `other` corresponding to the intersection, or
            None if the intersection is empty. Note that the returned datasets
            are sorted according to their IDs. If `return_indices` is True, also
            returns two 1-D integer arrays containing the indices of rows in
            `self` and `other` that correspond to the returned subsets.
        """
        _, self_indices, other_indices = np.intersect1d(self.ids,
                                                        other.ids,
                                                        return_indices=True)
        if match_service:
            if np.array_equal(self.service_names, other.service_names):
                mask = (self.services[self_indices] ==
                        other.services[other_indices])
            else:
                other_services = self.encode_service(
                    other.service_names[other.services[other_indices]])
                mask = self.services[self_indices] == other_services

            self_indices = self_indices[mask]
            other_indices = other_indices[mask]
        if not self_indices.shape[0]:
            return None

        outputs = (self.select(self_indices,
                               **kwargs), other.select(other_indices, **kwargs))
        if return_indices:
            outputs += (self_indices, other_indices)

        return outputs

    def train_test_split(self,
                         *datasets: "InternetDataset",
                         test_size: Union[int, float],
                         max_test_size: Optional[int] = None,
                         shuffle: bool = True,
                         seed: Optional[int] = None,
                         **kwargs) -> List["InternetDataset"]:
        """Splits datasets into train and test subsets.

        Args:
            *datasets: Datasets to split.
            test_size: The number of test samples, or their proportion with
                respect to the total number of samples.
            max_test_size: The maximum number of test samples (if `test_size` is
                a float), ignored if `test_size` is an int.
            shuffle: Whether to shuffle before splitting.
            seed: Random seed for shuffling.
            **kwargs: Keyword arguments to forward to ds.Dataset.select.

        Returns:
            List containing train/test splits of `datasets`.

        Raises:
            ValueError: If `datasets` do not have the same number of rows.
	"""
        datasets = (self,) + datasets
        num_rows = datasets[0].num_rows
        for i in range(1, len(datasets)):
            if datasets[0].num_rows != num_rows:
                raise ValueError(
                    f'datasets[{i}].num_rows ({datasets[i].num_rows}) != '
                    f'datasets[0].num_rows ({datasets[0].num_rows})')

        if isinstance(test_size, float):
            test_size = int(test_size * num_rows)
            if max_test_size is not None:
                test_size = min(test_size, max_test_size)
        else:
            test_size = min(test_size, num_rows)

        if shuffle:
            train_indices, test_indices = train_test_split(np.arange(num_rows),
                                                           test_size=test_size,
                                                           random_state=seed)
        else:
            train_indices = np.arange(num_rows - test_size)
            test_indices = np.arange(num_rows - test_size, num_rows)

        outputs = [[
            d.select(train_indices, **kwargs),
            d.select(test_indices, **kwargs)
        ] for d in datasets]
        return sum(outputs, [])

    def transform(
        self,
        column: str,
        tokenizer: Tokenizer,
        models: Union[Model, Dict[Union[None, str], Model]],
        transform_fn: Optional[TransformFunction] = None,
        batch_size: int = 32,
        padding: Union[bool, str] = True,
        max_length: int = None,
        keep_in_memory: bool = True
    ) -> Union[npt.NDArray, Dict[str, npt.NDArray]]:
        """Transforms strings in a column.

        Uses a (group) of model(s) to transform strings in a given column. If a
        group of models are provided, then each is assumed to correpond to a
        service name, with `models[None]` specifying the model to default to if
        a row's associated service does not match any of the other items in
        `models`.

        Args:
            column: Dataset column to transform.
            tokenizer: Tokenizer to use for preprocessing the data.
            models: Model(s) to use. If a dictionary is provided, each key must
                specify the associated service name that could be processed by
                the model, with `models[None]` specifying a default model that
                could be used if a row does not match any of the other keys.
            transform_fn: Function for transforming a batch of encoded strings.
            batch_size: Batch size for processing examples.
            max_length: The maximum sequence length for truncating examples.
            keep_in_memory: Whether to keep the resulting dataset in memory.

        Returns:
            Result of the transformation.
        """
        truncation = max_length is not None

        def tokenize(inputs):
            # Tokenizes a batch of inputs.
            return tokenizer(inputs.tolist(),
                             padding=padding,
                             truncation=truncation,
                             max_length=max_length,
                             return_tensors='pt')

        @torch.no_grad()
        def _transform(model, tokenize_future):
            # Runs tokenized inputs through the model.
            encodings = tokenize_future.result()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if transform_fn is None:
                    outputs = model(encodings)
                else:
                    outputs = transform_fn(model, encodings)

            if isinstance(outputs, dict):
                num_samples = next(iter(outputs.values())).shape[0]
                outputs = [{k: outputs[k][[i]]
                            for k in outputs}
                           for i in range(num_samples)]
            else:
                outputs = np.vsplit(outputs, outputs.shape[0])

            return outputs

        if not isinstance(models, dict):
            models = {None: models}

        # Find duplicates and sort unique values according to their lengths.
        inputs = np.asarray(self[column], dtype=np.object)
        hashes = np.array([xxhash.xxh128_digest(i) for i in inputs])
        lengths = np.array([len(i) for i in inputs])
        _, indices_outer, inverse_outer = np.unique(hashes,
                                                    return_index=True,
                                                    return_inverse=True)
        indices_inner = np.argsort(lengths[indices_outer])
        indices = indices_outer[indices_inner]
        inputs = inputs[indices]
        if list(models) != [None]:
            services = self.services[indices]

        tokenize_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        transform_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1)
        outputs = np.empty(inputs.shape, dtype=np.object)

        def process_indices(model, indices):
            # Transforms dataset[indices] using the provided model and stores
            # the results in outputs[indices].
            indices = np.asarray(indices)
            num_batches = int(np.ceil(indices.shape[0] / batch_size))
            last_job = None
            for i in tqdm.trange(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                i_batch = indices[start:end]
                future = tokenize_executor.submit(tokenize, inputs[i_batch])
                if last_job is not None:
                    outputs[last_job[0]] = last_job[1].result()
                    last_job = None

                future = transform_executor.submit(_transform, model, future)
                if i == num_batches - 1:
                    outputs[i_batch] = future.result()
                else:
                    last_job = (i_batch, future)

        # Extract substrings using service-specific/default models.
        for service_name, model in models.items():
            if service_name is None:
                continue

            mask = services == self.encode_service(service_name)
            indices = mask.nonzero()[0]
            if indices.shape[0]:
                process_indices(model, indices)

        indices = [i for i, output in enumerate(outputs) if output is None]
        if indices:
            process_indices(models[None], indices)

        # Reorder and return outputs.
        del inputs
        outputs = outputs[np.argsort(indices_inner)][inverse_outer]
        if isinstance(outputs[0], dict):
            outputs = {
                k:
                np.concatenate([outputs[i][k] for i in range(outputs.shape[0])])
                for k in outputs[0]
            }
        else:
            outputs = np.concatenate(outputs)

        return outputs
