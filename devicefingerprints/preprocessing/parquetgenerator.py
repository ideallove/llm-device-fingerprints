import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from ..typing import PathLike
from ..utility import fopen
from .parser import Parser


class ParquetGenerator:
    """Class for parsing a raw dataset and converting it to a Parquet file."""

    def __init__(self,
                 data_files: Union[PathLike, Sequence[PathLike]],
                 parser: Parser,
                 filter_func: Optional[Callable[[dict], bool]] = None,
                 sort: bool = False,
                 sort_column: Optional[str] = None,
                 shuffle: bool = False,
                 seed: Optional[int] = None,
                 num_workers: int = 1,
                 chunk_size: int = 1000,
                 batch_size: int = 1000):
        """Creates a Parquet generator.

        Args:
            data_files: Path to data file(s). Each line in a data file is
                assumed to contain a single example.
            parser: Parser for parsing lines/examples from data files.
            filter_func: If provided, examples for which this function returns
                False are excluded from the output.
            sort: Whether to sort examples before writing to the output.
            sort_column: Column to use for sorting examples. Defaults to
                ``parser.sort_column`` if not specified.
            shuffle: Whether to shuffle examples before writing to the output.
            seed: Random seed for shuffling examples.
            num_workers: Number of parallel workers for parsing examples.
            chunk_size: Chunk size for workers.
            batch_size: Batch size for writing to the output.

        Raises:
            ValueError: If both `sort` and `shuffle` are True, or if `sort` is
                True and a sort column is not specified (either directly or
                through `parser`).
        """
        if sort and shuffle:
            raise ValueError('Only one of sort and shuffle can be True')
        if sort and sort_column is None:
            sort_column = parser.sort_column
            if sort_column is None:
                raise ValueError('Was asked to sort examples, but a sort '
                                 'column was not specified and could not '
                                 'infer one from the provided parser')
        if isinstance(data_files, PathLike.__args__):
            data_files = [data_files]

        self.data_files = list(map(Path, data_files))
        self.parser = parser
        self.filter_func = filter_func
        self.sort = sort
        self.sort_column = sort_column
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.batch_size = batch_size

    def _parse_file(
        self,
        data_file: Path,
        writer: pq.ParquetWriter,
        rng: np.random.Generator,
        last_batch: Optional[List[dict]] = None,
        write_last_batch: bool = True
    ) -> Union[List[dict], Tuple[int, Optional[npt.NDArray]]]:
        # Parses a single data file and writes the results to a Parquet file.
        schema = self.parser.schema

        def make_table(batch: List[dict]) -> pa.Table:
            # Converts a batch of examples to an Arrow table.
            batch_ = {}
            for example in batch:
                for key, value in example.items():
                    if key in batch_:
                        batch_[key].append(value)
                    else:
                        batch_[key] = [value]

            return pa.Table.from_pydict(batch_, schema=schema)

        # Parse data file.
        examples = last_batch or []
        sort_keys = []
        eager = not self.sort and not self.shuffle
        with mp.Pool(self.num_workers) as pool:
            with fopen(data_file) as f:
                for example in pool.imap(self.parser.parse,
                                         f,
                                         chunksize=self.chunk_size):
                    if self.filter_func is None or self.filter_func(example):
                        examples.append(example)
                        if self.sort:
                            sort_keys.append(example[self.sort_column])

                    # If writing eagerly (i.e., neither sorting nor shuffling),
                    # write complete batches to the Parquet file.
                    if eager and len(examples) == self.batch_size:
                        writer.write_table(make_table(examples))
                        examples = []

        # If writing eagerly, write or return remaining examples.
        if eager:
            if examples and write_last_batch:
                writer.write_table(make_table(examples))
                examples = []

            return examples

        # Sort/shuffle examples.
        if self.sort:
            sort_keys = np.array(sort_keys)
            indices = np.argsort(sort_keys)
            examples = [examples[i] for i in indices]
            sort_keys = sort_keys[indices]
        elif self.shuffle:
            indices = rng.permutation(np.arange(len(examples)))
            examples = [examples[i] for i in indices]

        # Write examples to the Parquet file.
        for i in range(0, len(examples), self.batch_size):
            writer.write_table(make_table(examples[i:i + self.batch_size]))

        return len(examples), sort_keys if self.sort else None

    def _skip_file(self,
                   parquet_file: Path) -> Tuple[int, Optional[npt.NDArray]]:
        table = pq.read_table(parquet_file)
        sort_keys = table[self.sort_column].to_numpy() if self.sort else None
        return len(table), sort_keys

    def _generate(self, data_files: List[Path], parquet_file: Path,
                  rng: np.random.Generator):
        # Generates a Parquet file when only a single data file is provided, or
        # when neither sorting nor shuffling.
        last_batch = []
        with pq.ParquetWriter(parquet_file, self.parser.schema) as writer:
            for data_file in tqdm.tqdm(data_files):
                if data_file == data_files[-1]:
                    self._parse_file(data_file,
                                     writer,
                                     rng,
                                     last_batch=last_batch)
                else:
                    last_batch = self._parse_file(data_file,
                                                  writer,
                                                  rng,
                                                  last_batch=last_batch,
                                                  write_last_batch=False)

    def _generate_external_sort(self, data_files: List[Path],
                                parquet_file: Path, rng: np.random.Generator):
        # Generates a Parquet file when multiple data files are provided, and
        # either sorting or shuffling.

        # Set up a temp directory.
        temp_dir = parquet_file.parent / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Parse data files and write the results to temporary Parquet files.
        schema = self.parser.schema
        temp_files = []
        file_lengths = []
        sort_keys = []
        for data_file in tqdm.tqdm(data_files):
            temp_file = temp_dir / f"{data_file.name.split('.')[0]}.parquet"
            temp_files.append(temp_file)

            if temp_file.is_file():
                length, keys = self._skip_file(temp_file)
                file_lengths.append(length)
                if self.sort:
                    sort_keys.append(keys)
            else:
                with pq.ParquetWriter(f'{temp_file}.temp', schema) as writer:
                    length, keys = self._parse_file(data_file, writer, rng)
                    file_lengths.append(length)
                    if self.sort:
                        sort_keys.append(keys)

                shutil.move(f'{temp_file}.temp', temp_file)

        # Produce indices for merging the Parquet files.
        if self.sort:
            sort_keys = np.concatenate(sort_keys)
            indices = np.argsort(sort_keys)
            file_indices = np.searchsorted(np.cumsum(file_lengths),
                                           indices,
                                           side='right')
        else:
            file_indices = [
                i * np.ones(file_length, dtype=np.int64)
                for i, file_length in enumerate(file_lengths)
            ]
            file_indices = np.concatenate(file_indices)
            if self.shuffle:
                file_indices = rng.permutation(file_indices)

        # Merge Parquet files.
        readers = [pq.ParquetFile(temp_file) for temp_file in temp_files]
        readers = [r.iter_batches(batch_size=self.batch_size) for r in readers]
        batches = [None] * len(readers)
        indices = [0] * len(readers)
        batch = []
        with pq.ParquetWriter(parquet_file, schema) as writer:
            for i in tqdm.tqdm(file_indices):
                if batches[i] is None or indices[i] == batches[i].num_rows:
                    batches[i] = next(readers[i])
                    indices[i] = 0

                batch.append(batches[i].slice(indices[i], 1))
                indices[i] += 1
                if len(batch) == self.batch_size:
                    table = pa.Table.from_batches(batch, schema=schema)
                    writer.write_table(table)
                    batch = []

            if batch:
                table = pa.Table.from_batches(batch, schema=schema)
                writer.write_table(table)

        # Remove temp directory.
        shutil.rmtree(temp_dir)

    def generate(self, parquet_file: PathLike):
        """Parses data files and writes the results a Parquet file.

        Args:
            parquet_file: Path to the output Parquet file.
        """
        rng = None
        if self.shuffle:
            rng = np.random.default_rng(self.seed)

        parquet_file = Path(parquet_file)
        if len(self.data_files) == 1 or (not self.sort and not self.shuffle):
            self._generate(self.data_files, parquet_file, rng)
        else:
            self._generate_external_sort(self.data_files, parquet_file, rng)
