from dataclasses import dataclass
import functools
import multiprocessing as mp
import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import sklearn.utils
import tqdm

from .match import SequenceMatch
from .singleton import Glob
from .span import Spanner


@dataclass
class FingerprintExtractor:
    """Class for extracting fingerprints from clustered examples.

    Attributes:
        num_samples: Number of samples to draw from a cluster.
        num_draws: Number of draws per cluster.
        min_length: Minimum length for individual (contiguous) matches from a
            pair of examples.
        spanner: Function for generating a mapping between span names and the
            associated (start, end) indices in an example. If specified,
            matching is only performed between spans with matching names.
        num_workers: Number of worker threads for `fit` and `transform`.
        seed: Random seed for drawing samples.
        fingerprints: Extracted fingerprints (trained by calling `fit`). If
            `spanner` is None, then this is a list of Regex patterns. Otherwise,
            it is a mapping between span names and the associated patterns.
    """

    num_samples: int = 5
    num_draws: int = 1
    min_length: int = 1
    spanner: Optional[Spanner] = None
    num_workers: int = 1
    seed: Optional[int] = None
    fingerprints: Optional[Union[List[str], Dict[str, List[str]]]] = None

    def _generate_fingerprint(
            self, examples: List[str]) -> Union[str, Dict[str, str]]:
        # Generates a fingerprint for the provided examples.
        fingerprint = examples[0]
        fingerprint_spans = None
        if self.spanner is not None:
            fingerprint_spans = self.spanner(fingerprint)

        for example in examples[1:]:
            spans = None
            if self.spanner is not None:
                spans = self.spanner(example)

            match = SequenceMatch.generate_lcs(fingerprint,
                                               example,
                                               a_spans=fingerprint_spans,
                                               b_spans=spans,
                                               min_length=self.min_length)
            fingerprint = match.pattern
            if isinstance(fingerprint, dict):
                pos = 0
                fingerprint_spans = {}
                for key, sig in fingerprint.items():
                    fingerprint_spans[key] = (pos, pos + len(sig))
                    pos += len(sig)

                fingerprint = sum(fingerprint.values(), [])

        if fingerprint_spans is not None:
            fingerprint = {
                key: fingerprint[span[0]:span[1]]
                for key, span in fingerprint_spans.items()
            }

        def to_regex(fingerprint):
            if isinstance(fingerprint, list):
                pattern = ''.join('.*' if char is Glob else re.escape(char)
                                  for char in fingerprint)
                if fingerprint[-1] is not Glob:
                    pattern += '$'

                return pattern

            return {key: to_regex(sig) for key, sig in fingerprint.items()}

        return to_regex(fingerprint)

    def _fit_one(self, examples: Sequence[str], labels: npt.NDArray,
                 column: int, label: int) -> Set[Union[str, Dict[str, str]]]:
        # Generates fingerprints for a single cluster.
        fingerprints = set()
        indices = (labels[:, column] == label).nonzero()[0]
        if indices.size <= 1:
            return fingerprints

        random_state = None
        for i in range(self.num_draws):
            if self.seed is not None:
                random_state = np.random.RandomState((self.seed, label, i))

            indices_sampled = sklearn.utils.resample(indices,
                                                     replace=False,
                                                     n_samples=min(
                                                         self.num_samples,
                                                         indices.size),
                                                     random_state=random_state)
            examples_sampled = [examples[i] for i in indices_sampled]
            fingerprint = self._generate_fingerprint(examples_sampled)

            if isinstance(fingerprint, str):
                fingerprints.add(fingerprint)
            else:
                for key, sig in fingerprint.items():
                    fingerprints.add((key, sig))

        return fingerprints

    def _fit_worker(self, examples: Sequence[str], labels: npt.NDArray,
                    input_queue: mp.Queue, output_queue: mp.Queue):
        # A single worker thread for the fit method.
        random_state = None
        while True:
            task = input_queue.get()
            if task is None:
                return

            output_queue.put(self._fit_one(examples, labels, **task))

    @staticmethod
    def _transform_one(examples: Sequence[Union[str, Dict[str, str]]],
                       fingerprint: str,
                       key: Optional[str] = None) -> npt.NDArray[np.bool_]:
        # Matches a single fingerprint against the provided examples.
        fingerprint = re.compile(fingerprint)
        if key is None:
            return np.array(
                [fingerprint.match(e) is not None for e in examples])

        return np.array([
            key in e and fingerprint.match(e[key]) is not None for e in examples
        ])

    def _transform_worker(self, examples: Sequence[Union[str, Dict[str, str]]],
                          input_queue: mp.Queue, output_queue: mp.Queue):
        # A single worker thread for the transform method.
        while True:
            task = input_queue.get()
            if task is None:
                return

            fingerprint_id = task.pop('fingerprint_id')
            output_queue.put(
                (fingerprint_id, self._transform_one(examples, **task)))

    def fit(self,
            examples: Sequence[str],
            labels: npt.ArrayLike,
            exclude_labels: Optional[Sequence] = None
           ) -> "FingerprintExtractor":
        """Extracts fingerprints from the provided clustered examples.

        Args:
            examples: Examples to extract fingerprints from.
            labels: Cluster labels associated with `examples`. Can be a 1-D or
                2-D array, with the latter specifying labels for multiple
                clusterings.
            exclude_labels: Labels to exclude from fingerprint generation.

        Returns:
            `self`

        Raises:
            ValueError: If `labels` is not a 1-D or 2-D array, or if the number
                of rows in `labels` does not match the number of examples.
        """
        labels = np.asarray(labels)
        if labels.ndim not in [1, 2]:
            raise ValueError('labels must be a 1-D or 2-D array (a '
                             f'{labels.ndim} array was given)')
        if labels.shape[0] != len(examples):
            raise ValueError(f'labels.shape[0] {labels.shape[0]} does not '
                             f'match the number of examples ({len(examples)})')

        labels = labels.reshape([len(examples), -1])
        tasks = []
        for column in range(labels.shape[1]):
            unique_labels = np.unique(labels[:, column])
            if exclude_labels is not None:
                exclude = np.isin(unique_labels, exclude_labels)
                unique_labels = np.delete(unique_labels, exclude)

            for label in unique_labels:
                tasks += [{'column': column, 'label': label}]

        fingerprints = set()
        with tqdm.tqdm(total=len(tasks)) as progress_bar:
            if self.num_workers > 1:
                input_queue = mp.Queue()
                output_queue = mp.Queue()
                workers = [
                    mp.Process(target=self._fit_worker,
                               args=(examples, labels, input_queue,
                                     output_queue))
                    for _ in range(self.num_workers)
                ]
                for worker in workers:
                    worker.start()

                for task in tasks + [None] * self.num_workers:
                    input_queue.put(task)
                for _ in range(len(tasks)):
                    for fingerprint in output_queue.get():
                        fingerprints.add(fingerprint)

                    progress_bar.update(1)

                for queue in [input_queue, output_queue]:
                    queue.close()
                    queue.join_thread()
                for worker in workers:
                    worker.join()
            else:
                for task in tasks:
                    for fingerprint in self._fit_one(examples, labels, **task):
                        fingerprints.add(fingerprint)

                    progress_bar.update(1)

        if self.spanner is None:
            self.fingerprints = [
                sig for sig in sorted(fingerprints) if sig != '.*'
            ]
        else:
            fingerprints_per_key = {}
            for key, fingerprint in fingerprints:
                if fingerprint != '.*':
                    if key in fingerprints_per_key:
                        fingerprints_per_key[key] += [fingerprint]
                    else:
                        fingerprints_per_key[key] = [fingerprint]

            self.fingerprints = {
                key: sorted(sig) for key, sig in fingerprints_per_key.items()
            }

        return self

    def transform(
            self,
            examples: Sequence[str]) -> List[List[Union[int, Tuple[str, int]]]]:
        """Matches the extracted fingerprints against the provided examples.

        Args:
            examples: Examples to match against.

        Returns:
            Matching fingerprints for the provided examples. If `self.spanner`
                is None, then each item is a list of integers corresponding to
                matching fingerprints in `self.fingerprints` for the associated
                example. Otherwise, each item is a list of (span_name, index)
                pairs, with `self.fingerprints[span_name][index]` specifying the
                associated matching fingerprint.

        Raises:
            ValueError: If fingerprints have not yet been trained (by calling
                the `fit` method).
        """
        if self.fingerprints is None:
            raise ValueError('Fingerprints have not yet been trained (call '
                             'fit() before transform() to train fingerprints)')

        if isinstance(self.fingerprints, list):
            tasks = [{
                'fingerprint_id': i,
                'fingerprint': sig
            } for i, sig in enumerate(self.fingerprints)]
        else:
            tasks = [{
                'fingerprint_id': (key, i),
                'fingerprint': s,
                'key': key
            }
                     for key, sig in self.fingerprints.items()
                     for i, s in enumerate(sig)]

        matches = [[] for _ in range(len(examples))]
        with tqdm.tqdm(total=len(tasks)) as progress_bar:
            if self.num_workers > 1:
                if self.spanner is not None:
                    with mp.Pool(self.num_workers) as pool:
                        examples = pool.map(self.spanner.split, examples)

                input_queue = mp.Queue()
                output_queue = mp.Queue()
                workers = [
                    mp.Process(target=self._transform_worker,
                               args=(examples, input_queue, output_queue))
                    for _ in range(self.num_workers)
                ]
                for worker in workers:
                    worker.start()

                for task in tasks + [None] * self.num_workers:
                    input_queue.put(task)
                for _ in range(len(tasks)):
                    fingerprint_id, is_match = output_queue.get()
                    for i in is_match.nonzero()[0]:
                        matches[i] += [fingerprint_id]

                    progress_bar.update(1)

                for queue in [input_queue, output_queue]:
                    queue.close()
                    queue.join_thread()
                for worker in workers:
                    worker.join()
            else:
                if self.spanner is not None:
                    examples = [self.spanner.split(e) for e in examples]

                for task in tasks:
                    fingerprint_id = task.pop('fingerprint_id')
                    for i in self._transform_one(examples, **task).nonzero()[0]:
                        matches[i] += [fingerprint_id]

                    progress_bar.update(1)

        return matches
