import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import tokenizers
import torch
import transformers

from ..match import SequenceMatch
from ..span import Spanner, Spans
from .basedataset import BaseDataset, RawDataset


class LcsExtractionDataset(BaseDataset):
    """Dataset for extracting common text using LCS matches.

    Uses longest common subsequence matching to extract common subsequences from
    a pair of examples. The subsequences are then converted to labels to be fed
    to a token classification model for text extraction.
    """

    def __init__(self,
                 data: Tuple[RawDataset, RawDataset],
                 column: str,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 max_length: Optional[int] = None,
                 spanner: Optional[Spanner] = None,
                 min_match_length: int = 1,
                 min_match_total_length: int = 1,
                 shuffle: bool = False,
                 seed: Optional[int] = None,
                 sort: bool = False,
                 batch_size: int = 1024):
        """Creates a dataset.

        Args:
            data: Pair of datasets for generating LCS matches, with `data[0][i]`
                and `data[1][i]` corresponding to a pair of examples.
            column: Column containing the associated examples.
            tokenizer: Tokenizer for encoding the data.
            max_length: If provided, tokenized sequences are truncated to this
                maximum length.
            spanner: Function for generating a mapping between span names and
                the associated (start, end) indices in an example. If specified,
                matching is only performed between spans with matching names.
            min_match_length: Minimum length (number of characters) for
                individual (continguous) matches.
            min_match_total_length: Minimum length (number of tokens) for an
                entire match (i.e., a sequence of contiguous matches).
            shuffle: Whether to shuffle the datasets (in a consistent manner).
            seed: Random seed for shuffling.
            sort: Whether to sort the datasets according to the length of
                examples in `column`. Can be used to speed up evaluation. Note
                that the order obtained from the first dataset is used to sort
                the second one.
            batch_size: Batch size for processing examples.

        Raises:
            ValueError: If `data[0].num_rows` is not the same as
                `data[1].num_rows`.
            ValueError: If both `sort` and `shuffle` are True.
        """
        if data[0].num_rows != data[1].num_rows:
            raise ValueError(f'data[0].num_rows ({data[0].num_rows}) != '
                             f'data[1].num_rows ({data[0].num_rows})')

        super().__init__(data,
                         shuffle=shuffle,
                         seed=seed,
                         sort=sort,
                         sort_column=column,
                         batch_size=batch_size)
        self.column = column
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.spanner = spanner
        self.min_match_length = min_match_length
        self.min_match_total_length = min_match_total_length

    def _get_spans(self, example: Dict[str, Any],
                   encoding: tokenizers.Encoding) -> Optional[Spans]:
        # Generates and converts spans from char indices to token indices.
        char_spans = self.spanner(example)
        if char_spans is None:
            return

        starts, ends = zip(*encoding.offsets[1:-1])  # Ignore BOS/EOS tokens.
        starts = np.asarray(starts)
        ends = np.asarray(ends)

        spans = {}
        for key, (start, end) in char_spans.items():
            start_token = np.searchsorted(ends, start, side='right')
            end_token = np.searchsorted(starts, end, side='right') - 1
            if start_token >= end_token:
                continue

            spans[key] = (start_token, end_token)

        return spans

    def _is_junk_sequence(self, match: List[int]) -> bool:
        # Checks whether a subsequence match satisfies the minimum length.
        match: str = self.tokenizer.decode(match, skip_special_tokens=True)
        return len(match) < self.min_match_length

    def _iter(self) -> Iterator[Dict[str, Any]]:
        # Get worker information for multiprocess data loading.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        tokenize = functools.partial(self.tokenizer,
                                     padding=False,
                                     truncation=self.max_length is not None,
                                     max_length=self.max_length)
        dataset_a = self.data[0].with_format(columns=[self.column])
        dataset_b = self.data[1].with_format(columns=[self.column])
        for start in range(worker_id * self.batch_size, dataset_a.num_rows,
                           num_workers * self.batch_size):
            end = min(start + self.batch_size, dataset_a.num_rows)
            encodings_a = tokenize(dataset_a[start:end][self.column])
            encodings_b = tokenize(dataset_b[start:end][self.column])

            # Generate LCS matches. The BOS/EOS tokens are ignored.
            kwargs = {}
            if self.min_match_length:
                kwargs['is_junk_sequence'] = self._is_junk_sequence

            matches = []
            for i in range(end - start):
                if self.spanner is not None:
                    kwargs['a_spans'] = self._get_spans(self.data[0][start + i],
                                                        encodings_a[i])
                    kwargs['b_spans'] = self._get_spans(self.data[1][start + i],
                                                        encodings_b[i])

                matches.append(
                    SequenceMatch.generate_lcs(encodings_a[i].ids[1:-1],
                                               encodings_b[i].ids[1:-1],
                                               **kwargs))

            # Remove empty matches and generate/return samples using token masks
            # as labels. The BOS/EOS tokens are assigned a label of -100.
            indices = [
                i for i in range(len(matches))
                if matches[i].mask_a.sum() >= self.min_match_total_length
            ]
            encodings = [[encodings_a[i], encodings_b[i]] for i in indices]
            labels = [[
                np.hstack([[-100], matches[i].mask_a, [-100]]),
                np.hstack([[-100], matches[i].mask_b, [-100]])
            ] for i in indices]
            for i in np.arange(len(encodings)):
                for j in range(2):
                    yield {
                        'input_ids': encodings[i][j].ids,
                        'attention_mask': encodings[i][j].attention_mask,
                        'labels': labels[i][j]
                    }
