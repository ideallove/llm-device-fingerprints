import functools
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import transformers

from .basedataset import BaseDataset, RawDataset

# Hugging Face logger.
logger = transformers.utils.logging.get_logger(__name__)


class SmartBatchDataset(BaseDataset):
    """Dataset for generating dynamically-sized minibatches.

    Sorts examples in a batch according to their length, and then generates
    minibatches containing at most `max_tokens` token IDs. This is repeated
    until `gradient_accumulation_steps` minibatches have been generated, or the
    batch has been exhausted. Use in conjunction with SmartBatchTrainer.
    """

    def __init__(self,
                 data: RawDataset,
                 column: str,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 data_collator: transformers.DataCollator,
                 batch_size: int = 1024,
                 gradient_accumulation_steps: int = 1,
                 max_tokens: Optional[int] = None,
                 max_length: Optional[int] = None,
                 return_overflowing_tokens: bool = False,
                 stride: int = 0,
                 shuffle: bool = False,
                 seed: Optional[int] = None,
                 sort: bool = False,
                 skip_batches: int = 0):
        """Creates a dataset.

        Args:
            data: Associated dataset.
            column: Column containing the associated examples.
            tokenizer: Tokenizer for encoding the data.
            data_collator: Data collator to use to form a batch.
            batch_size: Batch size, split into `gradient_accumulation_steps`
                minibatches, with each minibatch containing at most `max_tokens`
                token IDs.
            skip_batches: Number of batches to skip at the beginning (e.g., if
                resuming training from a previous checkpoint).
            gradient_accumulation_steps: Number of update steps that gradients
                are accumulated for in the corresponding trainer. If a batch is
                exhausted before reaching this number of steps, returns empty
                minibatches for the remaining steps.
            max_tokens: Maximum number of tokens in each generated minibatch.
            max_length: If provided, tokenized sequences are truncated to this
                maximum length.
            return_overflowing_tokens: Whether to return overflowing tokens from
                truncated sequences. Ignored if `max_length` is not provided.
            stride: If `max_length` is provided and `return_overflowing_tokens`
                is set to True, the returned overflowing tokens will contains
                this number of tokens from the end of the truncated sequence.
            shuffle: Whether to shuffle the datasets (in a consistent way).
            seed: Random seed for shuffling.
            sort: Whether to sort the dataset according to the length of
                examples in `column`. Can be used to speed up evaluation.

        Raises:
            ValueError: If both `sort` and `shuffle` are True.
        """
        super().__init__(data,
                         shuffle=shuffle,
                         seed=seed,
                         sort=sort,
                         sort_column=column,
                         batch_size=batch_size)
        self.column = column
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.skip_batches = skip_batches
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.return_overflowing_tokens = return_overflowing_tokens
        self.stride = stride

    def __len__(self) -> int:
        num_batches = int(np.ceil(self.data.num_rows / self.batch_size))
        return num_batches * self.gradient_accumulation_steps

    def _iter(self) -> Iterator[Dict[str, Any]]:
        # Get worker information for multiprocess data loading.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise ValueError('Using multiple workers (num_workers = '
                             f'{worker_info.num_workers}) is not supported')

        tokenize = functools.partial(
            self.tokenizer,
            padding=False,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=self.return_overflowing_tokens)
        dataset = self.data.with_format(columns=[self.column])
        for i in range(0, dataset.num_rows, self.batch_size):
            if i < self.skip_batches * self.batch_size:
                for _ in range(self.gradient_accumulation_steps):
                    yield {'input_ids': np.zeros([0, 0])}

                if i == self.skip_batches * self.batch_size - 1:
                    self.skip_batches = 0

                continue

            encodings = tokenize(dataset[i:i + self.batch_size][self.column])
            num_examples = len(encodings.input_ids)

            lengths = np.array([len(ids) for ids in encodings.input_ids])
            indices = np.argsort(lengths)
            lengths = lengths[indices]
            encodings = [{
                'input_ids': encodings.input_ids[i],
                'attention_mask': encodings.attention_mask[i]
            } for i in indices]
            encodings = self.data_collator(encodings)
            num_active_tokens = (encodings.labels != -100).sum()

            start = 0
            num_minibatches = 0
            for end in range(1, num_examples + 1):
                if end != num_examples:
                    num_tokens = (end - start + 1) * lengths[end]
                if (end == num_examples or (self.max_tokens is not None and
                                            num_tokens > self.max_tokens)):
                    if num_minibatches < self.gradient_accumulation_steps:
                        length = lengths[end - 1]
                        if self.data_collator.pad_to_multiple_of is not None:
                            multiple_of = self.data_collator.pad_to_multiple_of
                            if length % multiple_of:
                                length += multiple_of - length % multiple_of

                        # The total number of active tokens in the batch is used
                        # to weight the loss from each minibatch.
                        minibatch = {
                            key: value[start:end, :length]
                            for key, value in encodings.items()
                        }
                        minibatch['num_active_tokens'] = num_active_tokens
                        yield minibatch

                    start = end
                    num_minibatches += 1

            if num_minibatches > self.gradient_accumulation_steps:
                logger.warn(
                    f'Batch was split into {num_minibatches} mini-batches, '
                    'which is higher than gradient_accumulation_steps of '
                    f'{self.gradient_accumulation_steps}. Consider increasing '
                    'gradient_accumulation_steps.')
            for i in range(num_minibatches, self.gradient_accumulation_steps):
                yield {'input_ids': np.zeros([0, 0])}
