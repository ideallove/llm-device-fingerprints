#!/usr/bin/env python
"""Generates a dataset from a Censys Universal Internet Dataset snapshot."""

import argparse
from pathlib import Path
import random

import xxhash

import devicefingerprints as dfp


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-dir',
        type=lambda p: Path(p).resolve(),
        required=True,
        help=('Path to the directory containing data files (JSON files '
              'exported from BigQuery).'))
    parser.add_argument(
        '--parquet-dir',
        default='data/parquet',
        type=lambda p: Path(p).resolve(),
        help='Path to the directory for saving the Parquet file.')
    parser.add_argument(
        '--dataset-dir',
        default='data/dataset',
        type=lambda p: Path(p).resolve(),
        help='Path to the directory for saving the Hugging Face dataset.')
    parser.add_argument(
        '--detect-encoding',
        action='store_true',
        help=('Whether to try and detect the encoding of banners for '
              'converting them to unicode strings.'))
    parser.add_argument(
        '--default-encoding',
        default='utf-8',
        type=str,
        help='Encoding to use to when encoding detection fails.')
    parser.add_argument('--include-truncated',
                        action='store_true',
                        help='Whether to include truncated examples.')
    parser.add_argument(
        '--include-empty-banner',
        action='store_true',
        help='Whether to include examples with an empty banner.')
    parser.add_argument(
        '--sort',
        action='store_true',
        help='Whether to sort examples before writing to the output.')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Whether to shuffle examples before writing to the output.')
    parser.add_argument(
        '--subsample',
        default=1.0,
        type=float,
        help='If less than one, (filtered) examples are sampled at this rate.')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='Random seed for shuffling/sampling examples.')
    parser.add_argument('--num-workers',
                        default=1,
                        type=int,
                        help='Number of parallel workers for parsing examples.')
    parser.add_argument('--chunk-size',
                        default=1000,
                        type=int,
                        help='Chunk size for workers.')
    parser.add_argument('--batch-size',
                        default=1000,
                        type=int,
                        help='Batch size for writing to the output.')

    args = parser.parse_args()

    data_files = sorted(args.data_dir.glob('*.json.gz'))
    parser = dfp.preprocessing.CensysParser(
        detect_encoding=args.detect_encoding,
        default_encoding=args.default_encoding)
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2**64 - 1)

    def filter_func(example: dict) -> bool:
        drop = False
        if not args.include_truncated:
            drop |= example['truncated']
        if not args.include_empty_banner:
            drop |= not example['banner']
        if args.subsample < 1:
            digest = xxhash.xxh64_intdigest(f"{example['ip']}-{seed}")
            drop |= (digest / 2**64) >= args.subsample

        return not drop

    generator = dfp.preprocessing.ParquetGenerator(data_files=data_files,
                                                   parser=parser,
                                                   filter_func=filter_func,
                                                   sort=args.sort,
                                                   shuffle=args.shuffle,
                                                   seed=args.seed,
                                                   num_workers=args.num_workers,
                                                   chunk_size=args.chunk_size,
                                                   batch_size=args.batch_size)

    args.parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = args.parquet_dir / 'data.parquet'
    generator.generate(parquet_file)

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    dfp.preprocessing.convert_parquet(parquet_file,
                                      args.dataset_dir,
                                      batch_size=args.batch_size)


if __name__ == '__main__':
    main()
