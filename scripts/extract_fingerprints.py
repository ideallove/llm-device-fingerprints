"""
Extracts regular expression fingerprints from clustered HTTP embeddings.
提取从聚类的HTTP banner中提取正则表达式指纹。
"""

import argparse
from pathlib import Path
import pickle

import numpy as np

import devicefingerprints as dfp


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_dir',
                        type=lambda p: Path(p).resolve(),
                        help='Path to pre-comupted clusters.')
    parser.add_argument('--num-samples',
                        default=10,
                        type=int,
                        help='Number of samples to draw from a cluster.')
    parser.add_argument('--num-draws',
                        default=100,
                        type=int,
                        help='Number of draws per cluster.')
    parser.add_argument('--min-length',
                        default=5,
                        type=int,
                        help='Minimum length for individual (contiguous) '
                        'matches from a pair of examples.')
    parser.add_argument('--num-workers',
                        default=1,
                        type=int,
                        help='Number of paraller worker threads.')
    parser.add_argument('--seed', default=None, type=int, help='Random seed.')

    args = parser.parse_args()

    dataset = dfp.InternetDataset.load_from_disk(args.dataset_dir)
    banners = dataset['banner']
    columns = [c for c in dataset.column_names if c.startswith('label-')]
    columns = sorted(columns, key=lambda c: float(c.split('-')[-1]))
    labels = np.vstack([dataset[c] for c in columns]).T

    spanner = dfp.BannerSpanner(service_name='HTTP')
    extractor = dfp.FingerprintExtractor(num_samples=args.num_samples,
                                         num_draws=args.num_draws,
                                         min_length=args.min_length,
                                         spanner=spanner,
                                         num_workers=args.num_workers,
                                         seed=args.seed)
    extractor.fit(banners, labels, exclude_labels=[-1])
    with open(args.dataset_dir / 'fingerprints.pkl', 'wb') as f:
        pickle.dump(extractor, f)

    del extractor.fingerprints['Content-Security-Policy-Report-Only']
    matches = extractor.transform(banners)
    with open(args.dataset_dir / 'matches.pkl', 'wb') as f:
        pickle.dump(matches, f)


if __name__ == '__main__':
    main()
