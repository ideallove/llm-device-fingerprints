#!/usr/bin/env python
"""Clusters banner embeddings using HDBSCAN."""

import argparse
from pathlib import Path
import shutil
from typing import Optional

import cuml
import numpy as np
import sklearn.utils

import devicefingerprints as dfp


def apply_pca(embeddings: np.ndarray,
              num_components: int,
              num_samples: Optional[int] = None,
              seed: Optional[int] = None) -> np.ndarray:
    embeddings_resampled = None
    if num_samples is not None and embeddings.shape[0] > num_samples:
        embeddings_resampled = sklearn.utils.resample(embeddings,
                                                      replace=False,
                                                      n_samples=num_samples,
                                                      random_state=seed)

    pca = cuml.PCA(n_components=num_components, random_state=seed)
    if embeddings_resampled is None:
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        pca.fit(embeddings_resampled)
        embeddings_reduced = []
        for i in range(0, embeddings.shape[0], num_samples):
            embeddings_reduced += [pca.transform(embeddings[i:i + num_samples])]

        embeddings_reduced = np.vstack(embeddings_reduced)

    explained_variance_ratio = pca.explained_variance_ratio_.sum()
    print(f'Explained variance ratio: {100 * explained_variance_ratio:.2f}%')

    return embeddings_reduced


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_dir',
                        type=lambda p: Path(p).resolve(),
                        help='Path to pre-computed banner embeddings.')
    parser.add_argument('--output-dir',
                        default='clusters',
                        type=lambda p: Path(p).resolve(),
                        help='Path for saving the generated clusters.')
    parser.add_argument('--num-samples',
                        default=int(5e6),
                        type=int,
                        help='Maximum number of samples to cluster.')
    parser.add_argument(
        '--pca-num-samples',
        default=None,
        type=int,
        help='Maximum number of samples for PCA dimensionality reduction.')
    parser.add_argument(
        '--pca-num-components',
        default=64,
        type=int,
        help='Number of components for PCA dimensionality reduction.')
    parser.add_argument('--min-cluster-size',
                        default=50,
                        type=int,
                        help='Minimum cluster size (for HDBSCAN).')
    parser.add_argument('--min-samples',
                        default=5,
                        type=int,
                        help='Number of samples in a neighbourhood for a point '
                        'to be considered a core point (for HDBSCAN).')
    parser.add_argument('--cluster-selection-epsilon',
                        nargs='*',
                        default=[0.25, 0.3, 0.35, 0.4, 0.45],
                        type=float,
                        help='Distance threshold(s) (for HDBSCAN).')
    parser.add_argument('--seed', default=None, type=int, help='Random seed.')

    args = parser.parse_args()
    cuml.common.logger.set_level(cuml.common.logger.level_error)

    for path in args.dataset_dir.iterdir():
        service_name = path.name
        print(f'Clustering {service_name}')

        dataset = dfp.InternetDataset.load_from_disk(path)
        embeddings = np.load(path / 'embeddings.npz')['embeddings']

        if args.num_samples is not None and dataset.num_rows > args.num_samples:
            indices = sklearn.utils.resample(np.arange(dataset.num_rows),
                                             replace=False,
                                             n_samples=args.num_samples,
                                             random_state=args.seed)
            indices = np.sort(indices)
            dataset = dataset.select(indices)
            embeddings = embeddings[indices]

        if args.pca_num_components is not None:
            embeddings = apply_pca(embeddings,
                                   args.pca_num_components,
                                   num_samples=args.pca_num_samples,
                                   seed=args.seed)

        for epsilon in args.cluster_selection_epsilon:
            params = {
                'min_cluster_size': args.min_cluster_size,
                'min_samples': args.min_samples,
                'cluster_selection_epsilon': epsilon
            }
            hdbscan = cuml.cluster.HDBSCAN(**params)
            labels = hdbscan.fit_predict(embeddings)
            print(f'epsilon={epsilon}: {labels.max() + 1} clusters, '
                  f'{100 * (labels == -1).mean():.2f}% anomalies')

            dataset = dataset.add_column(f'label-{epsilon}', labels)

        path = args.output_dir / service_name
        if path.is_dir():
            shutil.rmtree(path)

        path.mkdir(parents=True)
        dataset.flatten_indices()
        dataset.save_to_disk(path)


if __name__ == '__main__':
    main()
