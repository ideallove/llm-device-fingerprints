from pathlib import Path
import shutil

import datasets

from ..typing import PathLike


def convert_parquet(parquet_file: PathLike,
                    dataset_dir: PathLike,
                    batch_size: int = 1000):
    """Converts a Parquet file to a Hugging Face dataset.

    Args:
        parquet_file: Path to the Parquet file.
        dataset_dir : Directory for saving the Hugging Face dataset.
        batch_size: Batch size for writing the Hugging Face dataset.
    """
    dataset_dir = Path(dataset_dir)
    cache_dir = dataset_dir / 'cache'
    dataset = datasets.Dataset.from_parquet(str(parquet_file),
                                            cache_dir=cache_dir,
                                            batch_size=batch_size)
    dataset.save_to_disk(dataset_dir)
    shutil.rmtree(cache_dir)
