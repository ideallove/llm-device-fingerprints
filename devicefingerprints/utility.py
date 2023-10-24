import gzip
from pathlib import Path
from typing import IO

from .typing import PathLike


def fopen(file: PathLike, mode: str = 'r') -> IO:
    """Opens a regular (uncompressed) or a gzip compressed file.

    Args:
        file: Path to the file to open.
        mode: Mode to open the file in.

    Returns:
        Handle to the opened file.
    """
    file = Path(file)
    if len(mode) == 1:
        mode += 't'
    if file.suffix == '.gz':
        return gzip.open(file, mode)

    return open(file, mode)
