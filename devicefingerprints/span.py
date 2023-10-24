from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, Dict, Optional, Tuple, Union

# Mapping between span names and the associated (start, end) indices.
Spans = Dict[str, Tuple[int, int]]


class Spanner(ABC):
    """Base class for extracting spans from input examples."""

    @abstractmethod
    def __call__(self, example) -> Optional[Spans]:
        """Extracts spans from the provided example.

        Args:
            example: Input example.

        Returns:
            Spans associated with `example`. Each item maps a span name to its
                associated (start, end) indices in `example`. Returns None if
                not applicable.
        """
        ...

    @abstractmethod
    def split(self, example) -> Union[str, Dict[str, str]]:
        """Splits the provided example into its associated spans.

        Args:
            example: Input example.

        Returns:
            Mapping between span names and the associated substrings in
                `example`. Returns `example` if not applicable.
        """
        ...


@dataclass
class BannerSpanner(Spanner):
    """Class for extracting spans from banners.

    Splits HTTP banners into individual headers, with header names as span
    names. The status line is extracted as a span named 'status'. The extracted
    spans do not contain the header names themselves, or the 'HTTP/' prefix in
    the status line. Note that all non-HTTP banners are returned unchanged.

    Attributes:
        service_name: Default service name to use when one is not provided.
    """

    service_name: Optional[str] = None

    def __call__(self, example: Union[str, Dict[str, Any]]) -> Optional[Spans]:
        """Extracts spans from the provided example.

        Args:
            example: A banner, or a dictionary with the form {'banner': ...,
                'service_name': ...}. If a service name is not specified, it is
                inferred using the default service name.

        Return:
            Spans associated with `example` for HTTP banners, None otherwise.
                Each item maps a span name to its associated (start, end)
                indices in `example`.
        """
        service_name = self.service_name
        if isinstance(example, dict):
            service_name = example.get('service_name', self.service_name)
            example = example['banner']

        if service_name == 'HTTP':
            return self._http_spanner(example)

        return

    @staticmethod
    def _http_spanner(banner: str) -> Spans:
        # Extracts spans from an HTTP banner.
        spans = {}
        pos = 0
        for match in re.finditer('(\r?\n(?![ \t])|$)', banner):
            span = match.span()
            line = banner[pos:span[0]]
            if not line:
                continue

            if pos:
                key, value = line.split(':', 1)
                spans[key] = (pos + len(key) + 1, span[0])
            else:
                prefix = 'HTTP/'
                assert line.startswith(prefix)
                spans['status'] = (len(prefix), len(line))

            pos = span[1]

        return spans

    def split(
            self, example: Union[str, Dict[str,
                                           Any]]) -> Union[str, Dict[str, str]]:
        """Splits the provided example into its associated spans.

        Args:
            example: A banner, or a dictionary with the form {'banner': ...,
                'service_name': ...}. If a service name is not specified, it is
                inferred using the default service name.

        Returns:
            For HTTP banners, a mapping between span names and the associated
                substrings in `example`. Returns `example` for non-HTTP banners.
        """
        spans = self(example)
        if isinstance(example, dict):
            example = example['banner']

        if spans is None:
            return example

        return {key: example[span[0]:span[1]] for key, span in spans.items()}
