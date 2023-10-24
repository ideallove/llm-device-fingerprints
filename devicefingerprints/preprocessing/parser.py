from abc import abstractmethod

import pyarrow as pa


class Parser:
    """Base class for parsing examples.

    Attributes:
        schema: Arrow schema corresponding to parsed examples.
        sort_column: Column name to use for sorting examples.
    """

    @property
    def schema(self) -> pa.Schema:
        return self._get_schema()

    @property
    def sort_column(self) -> str:
        return self._get_sort_column()

    @abstractmethod
    def _get_schema(self) -> pa.Schema:
        # Returns the Arrow schema corresponding to parsed examples.
        ...

    def _get_sort_column(self) -> str:
        # Returns the column name to use for sorting parsed examples.
        return None

    @abstractmethod
    def parse(self, line: str) -> dict:
        """Parses and returns an example (i.e., a single line from a data file).

        Args
            line: Line to parse.

        Returns:
            Parsed example.
        """
        ...
