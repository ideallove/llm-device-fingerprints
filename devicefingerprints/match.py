from dataclasses import dataclass
import difflib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from .singleton import Glob
from .span import Spans


@dataclass
class SequenceMatch:
    """Represents matching subsequences extracted from a pair of sequences.

    If provided with spans (i.e., mappings between span names and the associated
    (start, end) indices in a sequence), matches are assumed to only appear
    between spans with matching names.

    Attributes:
        a: First sequence.
        b: Second sequence.
        block: Matching subsequences in `a` and `b`, or a mapping between span
            names and their associated matches.
        a_spans: Spans associated with `a`.
        b_spans: Spans associated with `b`.
        mask_a: Boolean mask over `a` that marks matching subsequences with `b`.
        mask_b: Boolean mask over `b` that marks matching subsequences with `a`.
        pattern: Pattern associated with the match, or a mapping between span
            names and their associated patterns. A pattern is comprised of
            matching subsequences separated by glob patterns.
    """

    a: list
    b: list
    blocks: Union[List[difflib.Match], Dict[str, List[difflib.Match]]]
    a_spans: Optional[Spans] = None
    b_spans: Optional[Spans] = None

    @property
    def mask_a(self) -> npt.NDArray[np.bool_]:
        blocks = self.blocks
        if isinstance(blocks, dict):
            blocks = sum(blocks.values(), [])

        mask = np.zeros(len(self.a), np.bool_)
        for block in blocks:
            mask[block.a:block.a + block.size] = True

        return mask

    @property
    def mask_b(self) -> npt.NDArray[np.bool_]:
        blocks = self.blocks
        if isinstance(blocks, dict):
            blocks = sum(blocks.values(), [])

        mask = np.zeros(len(self.b), np.bool_)
        for block in blocks:
            mask[block.b:block.b + block.size] = True

        return mask

    @property
    def pattern(self) -> Union[list, Dict[str, list]]:
        if isinstance(self.blocks, list):
            pattern = []
            for i, block in enumerate(self.blocks):
                if not i and (block.a or block.b):
                    pattern += [Glob]

                pattern += self.a[block.a:block.a + block.size]
                if (i != len(self.blocks) - 1 or
                        block.a + block.size != len(a) or
                        block.b + block.size != len(b)):
                    pattern += [Glob]

            return pattern

        pattern = {}
        for key, blocks in self.blocks.items():
            a_span = self.a_spans[key]
            b_span = self.b_spans[key]

            s = []
            for i, block in enumerate(blocks):
                if not i and (block.a != a_span[0] or block.b != b_span[0]):
                    s += [Glob]

                s += self.a[block.a:block.a + block.size]
                if (i != len(blocks) - 1 or block.a + block.size != a_span[1] or
                        block.b + block.size != b_span[1]):
                    s += [Glob]

            pattern[key] = s

        return pattern

    @staticmethod
    def generate_lcs(
        a: Sequence,
        b: Sequence,
        a_spans: Optional[Spans] = None,
        b_spans: Optional[Spans] = None,
        min_length: int = 1,
        is_junk_element: Optional[Callable[[object], bool]] = None,
        is_junk_sequence: Optional[Callable[[list], bool]] = None
    ) -> "SequenceMatch":
        """Generates LCS matches from a pair of sequences.

        Uses longest common subsequence matching to extract common subsequences
        from the provided pair. If provided with spans (i.e., mappings between
        span names and the associated (start, end) indices in a sequence), LCS
        matching is only performed between spans with matching keys.

        Args:
            a: First sequence.
            b: Second sequence.
            a_spans: Spans associated with `a`.
            b_spans: Spans associated with `b`.
            min_length: Minimum length for individual (contiguous) matches.
            is_junk_element: A function that returns True if a sequence element
                is considered junk and cannot appear in a match.
            is_junk_sequence: A funtion that returns True if a subsequence match
                is considered junk and should be ignored.

        Returns:
            Matching subsequences extracted from `a` and `b`.

        Raises:
            ValueError: If only one of `a_spans` and `b_spans` is specified.
        """
        if (a_spans is not None) + (b_spans is not None) == 1:
            raise ValueError(
                'Either both or none of `a_spans` and `b_spans` can be None')

        a = list(a)
        b = list(b)
        if a_spans is None:
            matcher = difflib.SequenceMatcher(a=a,
                                              b=b,
                                              isjunk=is_junk_element,
                                              autojunk=False)
            blocks = []
            for block in matcher.get_matching_blocks():
                if block.size >= min_length:
                    if (is_junk_sequence is None or not is_junk_sequence(
                            a[block.a:block.a + block.size])):
                        blocks += [block]

            return SequenceMatch(a, b, blocks)

        blocks = {}
        for key, a_span in a_spans.items():
            if key not in b_spans:
                continue

            b_span = b_spans[key]
            match = SequenceMatch.generate_lcs(
                a[a_span[0]:a_span[1]],
                b[b_span[0]:b_span[1]],
                min_length=min_length,
                is_junk_element=is_junk_element,
                is_junk_sequence=is_junk_sequence)
            if match.blocks:
                blocks[key] = [
                    difflib.Match(a_span[0] + block.a, b_span[0] + block.b,
                                  block.size) for block in match.blocks
                ]

        return SequenceMatch(a, b, blocks, a_spans=a_spans, b_spans=b_spans)
