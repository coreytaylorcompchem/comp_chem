import contextlib
import dataclasses
import enum
import functools
import io
import numbers
from typing import Any, Callable, Literal, Optional, Protocol, Tuple, Union
from xml.etree import ElementTree as ET

import pandas as pd
from aqemia.cdd_vault import utils
from awswrangler import s3
from typing_extensions import TypeAlias
from yarl import URL

import mol_cards


# need that construct instead of sentinel = object() for typing
class _SentinelType(enum.Enum):
    SENTINEL = enum.auto()


_SENTINEL = _SentinelType.SENTINEL


_normalized_modifiers = {
    '<': '<',
    "<=": "<=",
    '≤': '<=',
    "=": "=",
    "≥": ">=",
    ">=": ">=",
    '>': '>',
}
_modifiers = ["<", "<=", "=", ">=", ">"]

_modifier_inv = {mod: _modifiers[-i - 1] for i, mod in enumerate(_modifiers)}

# matrix indexed by [left, right] in the order of the dict above
_modifier_prod_matrix = [
    # <, <=, =, >=, >
    # <
    ["<", "<", "<", "?", "?"],
    # <=
    ["<", "<=", "<=", "?", "?"],
    # =
    ["<", "<=", "=", ">=", ">"],
    # >=
    ["?", "?", ">=", ">=", ">"],
    # >
    ["?", "?", ">", ">", ">"],
]
_modifier_prod = {
    (x, y): _modifier_prod_matrix[i][j]
    for i, x in enumerate(_modifiers)
    for j, y in enumerate(_modifiers)
}

# matrix indexed by [numerator, denominator] in the order of the dict above
_modifier_div_matrix = [
    # <, <=, =, >=, >
    # <
    ["?", "?", "<", "<", "<"],
    # <=
    ["?", "?", "<=", "<=", "<"],
    # =
    ["<", "<=", "=", ">=", ">"],
    # >=
    [">", ">=", ">=", "?", "?"],
    # >
    [">", ">", ">", "?", "?"],
]
_modifier_div = {
    (x, y): _modifier_prod_matrix[i][j]
    for i, x in enumerate(_modifiers)
    for j, y in enumerate(_modifiers)
}


@dataclasses.dataclass(frozen=True)
class MonotonousThreshold:
    low: float
    high: float
    direction: Literal["increasing", "decreasing"]


@dataclasses.dataclass(frozen=True)
class IntervalThreshold:
    min: float
    max: float


ThresholdType: TypeAlias = Union[MonotonousThreshold, IntervalThreshold]


class Colors:
    """namespace for storing data"""

    GOOD = "green"
    MEDIUM = "orange"
    BAD = "red"


@functools.singledispatch
def get_value_color(threshold: Any, value: float) -> str:
    raise NotImplementedError(f"{type(threshold).__name__} object {threshold=}")


@get_value_color.register(type(None))
def _get_value_color_none(threshold: None, value: float) -> Any:
    return "dark"


@get_value_color.register(MonotonousThreshold)
def _get_value_color_monotonous(threshold: MonotonousThreshold, value: float) -> Any:
    increasing = threshold.direction == "increasing"
    low = threshold.low
    high = threshold.high
    if low > high:
        raise RuntimeError(f"Incompatible bounds: low bound {low} > high bound {high}")
    if value > high:
        return Colors.GOOD if increasing else Colors.BAD
    elif high >= value >= low:
        return Colors.MEDIUM
    elif low > value:
        return Colors.BAD if increasing else Colors.GOOD
    else:
        raise RuntimeError("Impossible code-path")


@get_value_color.register(IntervalThreshold)
def _get_value_color_interval(threshold: IntervalThreshold, value: float) -> Any:
    if threshold.min <= value <= threshold.max:
        return Colors.GOOD
    else:
        return Colors.BAD


def _colored(
    value: Union[float, str],
    *,
    color: Optional[str] = None,
    threshold: Optional[ThresholdType] = None,
    modifier: Optional[str] = None,
    color_only: bool = False,
) -> ET.Element:
    if color is None:
        color = get_value_color(threshold, value)
    if color_only:
        return mol_cards.html.color_square(color)
    text = f"{value:.2f}" if isinstance(value, (int, float, numbers.Real)) else value
    return mol_cards.html.span(f"{modifier or ''}{text}", color=color)


def colored_number(
    column: str,
    threshold: ThresholdType,
    *,
    modifier_column: Union[_SentinelType, None, str] = None,
    color_only: bool = False,
) -> Callable[[pd.Series], ET.Element]:
    if modifier_column is _SENTINEL:
        modifier_column = f"{column}_modifier"

    def callback(row: pd.Series) -> ET.Element:
        if pd.notnull(row[column]):
            value = float(row[column])
            modifier = ""
            if modifier_column is not None:
                modifier = row[f"{column}_modifier"]
                modifier = "" if modifier == "=" else modifier
            return _colored(value, threshold=threshold, modifier=modifier, color_only=color_only)
        else:
            if color_only:
                return mol_cards.html.color_square("#00000000")
            return mol_cards.html.as_html("-")

    return callback


def join(
    *parts: Union[mol_cards.HtmlLike, Callable[[pd.Series], ET.Element]],
    sep: Optional[Union[mol_cards.HtmlLike, Callable[[pd.Series], ET.Element]]] = None,
) -> Callable[[pd.Series], ET.Element]:
    def callback(row: pd.Series):
        iterable = parts
        if sep is not None:
            local_sep = sep(row) if callable(sep) else sep
            iterable = mol_cards.html.generalized_join(local_sep, iterable)
        elements = [part(row) if callable(part) else part for part in iterable]
        return mol_cards.html.join(*elements)

    return callback


def _divide(
    numerator: float,
    denominator: float,
    *,
    # Assume "=" modifier if not provided
    # float allows np.nan to be passed from row, we handle it
    numerator_modifier: Union[float, str] = "=",
    denominator_modifier: Union[float, str] = "=",
) -> Tuple[float, str]:
    """Compute a division of values with modifier"""
    ratio = numerator / denominator
    if pd.notna(ratio):
        if pd.isna(numerator_modifier) or pd.isna(denominator_modifier):
            # One modifier is undefined while it should, no longer assume "="
            modifier = "?"
        elif isinstance(numerator_modifier, float):
            raise ValueError("numerator_modifier argument cannot be float if it is not nan")
        elif isinstance(denominator_modifier, float):
            raise ValueError("denominator_modifier argument cannot be float if it is not nan")
        else:
            modifier = _modifier_div[numerator_modifier, denominator_modifier]
    else:
        # Value is unknown, modifier is not meaningful, no need for a special value
        modifier = "="
    return ratio, modifier


def column_ratio(
    numerator: str,
    denominator: str,
    *,
    numerator_modifier: Optional[str] = None,
    denominator_modifier: Optional[str] = None,
) -> Callable[[pd.Series], Tuple[float, str]]:
    """Create a function that divide two columns"""

    if (numerator_modifier is None) and (denominator_modifier is None):

        def callback(row: pd.Series) -> Tuple[float, str]:
            return _divide(row[numerator], row[denominator])

    elif (numerator_modifier is not None) and (denominator_modifier is not None):

        def callback(row: pd.Series) -> Tuple[float, str]:
            return _divide(
                row[numerator],
                row[denominator],
                numerator_modifier=row[numerator_modifier],
                denominator_modifier=row[denominator_modifier],
            )

    else:
        raise ValueError("numerator and denominator modifiers should both be provided or None")

    return callback


def colored_value(
    value: Union[
        str,  # column
        Tuple[str, str],  # column & modifier column
        Callable[[pd.Series], Union[float, Tuple[float, str]]],  # computed value
    ],
    threshold: Optional[ThresholdType] = None,
    *,
    str_format: str = "{:.2f}",
    color_only: bool = False,
) -> Callable[[pd.Series], ET.Element]:
    """Create a function that colors a (computed) value"""

    def callback(row: pd.Series) -> ET.Element:
        if isinstance(value, str):
            val = row[value]
            modifier = "="
        elif isinstance(value, tuple):
            val = row[value[0]]
            modifier = row[value[1]]
        elif callable(value):
            result = value(row)
            if isinstance(result, tuple):
                val, modifier = result
            else:
                val = result
                modifier = "="
        else:
            raise RuntimeError(f"Unsupported 'value' argument {value}")

        if pd.isna(val):
            color = "#00000000" if color_only else "dark"
            val = "-"
        elif modifier == "?":
            color = "purple"
            val = "?"
        else:
            color = get_value_color(threshold, val)
            val = str_format.format(val)
        modifier = "" if modifier == "=" else modifier
        return _colored(val, modifier=modifier, color=color, color_only=color_only)

    return callback


class Affinity(Protocol):
    um: str
    modifier: str


def selectivity(
    target: Affinity,
    versus: Affinity,
    *,
    threshold: ThresholdType,
    color_only: bool = False,
) -> Callable[[pd.Series], ET.Element]:
    return colored_value(
        value=column_ratio(
            numerator=versus.um,
            denominator=target.um,
            numerator_modifier=versus.modifier,
            denominator_modifier=target.modifier,
        ),
        threshold=threshold,
        str_format="{:.1f}x",
        color_only=color_only,
    )

    # def callback(row: pd.Series) -> ET.Element:
    #     ratio, modifier = _divide(
    #         row[versus.um],
    #         row[target.um],
    #         numerator_modifier=row[versus.modifier],
    #         denominator_modifier=row[target.modifier],
    #     )
    #     if pd.isna(ratio):
    #         color = "#00000000" if color_only else "dark"
    #         value = "-"
    #         modifier = None
    #     elif modifier == "?":
    #         color = "purple"
    #         value = "?x"
    #     else:
    #         color = get_value_color(threshold, ratio)
    #         modifier = "" if modifier == "=" else modifier
    #         value = f"{ratio:.1f}x"
    #     return _colored(value, modifier=modifier, color=color, color_only=color_only)

    # return callback


def upload_s3(content: Union[str, bytes], remote: Union[str, URL]) -> URL:
    """Upload some content to an S3 file"""
    url = URL(remote)
    if url.scheme != "s3":
        raise ValueError(f"{url} is not an S3 URL")
    with utils.Timer(msg=f"Uploading {url}"):
        # Allows for conditional with-statement
        with contextlib.ExitStack() as stack:
            buffer = stack.enter_context(io.BytesIO())
            if isinstance(content, str):
                wrapper = stack.enter_context(io.TextIOWrapper(buffer, encoding="utf-8", write_through=True))
                wrapper.write(content)
                wrapper.flush()
                wrapper.seek(0)
            elif isinstance(content, bytes):
                buffer.write(content)
            else:
                raise TypeError(f"Unhandled content argument with type {type(content)}")
            buffer.seek(0)
            s3.upload(buffer, str(url))
    return url

            