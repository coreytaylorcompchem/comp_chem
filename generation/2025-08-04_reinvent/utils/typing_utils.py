"""
Utilities for python type system andwriting type annotations
"""
from dataclasses import Field
from typing import Any, ClassVar, Dict, Iterable, Protocol, TypeVar

# Keep both __all__ and module category then alphabetically sorted
__all__ = [
    # TypeVar
    "K",
    "K_contra",
    "V_co",
    # Protocol
    "Indexable",
    "SupportsKeysAndGetItem",
    "DataclassInstance",
]

K = TypeVar("K")
"""A type variable representing the type of keys"""


K_contra = TypeVar("K_contra", contravariant=True)
"""A type variable representing the type of keys of a mapping-like type

Note
----
This type variable is for generic types that are contravariant in their key type.
A discussion on variance, contravariance and invariance for type variable is outside
the scope of this documentation, and is an advanced typing topic. See
[wikipedia](https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science))
for a first introduction.
"""


V_co = TypeVar("V_co", covariant=True)
"""A type variable representing the type of values of a mapping-like type

Note
----
This type variable is for generic types that are covariant in their value type.
A discussion on variance, contravariance and invariance for type variable is outside
the scope of this documentation, and is an advanced typing topic. See
[wikipedia](https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science))
for a first introduction.
"""


class Indexable(Protocol[K_contra, V_co]):
    """
    Protocol for anything supporting indexing

    This describes anything that supports indexing, i.e. ``obj[key]``, for known
    types for the key and returned value
    """

    def __getitem__(self, __key: K_contra) -> V_co:
        ...


# copied from https://github.com/python/typeshed/blob/8365b1aaefd46d506ca0dfe73e9721da2d03c566/stdlib/_typeshed/__init__.pyi#L116
# because it is used in dict.__init__ annotations


class SupportsKeysAndGetItem(Protocol[K, V_co]):
    """
    Protocol for object supporting indexing and the keys() method

    This is a very small subpart of what `dict` supports, but all that
    is needed by some functions.
    """

    def keys(self) -> Iterable[K]:
        ...

    def __getitem__(self, __key: K) -> V_co:
        ...


# Copied from https://github.com/python/typeshed/blob/4ca5ee98df5654d0db7f5b24cd2bd3b3fe54f313/stdlib/_typeshed/__init__.pyi#L309


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, "Field[Any]"]]