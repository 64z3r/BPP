import struct
import hashlib
from typing import (
    TypeAlias,
    Protocol,
    Sequence,
    Mapping,
    MutableSet,
    FrozenSet,
)


Set: TypeAlias = FrozenSet | MutableSet

PHashable: TypeAlias = None | str | int | float | complex
Hashable: TypeAlias = (
    PHashable
    | Mapping[PHashable, "Hashable"]
    | Sequence["Hashable"]
    | FrozenSet[PHashable]
    | MutableSet[PHashable]
)


class Hash(Protocol):
    """Protocol for hash classes."""

    def update(self, data: Hashable) -> None:
        """Update state.

        Args:
            data: Data to be digested.
        """

    def hexdigest(self) -> str:
        """Digests data.

        Returns:
            Hex-string of digested data.
        """


class DigestError(TypeError):
    """Raised if :func:`digest` cannot hash a given object."""


def digest(obj: Hashable, hash: Hash | None = None) -> Hash:
    """Digest a given object and return hash.

    Args:
        obj: Hashable object.
        hash: Hash instance from :module:`hashlib`.

    Returns:
        Hex-string of digested object.

    Raises:
        DigestError: If `obj` cannot be hashed.
    """

    if hash is None:
        hash = hashlib.sha256()

    if obj is None:
        return hash

    if isinstance(obj, str):
        hash.update(bytes(obj.encode("utf-8")))
        return hash

    if isinstance(obj, int):
        hash.update(struct.pack("i", obj))
        return hash

    if isinstance(obj, float):
        hash.update(struct.pack("f", obj))
        return hash

    if isinstance(obj, complex):
        hash.update(struct.pack("f", obj.real))
        hash.update(struct.pack("f", obj.imag))
        return hash

    if isinstance(obj, Sequence):
        for x in obj:
            hash = digest(x, hash)
        return hash

    if isinstance(obj, Set):
        for x in sorted(obj):
            hash = digest(x, hash)
        return hash

    if isinstance(obj, Mapping):
        for x in sorted(obj):
            hash = digest(x, hash)
            hash = digest(obj[x], hash)
        return hash

    raise DigestError(f"cannot digest {obj!r}")
