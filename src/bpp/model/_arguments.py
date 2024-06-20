from typing import TypeAlias, TypeVar, Any, Sequence, Iterable, Optional
from functools import partial
from torch import Tensor


def pack_args(*args: Sequence[Any]) -> tuple[list[Any], list[slice]]:
    """ """

    packed = []
    slices = []
    offset = 0

    for part in args:
        packed.extend(part)
        slices.append(slice(offset, offset + len(part)))
        offset += len(part)

    return packed, slices


def split_args(args: Sequence[Any]) -> tuple[list[Tensor], list[tuple[int, Any]]]:
    """ """

    tensors = []
    others = []

    for i, element in enumerate(args):
        if isinstance(element, Tensor):
            tensors.append(element)
        else:
            others.append((i, element))

    return tensors, others


def restore_args(
    tensors: Sequence[Tensor],
    extras: Sequence[tuple[int, Any]],
) -> Iterable[Any]:
    """ """

    num_args = len(tensors) + len(extras)
    tensors = iter(tensors)
    extras = iter(extras)

    try:
        j, element = next(extras)
        for i in range(num_args):
            if i != j:
                yield next(tensors)
            else:
                yield element
                j, element = next(extras)
    except StopIteration:
        yield from tensors


T = TypeVar("T")
Nested: TypeAlias = Sequence[T | "Nested"]
Broadcast: TypeAlias = Sequence[T] | T


def broadcast_arg(
    name: str,
    value: Nested[T] | T,
    shape: Sequence[Optional[int]],
) -> Nested[T]:
    """Broadcasts sequences or single values to a desired shape of nested
    lists.

    This implementation recursively matches `value` (i.e., its innermost
    contents) with the right-hand side of `shape` and broadcasts all matched
    values to the same shape -- a totally intuitive solution that isn't at
    all an overkill for what is actually needed. (☉_☉ )

    `None` entries in `shape` correspond to dimensions that are kept as they
    are.

    Arguments:
        name:  Name of the argument. Needed for error messages.
        value: Value that needs to be broadcasted.
        shape: Desired shape that the value should have.

    Returns:
        Reshaped value with desired shape.

    Raises:
        ValueError: If shapes at a given level do not match or if the value
            is deeper than what the desired shape asks for.
    """

    def pad(shape, value):
        for size in reversed(shape):
            if size is None:
                size = 1
            value = [value] * size
        return value

    def rec(shape, value):
        if not isinstance(value, Sequence):
            return shape, value

        if len(value) == 0:
            shapes, prepared = [shape], value
        else:
            shapes, prepared = zip(*map(partial(rec, shape), value))
        level = min(map(len, shapes))

        reshaped = []
        for sha, val in zip(shapes, prepared):
            reshaped.append(pad(sha[level:], val))

        try:
            *shape, size = shape[:level]
        except ValueError as error:
            raise ValueError(
                f"argument '{name}' received value that is deeper "
                f"than what the desired shape asks for"
            ) from error

        if size is None:
            return shape, reshaped

        if len(reshaped) == 1:
            return shape, reshaped * size

        if len(reshaped) == size:
            return shape, reshaped

        raise ValueError(
            f"argument '{name}' expected {size} elements at level "
            f"{level-1}, but got {len(reshaped)} instead"
        )

    broadcasted = pad(*rec(shape, value))

    return broadcasted
