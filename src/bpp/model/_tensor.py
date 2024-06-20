import logging

from typing import Any, Sequence, Iterable
from math import prod
from torch import Tensor
from torch_geometric.typing import WITH_PT20

logger = logging.getLogger(__name__)


if WITH_PT20:
    from torch.storage import UntypedStorage

    def storage_of(element: Tensor) -> UntypedStorage:
        """ """

        return element.untyped_storage()

    def size_of(element: Tensor) -> int:
        """ """

        return prod(element.size()) * element.element_size()

else:

    def storage_of(element: Tensor) -> Any:
        """ """

        return element.storage()

    def size_of(element: Tensor) -> int:
        """ """

        return prod(element.size())


def free(element: Tensor) -> None:
    """ """

    try:
        storage_of(element).resize_(0)
    except RuntimeError:
        logger.warning(f"Could not free tensor:\n{element}")


def allocate(element: Tensor) -> int:
    """ """

    storage_of(element).resize_(size_of(element))


def detached(elements: Sequence[Any]) -> Iterable[Any]:
    """ """

    for element in elements:
        if isinstance(element, Tensor):
            yield element.detach()
        else:
            yield element


def tensors_of(
    elements: Sequence[Any],
    *,
    without: Sequence[Any] = (),
    with_grads_only: bool = False,
) -> Iterable[Tensor]:
    """ """

    if without:
        without = tuple(storage_of(x).data_ptr() for x in tensors_of(without))

    for element in elements:
        if not isinstance(element, Tensor):
            continue
        if without and storage_of(element).data_ptr() in without:
            continue
        if with_grads_only and not element.requires_grad:
            continue
        yield element


def unsqueeze_grads(
    elements: Sequence[Tensor],
    placement: Sequence[bool],
) -> Iterable[Tensor | None]:
    """ """

    elements = iter(elements)

    for here in placement:
        if not here:
            yield None
        else:
            yield next(elements)
