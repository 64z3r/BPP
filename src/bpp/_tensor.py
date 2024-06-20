import torch
from torch import Tensor


def assure_2d(a: Tensor, /) -> Tensor:
    """Similar to :fun:`torch.atleast_2d` this will make sure that any array
    is at least two dimensional. However, unlike :func:`torch.atleast_2d` this
    will add an additional dimension to the rightmost side.

    **Note**: empty tensors will remain untouched.

    Arguments:
        a: An array-like object.

    Returns:
        Array with at least two dimensions.
    """

    if len(a) and a.dim() < 2:
        a = torch.atleast_2d(a).T

    return a
