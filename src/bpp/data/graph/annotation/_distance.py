import numpy as np

from numba import jit
from numba.types import Array, float64
from numpy.typing import NDArray


@jit(
    Array(float64, 1, "C", readonly=True)(
        Array(float64, 2, "A", readonly=True),
        Array(float64, 2, "A", readonly=True),
    ),
    nopython=True,
    nogil=True,
)
def pairwise_min_distance(
    coords_a: NDArray[np.float64],
    coords_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Computes the minimum pairwise distances between `coords_a` and `coords_b`.

    Args:
        coords_a: Matrix containing coordinates.
        coords_b: Matrix containing coordinates.

    Returns:
        Vector containing minimum distances :math:`d_i` between `coords_a[i]`
        and `coords_b`.
    """

    d_min = np.full(len(coords_a), np.inf, dtype=np.float64)

    for j in range(len(coords_b)):
        d = np.sum((coords_a - coords_b[j]) ** 2, axis=-1)
        d_min = np.minimum(d_min, d)

    return np.sqrt(d_min)
