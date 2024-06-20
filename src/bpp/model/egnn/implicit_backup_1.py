r"""Module containing implementations of Deep Equilibrium E(n) Equivariant Graph
Neural Networks.

References:
    .. [1] Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep Equilibrium Models. 
"""

from typing import TypeAlias, Any, Callable, Iterable, Sequence, Optional
from functools import partial
from math import prod

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch import autograd
from torch_geometric import nn as gnn
from torch_geometric.typing import Adj

from . import EGNN
from .._deterministic import deterministic_context, reproducible_context
from .._arguments import broadcast_arg, Broadcast
from .._solvers import broyden
from .._jacobian import jac_loss_estimate

Size: TypeAlias = tuple[int, ...]


def sizeof(tensors: Sequence[Tensor]) -> tuple[Size]:
    r"""Returns sizes of tensors.

    **Note**: first dimension is ignored.

    Arguments:
        tensors: List of tensors.

    Returns:
        List of sizes of tensors.
    """

    sizes = [x.shape[1:] for x in tensors]

    return sizes


def pack(tensors: Sequence[Tensor]) -> Tensor:
    r"""Pack list of tensors into vector.

    Arguments:
        tensors: List of tensors.

    Returns:
        Vector.
    """

    vec = torch.cat([x.flatten(1) for x in tensors], dim=1)

    return vec


def unpack(vec: Tensor, sizes: Sequence[Size]) -> list[Tensor]:
    r"""Unpack vector into list of tensors.

    Arguments:
        vec: Vector.
        sizes: List of sizes for tensors.

    Returns:
        List of tensors.
    """

    chunks = vec.split(tuple(map(prod, sizes)), dim=1)
    tensors = [x.reshape(-1, *s) for x, s in zip(chunks, sizes)]

    return tensors


class DeepImplicitEGNN(nn.Module):
    r""" """

    def __init__(
        self,
        dims_node: Sequence[int] = (64, 32),
        dims_edge: Sequence[int] = (64, 32, 64),
        num_pos: int = 1,
        dim_pos: int = 3,
        num_encode: int = 1,
        update: bool = True,
        gate: bool = False,
        norm: bool = False,
        clamp: float = 0.0,
        dropout: tuple[float, float] | float = 0.0,
        activation: Optional[nn.Module] = None,
        init: Optional[Callable[[nn.Module, str], None]] = None,
        fw_solver: None = None,
        bw_solver: None = None,
    ) -> None:
        r""" """

        if fw_solver is None:
            fw_solver = broyden
        if bw_solver is None:
            bw_solver = fw_solver

        super().__init__()

        self.egnn = EGNN(
            dims_node,
            dims_edge,
            num_pos,
            num_encode,
            update,
            gate,
            norm,
            clamp,
            dropout[1],
            activation,
            init,
        )

        self.norm_x_0 = nn.BatchNorm1d(dims_node[0])
        self.norm_x_1 = nn.BatchNorm1d(dims_node[0])
        self.norm_pos = nn.BatchNorm1d(dim_pos)


        self.fw_solver = fw_solver
        self.bw_solver = bw_solver
        self.hook = None

    def reset_parameters(self):
        r"""Reset parameters."""

        self.egnn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Adj] = None,
    ) -> tuple[Tensor | Tensor]:
        r"""Compute forward pass.

        Arguments:
            x: Node features.
            pos: Node coordinates.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            batch: Node to batch mapping.

        Returns:
            Tuple of updated node features and updated coordinates.
        """

        reproducible = reproducible_context(x, pos, edge_index, edge_attr, batch)

        # sizes = sizeof([x, pos])
        sizes = sizeof([x])

        def func(z):
            # x_, pos_ = unpack(z, sizes)

            # x_, pos_ = self.egnn(x_, pos_, edge_index, edge_attr, batch)

            # x_ += x
            # pos_ += pos

            # z = pack([x_, pos_]).unsqueeze(-1)

            x_ = unpack(z, sizes)[0]


            with reproducible():

                x_, _ = self.egnn(x_, pos, edge_index, edge_attr, batch)

                x_ = self.norm_x_0(x_)

                x_ += x

                x_ = self.norm_x_1(x_)

            z = pack([x_]).unsqueeze(-1)
            return z

        jac_loss = torch.tensor(0.0).to(x)

        # z1 = pack(map(torch.zeros_like, [x, pos])).unsqueeze(-1)
        z1 = pack(map(torch.zeros_like, [x])).unsqueeze(-1)

        with torch.no_grad():
            result = self.fw_solver(func, z1, threshold=100, stop_mode="rel")
            z1 = result["result"]

        z1_new = z1

        if self.training:
            z1_new = func(z1.requires_grad_())
            jac_loss = jac_loss_estimate(z1_new, z1)

            def backwards_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                with reproducible():
                    result = self.bw_solver(
                        lambda y: autograd.grad(z1_new, z1, y, retain_graph=True)[0] + grad,
                        torch.zeros_like(grad),
                        threshold=100,
                        stop_mode="rel",
                    )
                return result["result"]
            
            self.hook = z1_new.register_hook(backwards_hook)

        # x, pos = unpack(z1_new, sizes)
        x = unpack(z1_new, sizes)[0]

        return x, pos, jac_loss * .4