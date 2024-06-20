r"""Module containing implementations of Deep Equilibrium E(n) Equivariant Graph
Neural Networks.

References:
    .. [1] Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep Equilibrium Models. 
"""

# TODO: norm_cls and dropout_cls for within and after layers.

import gc
from typing import Any, Callable, Sequence, Optional

import torch
from torch import nn
from torch import Tensor
from torch_geometric.typing import Adj

from torchdeq import get_deq
from torchdeq.loss import jac_reg
from torchdeq.utils.layer_utils import deq_decorator

from . import iter_net_segments, init_net_segments, parameter_init
from .._deterministic import reproducible_context
from .unrecorded import UnrecordedEGNN


DEFAULT_DEQ_ARGS = {
    "core": "sliced",
    "ift": True,
    "hook_ift": False,
    "f_solver": "anderson",
    "f_max_iter": 75,
    "f_tol": 1e-3,
    "b_solver": "anderson",
    "b_max_iter": 100,
    "b_tol": 1e-6,
}

DEFAULT_SOLVER_ARGS = {
    "tau": 1.0,
    "m": 6,
    "lam": 1e-4,
}


class ImplicitEGNN(nn.Module):
    r""" """

    def __init__(
        self,
        dims_node: Sequence[int],
        dims_edge: Sequence[int],
        dims_gate: Sequence[int] = (),
        dims_pos: Sequence[int] = (),
        dims_inject: Sequence[int] = (),
        channels: int = 4,
        num_pos: int = 1,
        dim_pos: int = 3,
        num_encode: int = 1,
        pos_scale: Optional[float] = None,
        norm: tuple[bool, bool] | bool = False,
        clamp: float = 0.0,
        dropout: tuple[float, float] | float = 0.0,
        dropout_edge: float = 0.0,
        jacobi_penalty: float = 0.0,
        residual_penalty: float = 0.0,
        activation: Optional[nn.Module] = None,
        init: Optional[Callable[[nn.Module, str], None]] = None,
        norm_cls: type[nn.Module] = nn.LayerNorm,
        dropout_cls: type[nn.Module] = nn.Dropout,
        deq_args: dict[str, Any] = DEFAULT_DEQ_ARGS,
        solver_args: dict[str, Any] = DEFAULT_SOLVER_ARGS,
        unrecorded: bool = True,
    ) -> None:
        r""" """

        if not isinstance(dropout, Sequence):
            dropout = (dropout, dropout)

        if not isinstance(norm, Sequence):
            norm = (norm, norm)

        if init is None:
            init = parameter_init

        if activation is None:
            activation = nn.GELU(approximate="tanh")

        deq_args = dict(deq_args)
        solver_args = dict(solver_args)

        super().__init__()

        self.update_pos = 0 < len(dims_pos)
        self.jacobi_penalty = jacobi_penalty
        self.residual_penalty = residual_penalty
        self.activation = activation
        self.init = init
        self.deq = get_deq(**deq_args)
        self.deq_args = deq_args
        self.solver_args = solver_args

        self.egnn_1 = UnrecordedEGNN(
            (dims_node[-1], *dims_node[1:]),
            dims_edge,
            dims_gate,
            dims_pos,
            channels,
            num_pos,
            num_encode,
            pos_scale,
            norm[1],
            clamp,
            dropout,
            dropout_edge,
            activation,
            init,
            norm_cls,
            dropout_cls,
            disable=(not unrecorded),
        )

        self.egnn_2 = UnrecordedEGNN(
            (dims_node[-1], *dims_node[1:]),
            dims_edge,
            dims_gate,
            dims_pos,
            channels,
            num_pos,
            num_encode,
            pos_scale,
            norm[1],
            clamp,
            dropout,
            dropout_edge,
            activation,
            init,
            norm_cls,
            dropout_cls,
            disable=(not unrecorded),
        )

        self.inject = nn.Sequential(
            *iter_net_segments(
                dims_node[0],
                *dims_inject,
                dims_node[-1],
                norm=norm[1],
                dropout=dropout[1],
                activation=activation,
                norm_cls=norm_cls,
                dropout_cls=dropout_cls,
            ),
        )

        if norm[0]:
            self.x_norm_1 = norm_cls(dims_node[0])
            self.x_norm_2 = norm_cls(dims_node[-1])
            self.z_norm_1 = norm_cls(dims_node[-1])
            self.z_norm_2 = norm_cls(dims_node[-1])
            self.z_norm_3 = norm_cls(dims_node[-1])
            self.z_norm_4 = norm_cls(dims_node[-1])
            self.z_norm_5 = norm_cls(dims_node[-1])
            if self.update_pos:
                self.pos_norm_1 = nn.BatchNorm1d(dim_pos * num_pos)
                self.pos_norm_2 = nn.BatchNorm1d(dim_pos * num_pos)
                self.pos_norm_3 = nn.BatchNorm1d(dim_pos * num_pos)
                self.pos_norm_4 = nn.BatchNorm1d(dim_pos * num_pos)
            else:
                self.pos_norm_1 = nn.Identity()
                self.pos_norm_2 = nn.Identity()
                self.pos_norm_3 = nn.Identity()
                self.pos_norm_4 = nn.Identity()
        else:
            self.x_norm_1 = nn.Identity()
            self.x_norm_2 = nn.Identity()
            self.z_norm_1 = nn.Identity()
            self.z_norm_2 = nn.Identity()
            self.z_norm_3 = nn.Identity()
            self.z_norm_4 = nn.Identity()
            self.z_norm_5 = nn.Identity()
            self.pos_norm_1 = nn.Identity()
            self.pos_norm_2 = nn.Identity()
            self.pos_norm_3 = nn.Identity()
            self.pos_norm_4 = nn.Identity()

        if 0 < dropout[0]:
            self.x_drop_1 = dropout_cls(dropout[0])
            self.x_drop_2 = dropout_cls(dropout[0])
        else:
            self.x_drop_1 = nn.Identity()
            self.x_drop_2 = nn.Identity()

    def reset_parameters(self):
        r"""Reset parameters."""

        init_net_segments(self.init, self.inject, "linear")
        self.egnn_1.reset_parameters()
        self.egnn_2.reset_parameters()
        self.x_norm_1.apply(self.init)
        self.x_norm_2.apply(self.init)
        self.z_norm_1.apply(self.init)
        self.z_norm_2.apply(self.init)
        self.z_norm_3.apply(self.init)
        self.z_norm_4.apply(self.init)
        self.z_norm_5.apply(self.init)
        self.pos_norm_1.apply(self.init)
        self.pos_norm_2.apply(self.init)
        self.pos_norm_3.apply(self.init)
        self.pos_norm_4.apply(self.init)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Adj] = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, dict[str, Any]]:
        r"""Compute forward pass.

        Arguments:
            x: Node features.
            pos: Node coordinates.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            batch: Node to batch mapping.

        Returns:
            Updated node features, coordinates, optional penalty and info
            dictionary with statistics generated by the DEQ fixed point solver.
        """

        # Free unused memory.
        if self.training:
            gc.collect()
            torch.cuda.empty_cache()

        reproducible = reproducible_context(x, pos, edge_index, edge_attr, batch)

        # x = self.x_norm_1(x)
        x = self.x_drop_1(x)
        x = self.inject(x)
        x = self.x_norm_2(x)
        # x = self.x_drop_2(x)

        if self.update_pos:

            def func(z0, p0):

                with reproducible():

                    #     z1 = z0 + x
                    #     # z1 = self.activation(z1)
                    #     z1 = self.z_norm_1(z1)

                    #     p1 = (p0 + pos) * 0.5

                    #     z2, p2 = self.egnn_1(
                    #         z1,
                    #         p1,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )
                    #     z2 = self.z_norm_2(z2)

                    #     z3 = z0 + z2
                    #     # z3 = self.activation(z3)
                    #     z3 = self.z_norm_3(z3)

                    #     p3 = (p0 + p2) * 0.5

                    #     z4, p4 = self.egnn_2(
                    #         z3,
                    #         p3,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )
                    #     z4 = self.z_norm_4(z4)

                    #     z5 = z2 + z4
                    #     # z5 = self.activation(z5)
                    #     z5 = self.z_norm_5(z5)

                    #     p5 = (p2 + p4) * 0.5

                    # return z5, p5

                    #     z1 = self.z_norm_1(z1)
                    #     z2 = self.activation(z1)
                    #     p1 = self.pos_norm_1(p1)

                    #     z2, p2 = self.egnn_2(
                    #         z2,
                    #         p1,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )

                    #     z2 = z2 + x
                    #     z2 = self.z_norm_2(z2)

                    #     z3 = z1 + z2
                    #     z3 = self.activation(z3)
                    #     z3 = self.z_norm_3(z3)

                    #     p2 = (p2 + pos) * 0.5
                    #     p2 = self.pos_norm_2(p2)

                    #     p3 = (p1 + p2) * 0.5
                    #     p3 = self.pos_norm_3(p3)

                    # return z3, p3

                    z1, p1 = self.egnn_1(
                        z0,
                        p0,
                        edge_index,
                        edge_attr,
                        batch,
                    )
                    z1 = self.z_norm_1(z1)
                    p1 = self.pos_norm_1(p1.flatten(x.dim() - 1)).view(p1.shape)

                    z2 = z1 + x
                    z2 = self.activation(z2)
                    z2 = self.z_norm_2(z2)

                    # p2 = (p1 + pos) * 0.5
                    p2 = pos - p0 + p1
                    p2 = self.pos_norm_2(p2.flatten(x.dim() - 1)).view(p2.shape)

                    z3, p3 = self.egnn_2(
                        z2,
                        p2,
                        edge_index,
                        edge_attr,
                        batch,
                    )
                    z3 = self.z_norm_3(z3)
                    p3 = self.pos_norm_3(p3.flatten(x.dim() - 1)).view(p3.shape)

                    z4 = z3 + z0
                    z4 = self.activation(z4)
                    z4 = self.z_norm_4(z4)

                    # p4 = (p3 + p0) * 0.5
                    p4 = pos - p0 + p3
                    p4 = self.pos_norm_4(p4.flatten(x.dim() - 1)).view(p4.shape)

                return z4, p4

                #     z1, p1 = self.egnn_1(
                #         z0,
                #         pos,
                #         edge_index,
                #         edge_attr,
                #         batch,
                #     )
                #     z1 = self.z_norm_1(z1)
                #     p1 = self.pos_norm_1(p1)

                #     z2 = z1 + x
                #     z2 = self.activation(z2)
                #     z2 = self.z_norm_2(z2)

                #     p2 = p0 + p1
                #     p2 = self.pos_norm_2(p2)

                #     z3, p3 = self.egnn_2(
                #         z2,
                #         p2,
                #         edge_index,
                #         edge_attr,
                #         batch,
                #     )
                #     z3 = self.z_norm_3(z3)
                #     p3 = self.pos_norm_3(p3)

                #     z4 = z3 + z0
                #     z4 = self.activation(z4)
                #     z4 = self.z_norm_4(z4)

                #     p4 = p3 - p1

                # return z4, p4

            z = list(map(torch.zeros_like, [x, pos]))

        else:

            def func(z0):

                with reproducible():

                    #     z1, _ = self.egnn_1(
                    #         z0,
                    #         pos,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )

                    #     z1 = self.z_norm_1(z1)
                    #     z2 = self.activation(z1)

                    #     z2, _ = self.egnn_2(
                    #         z2,
                    #         pos,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )

                    #     z2 = z2 + x
                    #     z2 = self.z_norm_2(z2)

                    #     z3 = z1 + z2
                    #     z3 = self.activation(z3)
                    #     z3 = self.z_norm_3(z3)

                    # return z3

                    #     z1 = z0 + x
                    #     z1 = self.activation(z1)
                    #     z1 = self.z_norm_1(z1)

                    #     z2, _ = self.egnn_1(
                    #         z1,
                    #         pos,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )
                    #     z2 = self.z_norm_2(z2)

                    #     z3 = z0 + z2
                    #     z3 = self.activation(z3)
                    #     z3 = self.z_norm_3(z3)

                    #     z4, _ = self.egnn_2(
                    #         z3,
                    #         pos,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )
                    #     z4 = self.z_norm_4(z4)

                    #     z5 = z2 + z4
                    #     z5 = self.activation(z5)
                    #     z5 = self.z_norm_5(z5)

                    # return z5

                    #     z1, _ = self.egnn_1(
                    #         z0,
                    #         pos,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )
                    #     z1 = self.z_norm_1(z1)

                    #     z2 = z0 + z1
                    #     z2 = self.activation(z2)
                    #     z2 = self.z_norm_2(z2)

                    #     z3, _ = self.egnn_2(
                    #         z2,
                    #         pos,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )
                    #     z3 = self.z_norm_3(z3)

                    #     z4 = z1 + z3
                    #     z4 = self.activation(z4)
                    #     z4 = self.z_norm_4(z4)

                    #     z5 = z4 + x
                    #     z5 = self.activation(z5)
                    #     z5 = self.z_norm_5(z5)

                    # return z5

                    z1, _ = self.egnn_1(
                        z0,
                        pos,
                        edge_index,
                        edge_attr,
                        batch,
                    )
                    z1 = self.z_norm_1(z1)

                    z2 = z1 + x
                    z2 = self.activation(z2)
                    z2 = self.z_norm_2(z2)

                    z3, _ = self.egnn_2(
                        z2,
                        pos,
                        edge_index,
                        edge_attr,
                        batch,
                    )
                    z3 = self.z_norm_3(z3)

                    z4 = z3 + z0
                    z4 = self.activation(z4)
                    z4 = self.z_norm_4(z4)

                return z4

            z = torch.zeros_like(x)

        compute_sradius = not (self.training or torch.is_inference_mode_enabled())

        z_result, info = self.deq(
            func,
            z,
            solver_kwargs=self.solver_args,
            sradius_mode=compute_sradius,
            backward_writer=None,
        )

        if self.update_pos:
            z, pos = z_result[-1]
        else:
            z = z_result[-1]

        penalty = 0

        if self.training:

            if self.jacobi_penalty > 0 or self.residual_penalty > 0:
                f, z0 = deq_decorator(func, z_result[-1], no_stat=None)
                z1 = f(z0)

            if self.jacobi_penalty > 0:
                p = jac_reg(z1, z0, vecs=3)
                penalty += p * self.jacobi_penalty

            if self.residual_penalty > 0:
                p = ((z1 - z0) ** 2).mean()
                penalty += p * self.residual_penalty

        if penalty == 0:
            penalty = None

        if not self.training:
            print(
                f"*** {info['sradius'].min().item():13.6e} {info['sradius'].max().item():13.6e}"
            )

        return z, pos, penalty, info
