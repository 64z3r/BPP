r"""Module containing implementations of Deep Equilibrium E(n) Equivariant Graph
Neural Networks.

References:
    .. [1] Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep Equilibrium Models. 
"""

from typing import TypeAlias, Any, Callable, Iterable, Sequence, Optional
from functools import partial
from math import prod

import torch
from torch import nn
from torch import Tensor
from torch_geometric import nn as gnn
from torch_geometric.typing import Adj

from . import EGNN
from .._deterministic import deterministic_context, reproducible_context
from .._arguments import broadcast_arg, Broadcast
from .._solvers import broyden, anderson
from .._jacobian import jac_loss_estimate

from torchdeq import get_deq
from torchdeq.core import DEQBase, DEQSliced
from torchdeq.loss import jac_reg
from torchdeq.utils.layer_utils import deq_decorator

Size: TypeAlias = tuple[int, ...]


from .unrecorded import UnrecordedEGNN


def parameter_init(module: nn.Module, nonlinearity: str = "relu") -> None:
    r"""Initialize module parameters. Uses kaiming initialization for linear
    modules and sets parameters of other modules accordingly.

    Arguments:
        module:
            Module that parameters should be initialized.
        nonlinearity:
            Nonlinearity that should be assumed for linear modules.
    """

    match module:
        case nn.Linear():
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
            # module.weight.data.set_(module.weight.data / 10)
            # nn.init.normal_(module.weight, 0.0, 0.05)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            # nn.utils.parametrizations.weight_norm(module, name="weight", dim=0)
            # nn.utils.parametrizations.spectral_norm(module, name="weight", dim=1)
        case nn.LayerNorm():
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        case nn.BatchNorm1d():
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        case gnn.norm.MessageNorm():
            nn.init.ones_(module.scale)


# class ImplicitEGNNCell(nn.Module):
#     r""" """

#     def __init__(self) -> None:
#         r""" """


#     def forward(self, x, pos) ->


class ImplicitEGNN(nn.Module):
    r""" """

    def __init__(
        self,
        dims_node: Sequence[int] = (256, 64, 64),
        dims_edge: Sequence[int] = (256, 128, 32),
        dims_gate: Sequence[int] = (),
        dims_pos: Sequence[int] = (),
        channels: int = 4,
        num_pos: int = 1,
        dim_pos: int = 3,
        num_encode: int = 1,
        pos_scale: float = 1.0,
        norm: bool = False,
        clamp: float = 0.0,
        dropout: tuple[float, float] | float = 0.0,
        jacobi_penalty: float = 0.0,
        fixed_point_penalty: float = 0.0,
        residual: bool = True,
        activation: Optional[nn.Module] = None,
        init: Optional[Callable[[nn.Module, str], None]] = None,
        norm_cls: type[nn.Module] = nn.LayerNorm,
        dropout_cls: type[nn.Module] = nn.Dropout,
        deq: Optional[DEQBase] = None,
    ) -> None:
        r""" """

        if not isinstance(dropout, Sequence):
            dropout = (dropout, dropout)

        if init is None:
            init = parameter_init

        if activation is None:
            activation = nn.GELU(approximate="tanh")

        if deq is None:
            deq = deq = get_deq(
                core="sliced",
                # n_states=5,
                ift=True,
                hook_ift=False,
                f_solver="anderson",
                # f_solver="broyden",
                f_max_iter=75,
                f_tol=1e-3,
                b_solver="anderson",
                # b_solver="broyden",
                b_max_iter=100,
                b_tol=1e-6,
            )

            # deq = DEQSliced(
            #     f_solver="fixed_point_iter",
            #     b_solver="fixed_point_iter",
            #     # f_solver="broyden",
            #     # b_solver="broyden",
            #     # f_solver="anderson",
            #     # b_solver="anderson",
            #     # no_stat=None,
            #     f_max_iter=100,
            #     b_max_iter=100,
            #     f_tol=1e-3,
            #     b_tol=1e-6,
            #     f_stop_mode="rel",
            #     b_stop_mode="rel",
            #     eval_factor=1.0,
            #     eval_f_max_iter=0,
            #     ift=False,
            #     hook_ift=False,
            #     grad=10,
            #     # tau=1.0,
            #     # sup_gap=-1,
            #     # sub_loc=None,
            #     # n_states=1,
            #     # indexing=None,
            # )

        super().__init__()

        self.dims_node = dims_node
        self.dims_edge = dims_edge
        self.num_pos = num_pos
        self.dim_pos = dim_pos
        self.num_encode = num_encode
        self.norm = norm
        self.clamp = clamp
        self.dropout = dropout
        self.jacobi_penalty = jacobi_penalty
        self.fixed_point_penalty = fixed_point_penalty
        self.residual = residual
        self.activation = activation
        self.init = init
        self.deq = deq
        
        
        self.node_mix_in = nn.Linear(
            dims_node[0] * 2,
            dims_node[-1],
            bias=False,
        )

        # self.node_mix_in = nn.Linear(
        #     dims_node[0] * 2,
        #     dims_node[1] * channels,
        #     bias=False,
        # )
        # self.node_mix_out = nn.Linear(
        #     dims_node[-1] * channels,
        #     dims_node[0],
        # )

        # if update:
        #     self.pos_mix = nn.Linear(dim_pos * 2, dim_pos, bias=False)
        # else:
        #     self.pos_mix = None

        self.channels = nn.ModuleList()

        # TODO: node dims with different in/out dims... due to inject net.

        # for _ in range(channels):
        # for _ in range(2):
        # channel = nn.ModuleDict()
        # channel.egnn = EGNN(
        #     dims_node,  # [1:],
        #     dims_edge,
        #     num_pos,
        #     num_encode,
        #     update,
        #     gate,
        #     norm,
        #     clamp,
        #     dropout[1],
        #     activation,
        #     init,
        # )
        # if 0 < dropout[0]:
        #     channel.drop = nn.Dropout(dropout[0])
        # else:
        #     channel.drop = nn.Identity()
        # self.channels.append(channel)

        # channel = nn.ModuleDict()
        # channel.egnn = EGNN(
        #     dims_node,  # [1:],
        #     dims_edge,
        #     num_pos,
        #     num_encode,
        #     update,
        #     gate,
        #     norm,
        #     clamp,
        #     dropout[1],
        #     activation,
        #     init,
        # )
        # if 0 < dropout[0]:
        #     channel.drop = nn.Dropout(dropout[0])
        # else:
        #     channel.drop = nn.Identity()
        # self.channels.append(channel)

        channel = nn.ModuleDict()
        channel.egnn = UnrecordedEGNN(
            dims_node,
            dims_edge,
            dims_gate,
            dims_pos,
            channels,
            num_pos,
            num_encode,
            pos_scale,
            norm,
            clamp,
            dropout,
            activation,
            init,
            # activation=nn.Identity(),
            # init=lambda module, nonlinearity="linear": init(module, nonlinearity),
            norm_cls,
            dropout_cls,
            disable=False,
        )
        self.channels.append(channel)

        channel = nn.ModuleDict()
        channel.egnn = UnrecordedEGNN(
            dims_node,
            dims_edge,
            dims_gate,
            dims_pos,
            channels,
            num_pos,
            num_encode,
            pos_scale,
            norm,
            clamp,
            dropout,
            activation,
            init,
            # activation=nn.Identity(),
            # init=lambda module, nonlinearity="linear": init(module, nonlinearity),
            norm_cls,
            dropout_cls,
            disable=False,
        )
        # if 0 < dropout[0]:
        #     channel.drop = nn.Dropout(dropout[0])
        # else:
        #     channel.drop = nn.Identity()
        self.channels.append(channel)

        self.inject = nn.Sequential(
            norm_cls(dims_node[0]),
            dropout_cls(dropout[1]),
            *EGNN._iter_net_segments(
                dims_node[0],
                *dims_node[2:-2],
                dims_node[-1],
                dropout=dropout[1],
                activation=activation,
            ),
            norm_cls(dims_node[-1]),
            dropout_cls(dropout[1]),
        )
        # self.residual = nn.Sequential(
        #     nn.LayerNorm(dims_node[0]),
        #     *EGNN._iter_net_segments(
        #         dims_node[0],
        #         *dims_node[2:-2],
        #         dims_node[-1],
        #         dropout=dropout[1],
        #         activation=activation,
        #     ),
        #     # nn.LayerNorm(dims_node[-1]),
        # )

        # self.norm1 = nn.LayerNorm(dims_node[0])
        self.norm2 = norm_cls(dims_node[0])
        self.norm3 = norm_cls(dims_node[0])
        self.norm4 = norm_cls(dims_node[0])
        self.pos_norm_1 = nn.BatchNorm1d(dim_pos * num_pos)
        self.pos_norm_2 = nn.BatchNorm1d(dim_pos * num_pos)

        # self.f = nn.Sequential(
        #     nn.Linear(dims_node[0], dims_node[1]),
        #     nn.GELU(),
        #     nn.Linear(dims_node[1], dims_node[0]),
        # )

    def reset_parameters(self):
        r"""Reset parameters."""

        for channel in self.channels:
            channel.egnn.reset_parameters()

        self.inject.apply(self.init)

        # self.f.apply(self.init)

        # nn.init.kaiming_uniform_(self.node_mix_in.weight, nonlinearity="linear")
        # nn.utils.parametrizations.spectral_norm(self.node_mix_in, name="weight", dim=0)

        # nn.init.kaiming_uniform_(self.node_mix_out.weight, nonlinearity="linear")
        # nn.utils.parametrizations.spectral_norm(self.node_mix_out, name="weight", dim=0)

        # if self.update:
        #     nn.init.kaiming_uniform_(self.pos_mix.weight, nonlinearity="linear")
        #     nn.utils.parametrizations.spectral_norm(self.pos_mix, name="weight", dim=0)

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

        # num_channels = len(self.channels)

        # x = self.norm1(x)
        # x = self.f(x)
        # x = self.norm2(x)

        # z2, _ = self.channels[1].egnn(x, pos, edge_index, edge_attr, batch)

        # Bernoulli = torch.distributions.Bernoulli

        # p = 1 - self.dropout[0]
        # mask = Bernoulli(probs=p).sample(x.size()).to(x)
        # mask /= mask.mean()

        x = self.inject(x)
        # r = self.residual(x)

        if self.update:

            def deq_func(x_0, pos_0):

                with reproducible():

                    #         x_in = x_0

                    #         x_0 = self.node_mix_in(torch.cat([x_0, x], dim=-1))
                    #         pos_0 = self.pos_mix(torch.cat([pos_0, pos], dim=-1))

                    #         u_chunks = torch.chunk(x_0, num_channels, dim=-1)
                    #         v_chunks = []
                    #         pos_s = 0

                    #         for u_i, channel in zip(u_chunks, self.channels):
                    #             v_i, pos_i = channel.egnn(
                    #                 u_i,
                    #                 pos_0,
                    #                 edge_index,
                    #                 edge_attr,
                    #                 batch,
                    #             )
                    #             v_i = channel.drop(v_i)
                    #             v_chunks.append(v_i)
                    #             pos_s = pos_s + pos_i

                    #         v_0 = torch.cat(v_chunks, dim=-1)
                    #         x_0 = self.node_mix_out(v_0)

                    #         # if self.residual:
                    #         #     x_0 = x_0 + x_in

                    #         x_0 = self.norm(x_0)
                    #         pos_0 = pos_s / num_channels

                    # z0, p0 = self.channels[0].egnn(x_0, pos, edge_index, edge_attr, batch)
                    # z1 = self.norm3(z0 + x)
                    # p1 = self.pos_norm_1(p0)
                    # z2, p2 = self.channels[1].egnn(x, p1, edge_index, edge_attr, batch)
                    # # z3 = self.norm4(nn.functional.gelu(z0 + z2))
                    # z3 = nn.functional.gelu(z0 + z2)
                    # p3 = self.pos_norm_2(p0 + p2)
                    # # p3 = p0 + p2

                    z0, p0 = self.channels[0].egnn(
                        x_0, pos, edge_index, edge_attr, batch
                    )
                    z1 = self.norm3(z0 + x)
                    p1 = self.pos_norm_1(p0 + pos_0)
                    z2, p2 = self.channels[1].egnn(
                        self.channels[1].drop(nn.functional.gelu(z1)),
                        p1,
                        edge_index,
                        edge_attr,
                        batch,
                    )
                    z3 = self.norm4(nn.functional.gelu(z0 + z2))
                    # z3 = nn.functional.gelu(z0 + z2)
                    p3 = self.pos_norm_2(p0 + p2)
                    # p3 = p0 + p2

                    # z0, p0 = self.channels[0].egnn(x_0, pos_0, edge_index, edge_attr, batch)
                    # z1 = self.norm3(z0 + x)
                    # p1 = self.pos_norm_1(p0 + pos)
                    # z2, p2 = self.channels[1].egnn(self.channels[1].drop(nn.functional.gelu(z1)), p1, edge_index, edge_attr, batch)
                    # # z3 = self.norm4(nn.functional.gelu(z0 + z2))
                    # z3 = nn.functional.gelu(z0 + z2)
                    # p3 = self.pos_norm_2(p0 + p2)
                    # # p3 = p0 + p2

                    x_0 = z3
                    pos_0 = p3

                return x_0, pos_0

            z_0 = list(map(torch.zeros_like, [x, pos]))

        else:

            def deq_func(x_0):

                with reproducible():

                    # x_in = x_0

                    # # x_0 = self.node_mix_in(torch.cat([x_0, x], dim=-1))
                    # # x_0 = self.norm3(x_0)
                    # x_0 = self.node_mix_in(torch.cat([x_0, x], dim=-1))
                    # # x_0 = x_0 + x

                    # # x_0 = self.norm4(x_0)

                    # u_chunks = torch.chunk(x_0, num_channels, dim=-1)
                    # v_chunks = []

                    # for u_i, channel in zip(u_chunks, self.channels):
                    #     v_i, _ = channel.egnn(
                    #         u_i,
                    #         pos,
                    #         edge_index,
                    #         edge_attr,
                    #         batch,
                    #     )
                    #     v_i = channel.drop(v_i)
                    #     v_chunks.append(v_i)

                    # v_0 = torch.cat(v_chunks, dim=-1)
                    # x_0 = self.node_mix_out(v_0)
                    # # x_0 = v_0

                    # # if self.residual:
                    # #     x_0 = x_0 + x_in

                    # # x_0 = self.f(x_0)

                    # # x_0 = self.norm3(x_0)

                    # # print(f"(0) {x_0.mean()=}")
                    # # print(f"(0) {x_0.std()=}")

                    # # x_0 = self.norm4(x_0)
                    # z0, _ = self.channels[0].egnn(
                    #     x_0,
                    #     pos,
                    #     edge_index,
                    #     edge_attr,
                    #     batch,
                    # )
                    # z1 = self.norm2(z0)
                    # z1 = self.norm3(z0 + x)
                    # z1 = nn.functional.gelu(z1)
                    # z1 = self.channels[1].drop(z1)
                    # # z1 = z1 * mask
                    # z2, _ = self.channels[1].egnn(
                    #     z1,
                    #     pos,
                    #     edge_index,
                    #     edge_attr,
                    #     batch,
                    # )
                    # z3 = self.norm4(nn.functional.gelu(z0 + z2))
                    # # z3 = nn.functional.gelu(z0 + z2)

                    # x_0 = z3

                    z0, _ = self.channels[0].egnn(
                        x_0,
                        pos,
                        edge_index,
                        edge_attr,
                        batch,
                    )
                    z0 = self.norm2(z0)
                    z1 = self.activation(z0)
                    z1, _ = self.channels[1].egnn(
                        z1,
                        pos,
                        edge_index,
                        edge_attr,
                        batch,
                    )
                    z1 = z1 + x
                    z1 = self.norm3(z1)
                    z2 = z0 + z1
                    z2 = self.activation(z2)
                    z2 = self.norm4(z2)

                    x_0 = z2

                    # # z0 = self.node_mix_in(torch.cat([x_0, x], dim=-1))
                    # z0 = x_0 + x
                    # z0 = self.norm2(z0)
                    # z0, _ = self.channels[0].egnn(
                    #     z0,
                    #     pos,
                    #     edge_index,
                    #     edge_attr,
                    #     batch,
                    # )
                    # z0 = self.norm3(z0)
                    # z0 = self.channels[0].drop(z0)
                    # z3 = self.norm4(z0 + x_0)

                    # x_0 = z3

                    # x_1, _ = self.egnn(x_0 + x, pos, edge_index, edge_attr, batch)

                    # x_1 = self.f(x_0 + x)

                return x_0

            z_0 = torch.zeros_like(x)

        # for name, parameter in self.f.named_parameters():
        #     print(f"{name}: {parameter.size()}")

        # z_0 = deq_func(z_0)
        # z_0 = z_0.detach()  # TODO: check if this works?

        # if self.training:
        #     print(f"(0) {self.channels[0].egnn.node_net[0].weight.grad=}")

        #     try:
        #         z_res, info = self.deq(
        #             deq_func,
        #             z_0,
        #             solver_kwargs={"tau": 1.0, "m": 6, "lam": 1e-4},
        #             sradius_mode=True,
        #             backward_writer=None,
        #         )
        #     except:
        #         z_res, info = self.deq(
        #             deq_func,
        #             z_0,
        #             solver_kwargs={"tau": 1.0, "m": 6, "lam": 1e-4},
        #             sradius_mode=False,
        #             backward_writer=None,
        #         )
        #     z_0 = z_res[-1]

        #     print(f"{z_res=}")
        #     print(f"{z_res[-1].grad_fn=}")

        #     # l = (z_0**2).mean()
        #     # l.backward()

        #     # print(f"(0) {self.channels[0].egnn.node_net[0].weight.grad=}")

        #     # assert False
        # else:
        compute_sradius = not (self.training or torch.is_inference_mode_enabled())
        z_res, info = self.deq(
            deq_func,
            z_0,
            solver_kwargs={"tau": 1.0, "m": 6, "lam": 1e-4},
            sradius_mode=compute_sradius,
            backward_writer=None,
        )
        z_0 = z_res[-1]

        # if self.training:

        #     from torchviz import make_dot

        #     # # with torch.set_grad_enabled(True):
        #     # #     # make_dot(deq_func(x), params=dict(self.f.named_parameters())).render("attached1", format="pdf")
        #     # #     make_dot(deq_func(x)).render("attached3", format="pdf")
        #     # # assert False

        #     # z_star = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        #     # print(f"(0) {z_star.requires_grad=}")
        #     # # z_star = torch.zeros_like(x)
        #     # # with torch.set_grad_enabled(True):
        #     # #     z_star = deq_func(z_star)
        #     # # with torch.no_grad():
        #     # #     # for _ in range(10):
        #     # #         z_star = deq_func(z_star)

        #     # # z_star = z_star.detach()
        #     # # z_star.requires_grad_()
        #     # # with torch.set_grad_enabled(True):
        #     # # new_z_star = deq_func(z_star)
        #     # new_z_star = self.f(z_star)
        #     # print(f"(1) {z_star.requires_grad=}")

        #     make_dot(z_0, params=dict(self.named_parameters())).render("attached6", format="pdf")

        #     assert False

        # return x, pos

        # # for name, parameter in self.f.named_parameters():
        # #     print(f"{name}: {parameter.requires_grad}")

        # for name, parameter in self.f.named_parameters():
        #     print(f"{name}: {parameter.requires_grad}")

        if self.update:
            x, pos = z_0
        else:
            x = z_0

        penalty = 0

        if self.training:

            if self.jacobi_penalty > 0 or self.fixed_point_penalty > 0:
                deq_func_, z_0_ = deq_decorator(deq_func, z_0, no_stat=None)
                z_next = deq_func_(z_0_)

            if self.jacobi_penalty > 0:
                p = jac_reg(z_next, z_0_, vecs=3)
                penalty += p * self.jacobi_penalty

            if self.fixed_point_penalty > 0:
                p = ((z_next - z_0_) ** 2).mean()
                penalty += p * self.fixed_point_penalty

        if penalty == 0:
            penalty = None

        if not self.training:
            print("***", info["sradius"].min(), info["sradius"].max())

        return x, pos, penalty, info

        # #     s_ = self.mixer_x(torch.cat([x_, x], dim=-1))
        # #     pos_ = self.mixer_pos(torch.cat([pos_, pos], dim=-1))

        # #     s_, pos_ = self.egnn(s_, pos_, edge_index, edge_attr, batch)
        # #     # s_ = self.norm_x_0(s_)
        # #     s_ += x_
        # #     # s_ = self.norm_x_1(s_)
        # #     # pos_ = self.norm_pos_0(pos_)
        # #     # pos_ += pos
        # #     # pos_ = self.norm_pos_1(pos_)

        # # return s_, pos_

        # z_now = list(map(torch.zeros_like, [x, pos]))

        # z_out, info = self.deq(
        #     func,
        #     z_now,
        #     # solver_kwargs={"LBFGS_thres": 8},
        #     sradius_mode=False,
        #     backward_writer=None,
        # )
        # z_now = z_out[-1]

        # # if self.training:
        # #     func_, z_now_ = deq_decorator(func, z_now, no_stat=None)
        # #     jac_loss = jac_reg(func_(z_now_), z_now_) * 0.1
        # # else:
        # #     jac_loss = None

        # jac_loss = None

        # x, pos = z_now

        # return x, pos, jac_loss
