r"""Module containing implementations of E(n) Equivariant Graph Neural Networks.

References:
    .. [1] Victor Garcia Satorras, , Emiel Hoogeboom, and Max Welling. "E(n)
        Equivariant Graph Neural Networks." (2022).
        `arXiv:2102.09844<https://arxiv.org/abs/2102.09844>`.
"""

from typing import Any, Callable, Iterable, Sequence, Optional
from functools import partial

import torch
from torch import nn
from torch import Tensor
from torch_geometric import nn as gnn
from torch_geometric.typing import Adj
from torch_geometric.utils import dropout_edge

from .._arguments import broadcast_arg, Broadcast


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
            if module.bias is not None:
                nn.init.zeros_(module.bias)
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


def iter_net_segments(
    *dims: int,
    norm: bool = False,
    clamp: float = 0.0,
    dropout: float = 0.0,
    activation: Optional[nn.Module] = None,
    norm_cls: type[nn.Module] = nn.BatchNorm1d,
    dropout_cls: type[nn.Module] = nn.Dropout,
) -> Iterable[nn.Module]:
    r"""Initializes and yields individual segments for feed-forward networks.

    Arguments:
        dims: Dimensions for input, hidden and output features.
        norm: Whether to layer-normalize inputs.
        clamp: Threshold for clamping the output (between `-clamp` and
            `clamp`).  Unclamped if `clamp <= 0`.
        dropout: Dropout probability in-between layers of network. No
            dropout if `0`.
        activation: Activation function that is used in-between network
            layers.
        norm_cls: Normalization module class instance.
        dropout_cls: Dropout module class instance.

    Yields:
        Individual segments of network.
    """

    # if 1 <= len(dims) and norm:
    #     yield norm_cls(dims[0])

    if 0 < dropout:
        yield dropout_cls(dropout)

    if 2 <= len(dims):
        yield nn.Linear(dims[0], dims[1])

    for k, (i, j) in enumerate(zip(dims[1:-1], dims[2:])):
        if activation is not None:
            yield activation
        if norm and k & 1:
            yield norm_cls(j)
        yield nn.Linear(i, j)

    if 0 < clamp:
        yield Apply(partial(torch.clamp, min=-clamp, max=clamp))


def init_net_segments(
    init: Optional[Callable[[nn.Module, str], None]] = None,
    sequence: Optional[Sequence] = None,
    head_nonlinearity: Optional[str] = None,
) -> None:
    r"""Initialize sequence of network segments.

    Arguments:
        init: Initializer that is used for network layers. First argument is
            the module that should be initialized, the second is the
            non-linearity, in case of a linear module, that should be used.
            Default non-linearity must be set.
        sequence: Sequence of network segments.
        head_nonlinearity: Non-linearity for head of network.
    """

    if sequence is None:
        return

    if head_nonlinearity is None:
        sequence.apply(init)
        return

    for i in reversed(range(len(sequence))):
        if isinstance(sequence[i], nn.Linear):
            head = sequence[i:]
            body = sequence[:i]
            head.apply(lambda module: init(module, head_nonlinearity))
            body.apply(init)
            break
    else:
        head = sequence
        head.apply(lambda module: init(module, head_nonlinearity))


def fourier_encode(x: Tensor, num: int) -> Tensor:
    r"""Encodes `x` to be represented by a vector of fourier features.

    Depending on whether `num` is even or odd, the vector of encoded features
    will also contain `x` itself: if even, `x` is not included, if odd `x` is
    included.

    Assuming `num` is odd, the features :math:`y` look as follows:

    .. math::

        \begin{aligned}
        s_k &= x \cdot 0.5^k \\\\
        \mathbf{y} &= \left[\sin(s_0), \dots, \sin(s_{\lfloor\text{num}/2\rfloor-1}),
                            \cos(s_0), \dots, \cos(s_{\lfloor\text{num}/2\rfloor-1}), x\right]
        \end{aligned}
                             
    Arguments:
        x: Value from which the fourier features should be computed.
        num: Number of fourier features that should be computed.

    Returns:
        Fourier features of `x`.
    """

    if num <= 1:
        return x

    k = torch.arange(num // 2, device=x.device, dtype=x.dtype)
    x = x.unsqueeze(-1)

    s = x * 0.5**k

    y = [s.sin(), s.cos()]
    if num & 1:
        y.append(x)
    y = torch.cat(y, dim=-1)

    return y.flatten(-2)


class Apply(nn.Module):
    r"""Apply Module.

    Applies :func:`fn` to an input and returns the result.
    """

    def __init__(self, fn: Callable[[Any], Any]) -> None:
        r"""
        Arguments:
            fn: Function that should be applied.
        """

        super().__init__()
        self.fn = fn

    def extra_repr(self) -> str:
        r"""Returns the representation of :func:`fn`.

        Returns:
            Representation of :func:`fn`.
        """

        return repr(self.fn)

    def forward(self, input: Any) -> Any:
        r"""Call :func:`fn` on `input` and return result.

        Arguments:
            x: Input for :func:`fn`.

        Returns:
            Result of `fn(input)`.
        """

        return self.fn(input)


class EGNN(gnn.MessagePassing):
    r"""E(n) Equivariant Graph Neural Network.

    Implementation of an E(n) Equivariant Graph Neural Network as described in
    [1]_ that supports multiple node coordinates. This way we can encode
    different spatial features as coordinate; e.g., vectors that originate from
    a single node.

    Node features are usually two-dimensional tensors of the size :math:`(N, F)`,
    with :math:`N` being the number of nodes and :math:`F` the number of
    features.

    Similarly, coordinates have the size :math:`(N, D)` with :math:`D` being the
    dimension of the coordinate space. Alternatively, to encode multiple
    coordinates per node, the coordinates can be :math:`(N, C, D)` with
    :math:`C` as the number of individual coordinates per node.

    Edge features are optional and have the size :math:`(E, G)` with :math:`E`
    as the number of edges and :math:`G` as the number of edge features.

    Example:
        EGNN instance with 16 input and 32 output features and two coordinates
        per node.

        >>> from bpp.model.egnn import EGNN
        >>> fn = EGNN(
        ...     dims_node=(16, 64, 32),
        ...     dims_edge=(0, 64, 16),
        ...     dims_pos=(32,),
        ...     num_pos=2,
        )
        >>> fn
        EGNN(
          (aggr_mean): MeanAggregation()
          (aggr_sum): SumAggregation()
          (edge_net): Sequential(
            (0): Linear(in_features=34, out_features=64, bias=True)
            (1): GELU(approximate='tanh')
            (2): Linear(in_features=64, out_features=16, bias=True)
            (3): GELU(approximate='tanh')
          )
          (node_net): Sequential(
            (0): Linear(in_features=32, out_features=64, bias=True)
            (1): GELU(approximate='tanh')
            (2): Linear(in_features=64, out_features=32, bias=True)
          )
          (pos_net): Sequential(
            (0): Linear(in_features=16, out_features=32, bias=True)
            (1): GELU(approximate='tanh')
            (2): Linear(in_features=32, out_features=2, bias=True)
          )
        )
    """

    def __init__(
        self,
        dims_node: Sequence[int],
        dims_edge: Sequence[int],
        dims_gate: Sequence[int] = (),
        dims_pos: Sequence[int] = (),
        num_pos: int = 1,
        num_encode: int = 1,
        pos_scale: Optional[float] = None,
        norm: bool = False,
        clamp: float = 0.0,
        dropout: float = 0.0,
        dropout_edge: float = 0.0,  # noqa: F811
        activation: Optional[nn.Module] = None,
        init: Optional[Callable[[nn.Module, str], None]] = None,
        norm_cls: type[nn.Module] = nn.BatchNorm1d,
        dropout_cls: type[nn.Module] = nn.Dropout,
    ) -> None:
        r"""
        Arguments:
            dims_node: Dimensions for the node network: `dims_node[0]` is the
                number of input features and `dims_node[-1]` is the number of
                output features.  At least 2 dimensions are required. All
                remaining dimensions are used for the hidden-layers of the
                network.
            dims_edge: Dimensions for the edge network: `dims_edge[0]` is the
                number of edge features (0 if graphs have no edge features),
                `dims_edge[-1]` is the number of message features.  All
                remaining dimensions are used for the hidden-layers of the edge
                network that computes the messages.
            dims_gate: Dimensions for the message gating network. If empty,
                no message gating network will be created and message gating
                will be disabled.
            dims_pos: Dimensions for the coordinate update network. If empty,
                no coordinate update network will be created and coordinate
                updates will be disabled.
            num_pos: Number of node positions. Each node can be assigned to one
                position, but can also be assigned to multiple positions.
            num_encode: Number of fourier-encoded features that are computed
                from the distances between node coordinates. If `num_encode` is
                even, only the fourier-encoded features will be used. If odd,
                the original distances will also be used alongside the
                fourier-encoded features.  Default is 1, which means that only
                the original distances are used, without any fourier features.
            pos_scale: Initial scaler for coordinate updates. If pos_scale is
                `None`, the penultimate layer of the coordinate update network
                will use a linear activation and omit the scaling factor.
            norm: Whether to use layer and message normalization while computing
                and aggregating messages.
            clamp: Threshold for clamping coordinate updates (between `-clamp`
                and `clamp`). Unclamped if 0.
            dropout: Dropout probability in-between network layers. No dropout
                if 0.
            dropout_edge: Probability for edges to be dropped. No edges will be
                dropped if 0.
            activation: Activation function that is used in-between network
                layers.
            init: Initializer that is used for network layers. First argument is
                the module that should be initialized, the second is the
                non-linearity, in case of a linear module, that should be used.
                The default non-linearity should adhere to whatever is used for
                `activation`. See :func:`parameter_init`.
            norm_cls: Class instance of normalization module.
            dropout_cls: Class instance of dropout module.
        """

        for dim in dims_node:
            if dim < 1:
                raise ValueError(
                    f"dimensions in dims_node must be larger than 0, but one "
                    f"dimension is {dim}"
                )

        for dim in dims_gate:
            if dim < 1:
                raise ValueError(
                    f"dimensions in dims_gate must be larger than 0, bot one "
                    f"dimension is {dim}"
                )

        for dim in dims_gate:
            if dim < 1:
                raise ValueError(
                    f"dimensions in dims_pos must be larger than 0, bot one "
                    f"dimension is {dim}"
                )

        for dim in dims_edge[1:]:
            if dim < 1:
                raise ValueError(
                    f"dimensions in dims_edge, with the exception of the first "
                    f"one, must be larger than 0, but one dimension is {dim}"
                )

        if dims_edge[0] < 0:
            raise ValueError(
                f"first dimension in dims_edge cannot be negative, but got "
                f"{dims_edge[-1]}"
            )

        if len(dims_node) < 2:
            raise ValueError(
                f"minimum number of dimensions for dims_node is 2, but got "
                f"{len(dims_node)}"
            )

        if len(dims_edge) < 2:
            raise ValueError(
                f"minimum number of dimensions for dims_edge is 2, but got "
                f"{len(dims_edge)}"
            )

        if num_pos < 1:
            raise ValueError(
                f"minimum number of positional coordinates is 1, but got {num_pos}"
            )

        if num_encode < 1:
            raise ValueError(
                f"minimum number of fourier-encoded features for positional "
                f"coordinates is 1, but got {num_encode}"
            )

        if clamp < 0:
            raise ValueError(f"clamp value must be positive, but got {clamp}")

        if dropout < 0:
            raise ValueError(f"dropout value must be positive, but got {dropout}")

        if dropout_edge < 0:
            raise ValueError(
                f"drop_edge value must be positive, but got {dropout_edge}"
            )

        super().__init__(aggr=None, node_dim=-2)

        self.aggr_mean = gnn.MeanAggregation()
        self.aggr_sum = gnn.SumAggregation()

        update = 0 < len(dims_pos) and 0 < num_pos
        gate = 0 < len(dims_gate)

        dims_edge = (
            (dims_edge[0] + dims_node[0] * 2 + num_pos * num_encode),
            *dims_edge[1:],
        )

        dims_node = (
            (dims_node[0] + dims_edge[-1]),
            *dims_node[1:],
        )

        if update:
            dims_pos = (
                dims_edge[-1],
                *dims_pos,
                num_pos,
            )

        if gate:
            dims_gate = (
                dims_edge[-1],
                *dims_gate,
                1,
            )

        self.dims_node = dims_node
        self.dims_edge = dims_edge
        self.dims_gate = dims_gate
        self.dims_pos = dims_pos
        self.num_pos = num_pos
        self.num_encode = num_encode
        self.update_pos = update
        self.gate_msg = gate
        self.norm = norm
        self.dropout_edge = dropout_edge

        if activation is None:
            activation = nn.GELU(approximate="tanh")

        if init is None:
            init = parameter_init

        self.init = init

        self.edge_net = nn.Sequential(
            *iter_net_segments(
                *dims_edge,
                dropout=dropout,
                activation=activation,
                norm_cls=norm_cls,
                dropout_cls=dropout_cls,
            ),
        )
        if 2 <= len(dims_edge):
            self.edge_net.append(activation)

        self.node_net = nn.Sequential(
            *iter_net_segments(
                *dims_node,
                norm=norm,
                dropout=dropout,
                activation=activation,
                norm_cls=norm_cls,
                dropout_cls=dropout_cls,
            )
        )
        if 3 <= len(dims_edge) and 3 <= len(dims_node):
            self.node_net.insert(0, norm_cls(dims_node[0]))

        if update:
            self.pos_net = nn.Sequential(
                *iter_net_segments(
                    *dims_pos,
                    norm=norm,
                    clamp=clamp,
                    dropout=dropout,
                    activation=activation,
                    norm_cls=norm_cls,
                    dropout_cls=dropout_cls,
                ),
            )
            if 3 <= len(dims_edge) and 3 <= len(dims_pos):
                self.pos_net.insert(0, norm_cls(dims_pos[0]))
            if pos_scale is not None:
                self.pos_scale = nn.Parameter(
                    torch.tensor(
                        pos_scale,
                        requires_grad=True,
                    )
                )
                self.pos_net.append(nn.Tanh())
            else:
                self.pos_scale = None
        else:
            self.pos_scale = None
            self.pos_net = None

        if gate:
            self.gate_net = nn.Sequential(
                *iter_net_segments(
                    *dims_gate,
                    norm=norm,
                    dropout=dropout,
                    activation=activation,
                    norm_cls=norm_cls,
                    dropout_cls=dropout_cls,
                ),
                nn.Sigmoid(),
            )
            if 3 <= len(dims_edge) and 3 <= len(dims_gate):
                self.gate_net.insert(0, norm_cls(dims_gate[0]))
        else:
            self.gate_net = None

        if norm:
            self.msg_norm = gnn.norm.MessageNorm(learn_scale=True)
        else:
            self.msg_norm = None

    def __repr__(self) -> str:
        r"""Calls original :meth:`torch.nn.Module.__repr__`; more informative
        than :meth:`torch_geometric.nn.Module.__repr__`.

        Returns:
            String-representation of the EGNN layer.
        """

        return super(gnn.MessagePassing, self).__repr__()

    def reset_parameters(self):
        r"""Reset parameters."""

        if self.pos_scale is None:
            pos_net_nonlinearity = "linear"
        else:
            pos_net_nonlinearity = "tanh"

        init_net_segments(self.init, self.edge_net)
        init_net_segments(self.init, self.node_net, "linear")
        init_net_segments(self.init, self.pos_net, pos_net_nonlinearity)
        init_net_segments(self.init, self.gate_net, "sigmoid")
        init_net_segments(self.init, self.msg_norm)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Adj] = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute forward pass.

        Arguments:
            x: Node features.
            pos: Node coordinates.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            batch: Node to batch mapping.

        Returns:
            Updated node features and coordinates.
        """

        if pos.dim() == x.dim():
            pos = pos.unsqueeze(-2)
            pos_is_unsqueezed = True
        else:
            pos_is_unsqueezed = False

        if 0 < self.dropout_edge:
            edge_index, edge_mask = dropout_edge(
                edge_index,
                p=self.dropout_edge,
                force_undirected=True,
                training=self.training,
            )
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]

        self.node_dim += x.dim()
        x, pos = self.propagate(
            edge_index=edge_index,
            x=x,
            pos=pos,
            edge_attr=edge_attr,
            batch=batch,
        )
        self.node_dim -= x.dim()

        if pos_is_unsqueezed:
            pos = pos.squeeze(-2)

        return x, pos

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Optional[Tensor],
        pos_j: Optional[Tensor],
        edge_attr: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        r"""Compute messages.

        Arguments:
            x_i: Source node features.
            x_j: Target node features.
            pos_i: Source node coordinates.
            pos_j: Target node coordinates.
            edge_attr: Edge features; between source and target node.

        Returns:
            Tuple of messages between source and target nodes, and difference
            between source and target node coordinates.
        """

        msg_input = [x_i, x_j]

        vec_ij = pos_i - pos_j
        dst_ij = vec_ij.square().sum(dim=-1)
        msg_input.append(fourier_encode(dst_ij, self.num_encode))

        if edge_attr is not None:
            msg_input.append(edge_attr)

        msg_ij = self.edge_net(torch.cat(msg_input, dim=-1))

        if self.pos_net is not None:
            force_ij = self.pos_net(msg_ij)
            norm_ij = (dst_ij + 1e-9).sqrt()
            if self.pos_scale is not None:
                force_ij = force_ij * self.pos_scale
            vec_ij = vec_ij * (force_ij / norm_ij).unsqueeze(-1)
        else:
            vec_ij = None

        if self.gate_net is not None:
            msg_ij = msg_ij * self.gate_net(msg_ij)

        return msg_ij, vec_ij

    def aggregate(
        self,
        messages: tuple[Tensor, Optional[Tensor]],
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        r"""Aggregate messages.

        Arguments:
            messages: Tuple of messages and coordinate differences between
                nodes.
            index: Indices of elements for applying the aggregation. Either
                `index` or `ptr` must be defined.
            ptr: Offsets for computing the aggregation based on sorted inputs,
                if provided. Either `index` or `ptr` must be defined.
            dim_size: Size of output at dimension `self.node_dim` after
                aggregation.

        Returns:
            Tuple of aggregated messages and aggregated coordinate updates.
        """

        def mean(x):
            return self.aggr_mean(
                x,
                index,
                ptr,
                dim_size,
                dim=self.node_dim,
            )

        def sum(x):
            return self.aggr_sum(
                x,
                index,
                ptr,
                dim_size,
                dim=self.node_dim,
            )

        msg_ij, vec_ij = messages

        msg = sum(msg_ij)

        if vec_ij is not None:
            pos_update = mean(vec_ij)
        else:
            pos_update = None

        return msg, pos_update

    def update(
        self,
        messages: tuple[Tensor, Optional[Tensor]],
        x: Tensor,
        pos: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        r"""Update nodes.

        Arguments:
            messages: Tuple of aggregated messages and coordinate differences.
            x: Node features.
            pos: Node coordinates.

        Returns:
            Tuple of updated node features and updated node coordinates.
        """

        msg, pos_update = messages

        if self.msg_norm:
            msg = self.msg_norm(x, msg)

        x = self.node_net(torch.cat([x, msg], dim=-1))

        if pos_update is not None:
            pos = pos + pos_update

        return x, pos


class DeepEGNN(nn.Module):
    r"""Deep E(n) Equivariant Neural Network.

    Implementation of an E(n) Equivariant Neural Network with :class:`EGNN` as
    its layers.

    Arguments can be broadcasted to individual layers during initialization. For
    example, if `layers = 4` and `dims_node = (64, 32)`, then `dims_node` will
    be used for each of the 4 layers to initialize the hidden and output layer
    of the node network. If on the other hand `dims_node = [(64, 32), (128, 64),
    (128, 64), (64, 32)]`, then the first and last layer will use `(64, 32)` and
    the second and third layer will use `(128, 64)` to initialize their hidden
    and output layers.

    Example:
        DeepEGNN instance with 16 input and 64 output features with 2 EGNN
        layers each having 64 and 128 hidden node features and 32 and 64
        intermediate output node features.

        >>> from egnn import DeepEGNN
        >>> fn = DeepEGNN(
        ...     dims_node=[(16, 64, 32), (128, 64)],
        ...     dims_edge=[(0, 64, 32, 64)],
        ...     layers=2,
        ... )
        >>> fn
        DeepEGNN(
          (layers): ModuleList(
            (0): ModuleDict(
              (egnn): EGNN(
                (aggr_mean): MeanAggregation()
                (aggr_sum): SumAggregation()
                (edge_net): Sequential(
                  (0): Linear(in_features=33, out_features=64, bias=True)
                  (1): GELU(approximate='tanh')
                  (2): Linear(in_features=64, out_features=32, bias=True)
                  (3): GELU(approximate='tanh')
                  (4): Linear(in_features=32, out_features=64, bias=True)
                  (5): GELU(approximate='tanh')
                )
                (node_net): Sequential(
                  (0): Linear(in_features=80, out_features=64, bias=True)
                  (1): GELU(approximate='tanh')
                  (2): Linear(in_features=64, out_features=32, bias=True)
                )
              )
              (drop): Identity()
              (norm): ModuleDict(
                (x): Identity()
                (pos): Identity()
              )
            )
            (1): ModuleDict(
              (egnn): EGNN(
                (aggr_mean): MeanAggregation()
                (aggr_sum): SumAggregation()
                (edge_net): Sequential(
                  (0): Linear(in_features=65, out_features=64, bias=True)
                  (1): GELU(approximate='tanh')
                  (2): Linear(in_features=64, out_features=32, bias=True)
                  (3): GELU(approximate='tanh')
                  (4): Linear(in_features=32, out_features=64, bias=True)
                  (5): GELU(approximate='tanh')
                )
                (node_net): Sequential(
                  (0): Linear(in_features=96, out_features=128, bias=True)
                  (1): GELU(approximate='tanh')
                  (2): Linear(in_features=128, out_features=64, bias=True)
                )
              )
              (drop): Identity()
              (norm): ModuleDict(
                (x): Identity()
                (pos): Identity()
              )
            )
            (2): ModuleDict(
              (egnn): EGNN(
                (aggr_mean): MeanAggregation()
                (aggr_sum): SumAggregation()
                (edge_net): Sequential(
                  (0): Linear(in_features=129, out_features=64, bias=True)
                  (1): GELU(approximate='tanh')
                  (2): Linear(in_features=64, out_features=32, bias=True)
                  (3): GELU(approximate='tanh')
                  (4): Linear(in_features=32, out_features=64, bias=True)
                  (5): GELU(approximate='tanh')
                )
                (node_net): Sequential(
                  (0): Linear(in_features=128, out_features=64, bias=True)
                )
              )
              (drop): Identity()
              (norm): ModuleDict(
                (x): Identity()
                (pos): Identity()
              )
            )
          )
        )
    """

    def __init__(
        self,
        dims_node: Broadcast[Sequence[int]],
        dims_edge: Broadcast[Sequence[int]],
        dims_gate: Broadcast[Sequence[int]] = [()],
        dims_pos: Broadcast[Sequence[int]] = [()],
        layers: int = 3,
        num_pos: int = 1,
        dim_pos: int = 3,
        num_encode: Broadcast[int] = 1,
        pos_scale: Broadcast[Optional[float]] = None,
        norm: Broadcast[tuple[bool, bool] | bool] = False,
        clamp: Broadcast[float] = 0.0,
        dropout: Broadcast[tuple[float, float] | float] = 0.0,
        dropout_edge: Broadcast[float] = 0.0,
        residual: Broadcast[bool] = False,
        activation: Broadcast[Optional[nn.Module]] = None,
        init: Broadcast[Optional[Callable[[nn.Module, str], None]]] = None,
        norm_cls: Broadcast[
            tuple[type[nn.Module], type[nn.Module]] | type[nn.Module]
        ] = nn.BatchNorm1d,
        dropout_cls: Broadcast[
            tuple[type[nn.Module], type[nn.Module]] | type[nn.Module]
        ] = nn.Dropout,
    ) -> None:
        r"""
        Arguments:
            dims_node: Dimensions for the node networks. This can be a sequence
                of integers with `dims_node[0]` describing the number of input
                features and `dims_node[-1]` describing the number output
                features for the node networks in all layers. Note that the
                number of input features is only valid for the first layer; in
                all subsequent layers the number of input features is inferred
                from the number of output features of the previous layer.
                Alternatively, `dims_node` can also be a sequence of sequences
                of integers, describing the aforementioned properties for each
                layer individually. In this case, `dims_node[0][0]` is the
                number of input features for the first layer, but for all
                subsequent layers the first value represents the dimension of
                the first hidden layer of the node networks, since the number of
                input features is inferred from the previous number of output
                features of the previous layer; e.g., `dims_node[1][0]` is the
                dimension of the first hidden layer in the node network of the
                second layer in this network.
            dims_edge: Dimensions for the edge networks. Follows the same
                broadcasting scheme as `dims_node`.
            dims_gate: Dimensions for the message gating network. Follows the
                same broadcasting scheme as `dims_node`. If empty, message
                gating is disabled.
            dims_pos: Dimensions for the coordinate update network. Follows the
                same broadcasting scheme as `dims_node`. If empty, coordinate
                update is disabled.
            layers: Number of :class:`EGNN` layers.
            num_pos: Number of node positions.
            dim_pos: Number of coordinate dimensions.
            num_encode: Number of fourier-encoded features.
            pos_scale: Initial scaler for coordinate updates. If pos_scale is
                `None`, the penultimate layer of a coordinate update network
                will use a linear activation and omit the scaling factor.
            norm: Whether to normalize a given layer and whether to use
                normalization after that given layer. If a single boolean or a
                list of single booleans, then normalization will be applied
                within and after the corresponding layers according to the
                boolean value. If a tuple of booleans or a list of tuples of
                booleans, then the first boolean will determine whether to use
                normalization after a layer and the second will determine
                whether to use normalization within that layer.
            clamp: Threshold for clamping coordinate updates (between `-clamp`
                and `clamp`). Unclamped if 0.
            dropout: Dropout probabilities for in-between layers and within
                layers. Values will be broadcasted to all layers, if only two
                probabilities are given, or to all layers and in-between and
                within layers, if only one probability is given. Dropout is
                omitted if 0.
            dropout_edge: Probability for edges to be dropped. No edges will be
                dropped if 0.
            residual: Whether a layer is a residual layer. If `residual` is a
                single value, then layers will only be residual layers if their
                number of input and output features matches. Coordinates are not
                updated through residual connections, independent of this
                toggle.
            activation: Activation function that is used for the :class:`EGNN`
                layers.
            init: Initializer that is used for network layers. First argument is
                the module that should be initialized, the second is the
                non-linearity, in case of a linear module, that should be used.
                The default non-linearity should adhere to whatever is used for
                `activation`. See :func:`parameter_init`.
            norm_cls: Class instance of normalization module. If tuple the first
                instance will be used in-between layers and the second within
                layers.
            dropout_cls: Class instance of dropout module. If tuple the first
                instance will be used in-between layers and the second within
                layers.
        """

        if init is None:
            init = parameter_init

        super().__init__()

        if not isinstance(dims_node[0], Sequence):
            features_node = dims_node[0]
            dims_node = dims_node[1:]
        else:
            features_node = dims_node[0][0]
            dims_node = [dims_node[0][1:], *dims_node[1:]]

        update_is_broadcasted = all(not isinstance(dim, Sequence) for dim in dims_pos)
        residual_is_broadcasted = not isinstance(residual, Sequence)

        dims_node = broadcast_arg("dims_node", dims_node, [layers, None])
        dims_edge = broadcast_arg("dims_edge", dims_edge, [layers, None])
        dims_gate = broadcast_arg("dims_gate", dims_gate, [layers, None])
        dims_pos = broadcast_arg("dims_pos", dims_pos, [layers, None])
        num_encode = broadcast_arg("num_encode", num_encode, [layers])
        pos_scale = broadcast_arg("pos_scale", pos_scale, [layers])
        norm = broadcast_arg("norm", norm, [layers, 2])
        clamp = broadcast_arg("clamp", clamp, [layers])
        dropout = broadcast_arg("dropout", dropout, [layers, 2])
        dropout_edge = broadcast_arg("dropout_edge", dropout_edge, [layers])
        residual = broadcast_arg("residual", residual, [layers])
        activation = broadcast_arg("activation", activation, [layers])
        init = broadcast_arg("init", init, [layers])
        norm_cls = broadcast_arg("norm_cls", norm_cls, [layers, 2])
        dropout_cls = broadcast_arg("dropout_cls", dropout_cls, [layers, 2])

        dim_in = features_node
        for i in range(len(dims_node)):
            dims_node[i] = (dim_in, *dims_node[i])
            dim_in = dims_node[i][-1]

        if update_is_broadcasted:
            dims_pos[-1] = ()

        for i in range(len(residual)):
            if residual[i] and dims_node[i][0] != dims_node[i][-1]:
                if residual_is_broadcasted:
                    residual[i] = False
                else:
                    raise ValueError(
                        f"cannot use residual connections if input and output "
                        f"dimensions disagree: number of input features at "
                        f"layer {i} is {dims_node[i][0]}, number of output "
                        f"features is {dims_node[i][-1]}"
                    )

        self.residual = residual
        self.init = init

        self.layers = nn.ModuleList()

        for i in range(len(dims_node)):
            layer = nn.ModuleDict()

            layer.egnn = EGNN(
                dims_node[i],
                dims_edge[i],
                dims_gate[i],
                dims_pos[i],
                num_pos,
                num_encode[i],
                pos_scale[i],
                norm[i][1],
                clamp[i],
                dropout[i][1],
                dropout_edge[i],
                activation[i],
                init[i],
                norm_cls[i][1],
                dropout_cls[i][1],
            )

            if dropout[i][0]:
                layer.drop = dropout_cls[0][i](dropout[i][0])
            else:
                layer.drop = nn.Identity()

            if norm[i][0]:
                layer.norm = nn.ModuleDict()
                layer.norm.x = norm_cls[i][0](dims_node[i][-1])
                if residual[i]:
                    layer.norm.s = norm_cls[i][0](dims_node[i][-1])
                else:
                    layer.norm.s = nn.Identity()
                if 0 < len(dims_pos[i]):
                    layer.norm.pos = nn.BatchNorm1d(num_pos * dim_pos)
                else:
                    layer.norm.pos = nn.Identity()
            else:
                layer.norm = nn.ModuleDict()
                layer.norm.x = nn.Identity()
                layer.norm.pos = nn.Identity()

            self.layers.append(layer)

    def reset_parameters(self):
        r"""Reset parameters."""

        for init, layer in zip(self.init, self.layers):
            layer.egnn.reset_parameters()
            layer.norm.x.apply(init)
            layer.norm.pos.apply(init)

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
            Updated node features and coordinates.
        """

        for i, layer in enumerate(self.layers):
            s, pos = layer.egnn(
                x,
                pos,
                edge_index,
                edge_attr,
                batch,
            )
            if self.residual[i]:
                s = layer.norm.s(s)
                s = layer.drop(s)
                x = s + x
            else:
                s = layer.drop(s)
                x = s
            if self.residual[i]:
                x = s + x
            else:
                x = s
            x = layer.norm.x(x)
            pos = layer.norm.pos(pos.flatten(x.dim() - 1)).view(pos.shape)

        return x, pos
