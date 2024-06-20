r"""Module containing implementations of unrecorded E(n) Equivariant Graph 
Neural Networks.

These variants are primarily used for saving memory by storing only the inputs
of each layer and recomputing the intermediate values and outputs during
back-propagation.
"""

import gc
from typing import Sequence, Callable, Optional

import torch
from torch import nn
from torch import Tensor
from torch_geometric.typing import Adj

from . import EGNN, parameter_init
from .._unrecorded import UnrecordedModule
from .._arguments import broadcast_arg, Broadcast


class UnrecordedEGNN(UnrecordedModule):
    r"""Unrecorded E(n) Equivariant Graph Neural Network.

    Implementation of an E(n) Equivariant Graph Neural Network that only stores
    the inputs when computing the forward pass during differentiation and that
    recomputes all intermediate values and the outputs during the backwards
    pass to save memory.

    Please refer to :class:`bpp.model.egnn.EGNN` for further details on how to
    use this module.
    """

    def __init__(
        self,
        dims_node: Sequence[int],
        dims_edge: Sequence[int],
        dims_gate: Sequence[int] = (),
        dims_pos: Sequence[int] = (),
        channels: int = 1,
        num_pos: int = 1,
        num_encode: int = 1,
        pos_scale: Optional[float] = None,
        norm: bool = False,
        clamp: float = 0.0,
        dropout: tuple[float, float] | float = 0.0,
        dropout_edge: float = 0.0,
        activation: nn.Module = None,
        init: Callable[[nn.Module, str], None] = None,
        norm_cls: type[nn.Module] = nn.BatchNorm1d,
        dropout_cls: (
            tuple[type[nn.Module], type[nn.Module]] | type[nn.Module]
        ) = nn.Dropout,
        disable: bool = False,
    ) -> None:
        """
        Arguments:
            dims_node: Dimensions for the node network: `dims_node[0]` is the
                number of input features and `dims_node[-1]` is the number of
                output features, `dims_node[1]` is the number of input-channel
                features and `dims_node[-2]` is the number of output-channel
                features. At least 4 dimensions are required. All
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
            channels: Number of channels; i.e., individual instances of
                :class:`bpp.model.egnn.EGNN`. Linear mixing layers are used to
                distribute and recombine features amongst different channels.
            num_pos: Number of node positions. Each node can be assigned to one
                position, but can also be assigned to multiple positions.
            num_encode: Number of fourier-encoded features that are computed
                from the distances between node coordinates. If `num_encode` is
                even, only the fourier-encoded features will be used. If odd,
                the original distances will also be used alongside the
                fourier-encoded features.  Default is 1, which means that only
                the original distances are used, without any fourier features.
            pos_scale: Initial scaler for coordinate updates. If pos_scale is
                `None`, the penultimate layer of a coordinate update network
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
                `activation`. See :func:`bpp.model.egnn.parameter_init`.
            norm_cls: Class instance of normalization module.
            dropout_cls: Class instance of dropout module. If tuple the first
                instance will be used after a channel and the second within a
                channel.
            disable: Whether to disable layer-wise recomputation and store
                intermediate values for the back-propagation method. Used
                primarily for debugging.
        """

        if init is None:
            init = parameter_init

        if len(dims_node) < 4:
            raise ValueError(
                f"minimum number of node dimensions is 4, but got {len(dims_node)}"
            )

        if channels < 1:
            raise ValueError(
                f"minimum number of channels is 1, but got {channels} for "
                f"number of channels"
            )

        if not isinstance(dropout, Sequence):
            dropout = (dropout, dropout)

        if not isinstance(dropout_cls, Sequence):
            dropout_cls = (dropout_cls, dropout_cls)

        super().__init__(
            disable,
            preserve_rng_state=True,
            forward_has_only_one_output=False,
        )

        self.mix_in = nn.Linear(
            dims_node[0],
            dims_node[1] * channels,
            bias=False,
        )
        self.mix_out = nn.Linear(
            dims_node[-2] * channels,
            dims_node[-1],
            bias=False,
        )
        self.channels = nn.ModuleList()

        for _ in range(channels):
            channel = nn.ModuleDict()
            channel.egnn = EGNN(
                dims_node[1:-1],
                dims_edge,
                dims_gate,
                dims_pos,
                num_pos,
                num_encode,
                pos_scale,
                norm,
                clamp,
                dropout[1],
                dropout_edge,
                activation,
                init,
                norm_cls,
                dropout_cls[1],
            )
            if 0 < dropout[0]:
                channel.drop = dropout_cls[0](dropout[0])
            else:
                channel.drop = nn.Identity()
            self.channels.append(channel)

    def _forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
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

        num_channels = len(self.channels)
        u = self.mix_in(x)
        u_chunks = torch.chunk(u, num_channels, dim=-1)
        v_chunks = []
        pos_s = 0

        for u_i, channel in zip(u_chunks, self.channels):
            v_i, pos_i = channel.egnn(u_i, pos, edge_index, edge_attr, batch)
            v_i = channel.drop(v_i)
            v_chunks.append(v_i)
            pos_s = pos_s + pos_i

        v = torch.cat(v_chunks, dim=-1)
        y = self.mix_out(v)
        pos = pos_s / num_channels

        return y, pos

    def reset_parameters(self):
        r"""Reset parameters."""

        for channel in self.channels:
            channel.egnn.reset_parameters()

        nn.init.kaiming_uniform_(self.mix_in.weight, nonlinearity="linear")
        nn.utils.parametrizations.spectral_norm(self.mix_in, name="weight", dim=0)

        nn.init.kaiming_uniform_(self.mix_out.weight, nonlinearity="linear")
        nn.utils.parametrizations.spectral_norm(self.mix_out, name="weight", dim=0)


class DeepUnrecordedEGNN(nn.Module):
    r"""Deep Unrecorded E(n) Equivariant Neural Network.

    Implementation of an E(n) Equivariant Neural Network that utilizes
    :class:`UnrecordedEGNN` modules as it's layers to save memory during
    differentiation.

    Please refer to :class:`bpp.model.egnn.DeepEGNN` for further detail on how
    to use this module.
    """

    def __init__(
        self,
        dims_node: Broadcast[Sequence[int]],
        dims_edge: Broadcast[Sequence[int]],
        dims_gate: Broadcast[Sequence[int]] = (),
        dims_pos: Broadcast[Sequence[int]] = (),
        channels: Broadcast[int] = 4,
        layers: int = 3,
        tied: bool = False,
        num_pos: int = 1,
        dim_pos: int = 3,
        num_encode: Broadcast[int] = 1,
        pos_scale: Broadcast[Optional[float]] = None,
        norm: Broadcast[tuple[bool, bool] | bool] = False,
        clamp: Broadcast[float] = 0.0,
        dropout: Broadcast[tuple[float, float] | float] = 0.0,
        dropout_edge: Broadcast[float] = 0.0,
        residual: Broadcast[bool] = True,
        activation: Broadcast[Optional[nn.Module]] = None,
        init: Broadcast[Optional[Callable[[nn.Module, str], None]]] = None,
        norm_cls: Broadcast[
            tuple[type[nn.Module], type[nn.Module]] | type[nn.Module]
        ] = nn.BatchNorm1d,
        dropout_cls: Broadcast[
            tuple[type[nn.Module], type[nn.Module]] | type[nn.Module]
        ] = nn.Dropout,
        disable: Broadcast[bool] = False,
    ) -> None:
        r"""
        Arguments:
            dims_node: Dimensions for the node networks. This can be a sequence
                of integers with `dims_node[0]` describing the number of input
                features and `dims_node[-1]` describing the number output
                features for the node networks in all layers. Note that the
                number of input features is only valid for the first layer and
                that in all subsequent layers the number of input features will
                be the same as the number of output features of the previous
                layer. Alternatively, `dims_node` can also be a sequence of
                sequences of integers, describing the aforementioned properties
                for each layer individually. In this case, `dims_node[0][0]` is
                the number of input features for the first layer, but for all
                subsequent layers the first value represents the dimension of
                the first hidden layer of the node networks, since the number of
                input features will be derived from the previous number of
                output features of the previous layer; e.g., `dims_node[1][0]`
                is the dimension of the first hidden layer in the node network
                of the second layer in this network. Also note that the
                dimensions must adhere to the specification in
                :class:`UnrecordedEGNN`, such that they must also specify the
                number of channel features.
            dims_edge: Dimensions for the edge networks. Follows the same
                broadcasting scheme as `dims_node`.
            dims_gate: Dimensions for the message gating network. Follows the
                same broadcasting scheme as `dims_node`. If empty, message
                gating is disabled.
            dims_pos: Dimensions for the coordinate update network. Follows the
                same broadcasting scheme as `dims_node`. If empty, coordinate
                update is disabled.
            channels: Number of channels within each :class:`UnrecordedEGNN`
                layer.
            layers: Number of :class:`UnrecordedEGNN` layers.
            tied: Whether the same layer is repeated. Layer-wise arguments are
                not supported in this case.
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
                omitted if  Dropout is
                omitted if 0.
            dropout_edge: Probability for edges to be dropped. No edges will be
                dropped if 0.
            residual: Whether a layer is a residual layer. If `residual` is a
                single value, then layers will only be residual layers if their
                number of input and output features matches. Coordinates are not
                updated through residual connections, independent of this
                toggle.
            activation: Activation function that is used for the
                :class:`UnrecordedEGNN` layers.
            init: Initializer that is used for network layers. First argument is
                the module that should be initialized, the second is the
                non-linearity, in case of a linear module.  The default
                non-linearity should adhere to whatever is used for
                `activation`. See :func:`bpp.model.egnn.parameter_init`.
            norm_cls: Class instance of normalization module. If tuple the first
                instance will be used in-between layers and the second within
                layers.
            dropout_cls: Class instance of dropout module. If tuple the first
                instance will be used in-between layers and the second within
                layers.
            disable: Whether to disable layer-wise recomputation and store
                intermediate values for the back-propagation method. Used
                primarily for debugging.
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

        if tied:
            layer_instances = 1
            pos_is_broadcasted = False
            residual_is_broadcasted = False
        else:
            layer_instances = layers
            pos_is_broadcasted = not any(isinstance(dim, Sequence) for dim in dims_pos)
            residual_is_broadcasted = not isinstance(residual, Sequence)

        dims_node = broadcast_arg("dims_node", dims_node, [layer_instances, None])
        dims_edge = broadcast_arg("dims_edge", dims_edge, [layer_instances, None])
        dims_gate = broadcast_arg("dims_gate", dims_gate, [layer_instances, None])
        dims_pos = broadcast_arg("dims_pos", dims_pos, [layer_instances, None])
        channels = broadcast_arg("channels", channels, [layer_instances])
        num_encode = broadcast_arg("num_encode", num_encode, [layer_instances])
        pos_scale = broadcast_arg("pos_scale", pos_scale, [layer_instances])
        norm = broadcast_arg("norm", norm, [layer_instances, 2])
        clamp = broadcast_arg("clamp", clamp, [layer_instances])
        dropout = broadcast_arg("dropout", dropout, [layer_instances, 2])
        dropout_edge = broadcast_arg("dropout_edge", dropout_edge, [layer_instances])
        residual = broadcast_arg("residual", residual, [layer_instances])
        activation = broadcast_arg("activation", activation, [layer_instances])
        init = broadcast_arg("init", init, [layer_instances])
        norm_cls = broadcast_arg("norm_cls", norm_cls, [layer_instances, 2])
        dropout_cls = broadcast_arg("dropout_cls", dropout_cls, [layer_instances, 2])
        disable = broadcast_arg("disable", disable, [layer_instances])

        dim_in = features_node
        for i in range(len(dims_node)):
            dims_node[i] = (dim_in, *dims_node[i])
            dim_in = dims_node[i][-1]

        if tied and dims_node[0][0] != dims_node[0][-1]:
            raise ValueError(
                f"input and output node dimensions must be the same if tied, "
                f"but got {dims_node[0][0]} input and {dims_node[0][-1]} "
                f"output dimensions"
            )

        if pos_is_broadcasted:
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

        self.tied = tied
        self.residual = residual
        self.init = init
        self.layers = nn.ModuleList()

        for i in range(layers):

            if tied and i != 0:
                layer = self.layers[0]
            else:
                layer = nn.ModuleDict()

                layer.egnn = UnrecordedEGNN(
                    dims_node[i],
                    dims_edge[i],
                    dims_gate[i],
                    dims_pos[i],
                    channels[i],
                    num_pos,
                    num_encode[i],
                    pos_scale[i],
                    norm[i][1],
                    clamp[i],
                    dropout[i],
                    dropout_edge[i],
                    activation[i],
                    init[i],
                    norm_cls[i][1],
                    dropout_cls[i],
                    disable[i],
                )

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
                    layer.norm.s = nn.Identity()
                    layer.norm.pos = nn.Identity()

            self.layers.append(layer)

    def reset_parameters(self):
        r"""Reset parameters."""

        for layer, init in zip(self.layers, self.init):
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

        # Free unused memory.
        if self.training:
            gc.collect()
            torch.cuda.empty_cache()

        for i, layer in enumerate(self.layers):
            if self.tied:
                i = 0
            s, pos = layer.egnn(
                x,
                pos,
                edge_index,
                edge_attr,
                batch,
            )
            if self.residual[i]:
                s = layer.norm.s(s)
                x = s + x
            else:
                x = s
            x = layer.norm.x(x)
            pos = layer.norm.pos(pos.flatten(x.dim() - 1)).view(pos.shape)

        return x, pos
