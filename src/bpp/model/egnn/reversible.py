r"""Module containing implementation of reversible E(n) Equivariant Graph Neural
Networks.

These variants are primarily used for saving memory by not storing any
intermediate values and recomputing the inputs and required intermediate values
in an auto-regressive scheme from the outputs of a given layer during
back-propagation.

**Note**: PyTorch does not provide a feasible solution on how to pass the
outputs to the backwards function without storing them as intermediate results,
meaning that we would need to free and reallocate the memory of a given value
manually. However, this often causes memory leaks, invalid accesses' or only
copies of a value to be freed. If we however manage to make an architecture
work, we often end up with numerical issues; e.g., normalization modules. This
is why, by default, the outputs of each layer will be stored as intermediate
values during the forward pass.

**Note**: Backward hooks might provide more control on how to pass the output
backwards.

References:
    .. [1] Li, G., Müller, M., Ghanem, B., & Koltun, V. (2022). Training Graph
        Neural Networks with 1000 Layers.
    .. [2] Rezende, D. J., & Mohamed, S. (2016). Variational Inference with
        Normalizing Flows.
"""

# TODO: norm_cls and dropout_cls for within and after blocks.

import gc
from typing import Sequence, Callable, Optional

import torch
from torch import nn
from torch import Tensor
from torch_geometric.typing import Adj

from . import EGNN, parameter_init
from .._reversible import ReversibleModule
from .._deterministic import deterministic_context
from .._arguments import broadcast_arg, Broadcast


class ReversibleEGNN(ReversibleModule):
    r"""Reversible E(n) Equivariant Graph Neural Network.

    Implementation of an E(n) Equivariant Graph Neural Network with reversible
    layers to omit storing any intermediate values during the forward pass when
    computing the differentials. Inputs of each layer are computed in reverse
    from the outputs to compute the differentials of a given layer.

    Please refer to :class:`bpp.model.egnn.EGNN` for further details on how to
    use this module.
    """

    def __init__(
        self,
        dims_node: Sequence[int],
        dims_edge: Sequence[int],
        dims_gate: Sequence[int] = (),
        dims_pos: Sequence[int] = (),
        blocks: int = 2,
        num_pos: int = 1,
        dim_pos: int = 3,
        num_encode: int = 1,
        pos_scale: Optional[float] = None,
        norm: tuple[bool, bool] | bool = False,
        clamp: float = 0.0,
        dropout: tuple[float, float] | float = 0.0,
        dropout_edge: float = 0.0,
        activation: nn.Module = None,
        init: Callable[[nn.Module, str], None] = None,
        norm_cls: type[nn.Module] = nn.BatchNorm1d,
        dropout_cls: type[nn.Module] = nn.Dropout,
        disable: bool = False,
        free_inputs=False,
        free_outputs=False,
    ) -> None:
        r"""
        Arguments:
            dims_node: Dimensions for the node network of a block:
                `dims_node[0]` is the number of input features and
                `dims_node[-1]` is the number of output features. At least 2
                dimensions are required. All remaining dimensions are used for
                the hidden-layers of the network.
            dims_edge: Dimensions for the edge network of a block:
                `dims_edge[0]` is the number of edge features (0 if graphs have
                no edge features), `dims_edge[-2]` is the number of message
                features and `dims_edge[-1]` is the number of hidden-layer
                features for the coordinate update and message gating networks
                (ignored if neither `update` nor `gate` are enabled). All
                remaining dimensions are used for the hidden-layers of the edge
                network that computes the messages.
            blocks: Number of blocks in the auto-regressive reversible layer;
                i.e., individual instances of :class:`bpp.model.egnn.EGNN`. At
                least 2 are necessary.
            num_pos: Number of node positions. Each node can be assigned to one
                position, but can also be assigned to multiple positions.
            num_encode: Number of fourier-encoded features that are computed
                from the distances between node coordinates. If `num_encode` is
                even, only the fourier-encoded features will be used. If odd,
                the original distances will also be used alongside the
                fourier-encoded features.  Default is 1, which means that only
                the original distances are used, without any fourier features.
            pos_scale: TODO
            norm: Whether to use layer and message normalization while computing
                and aggregating messages.
            clamp: Threshold for clamping coordinate updates (between `-clamp`
                and `clamp`). Unclamped if 0.
            dropout: Dropout probabilities for inputs and within network layers
                of a block. If only one probability is given, then this will be
                used for the inputs and in-between layers. Dropout is omitted if
                0.
            dropout_edge: Probability for edges to be dropped. No edges will be
                dropped if 0.
            activation: Activation function that is used in-between network
                layers.
            init: Initializer that is used for network layers. First argument is
                the module that should be initialized, the second is the
                non-linearity, in case of a linear module, that should be used.
                The default non-linearity should adhere to whatever is used for
                `activation`. See :func:`bpp.model.egnn.parameter_init`.
            disable: Whether to disable reversible computations and use
                conventional back-propagation method. Used primarily for
                debugging.
            free_inputs: Whether to free inputs. Disabled by default due to
                numerical issues.
            free_outputs: Whether to free outputs. Disabled by default due to
                numerical issues.
        """

        if init is None:
            init = parameter_init

        if not isinstance(norm, Sequence):
            norm = (norm, norm)

        if not isinstance(dropout, Sequence):
            dropout = (dropout, dropout)

        super().__init__(
            disable,
            free_inputs,
            free_outputs,
            preserve_rng_state=False,
            forward_has_only_one_output=False,
        )

        self.init = init
        self.deterministic_contexts = [None] * blocks
        self.blocks = nn.ModuleList()

        for _ in range(blocks):
            block = nn.ModuleDict()

            block.egnn = EGNN(
                dims_node,
                dims_edge,
                dims_gate,
                dims_pos,
                num_pos,
                num_encode,
                pos_scale,
                norm[1],
                clamp,
                dropout[1],
                dropout_edge,
                activation,
                init,
                norm_cls,
                dropout_cls,
            )

            if norm[0]:
                block.norm = nn.ModuleDict()
                block.norm.y = norm_cls(dims_node[0])
                if 0 < len(dims_pos):
                    block.norm.pos = nn.BatchNorm1d(num_pos * dim_pos)
                else:
                    block.norm.pos = nn.Identity()
            else:
                block.norm = nn.ModuleDict()
                block.norm.x = nn.Identity()
                block.norm.pos = nn.Identity()

            if 0 < dropout[0]:
                block.drop = nn.Dropout(dropout[0])
            else:
                block.drop = nn.Identity()

            self.blocks.append(block)

    def _forward(
        self,
        x: Tensor,
        pos_a: Tensor,
        pos_b: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Compute forward pass.

        Arguments:
            x: Node features.
            pos_a: Node coordinates. Used as inputs in EGNN blocks.
            pos_b: Node coordinates. Used for auto-regressive update of
                coordinates.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            batch: Node to batch mapping.

        Returns:
            Updated node features, updated coordinates and original coordinates
            `pos_a`.
        """

        num_blocks = len(self.blocks)
        x_chunks = torch.chunk(x, num_blocks, dim=-1)
        y_chunks = []
        y_i = sum(x_chunks[1:]) / (num_blocks - 1)
        pos_s = 0

        for i, block in enumerate(self.blocks):
            args = (y_i, pos_a, edge_index, edge_attr, batch)
            self.deterministic_contexts[i] = deterministic_context(*args)
            d_i, pos_i = block.egnn(*args)
            d_i = block.norm.y(d_i)
            d_i = block.drop(d_i)
            pos_i = block.norm.pos(pos_i.flatten(d_i.dim() - 1)).view(pos_i.shape)
            y_i = x_chunks[i] + d_i
            y_chunks.append(y_i)
            pos_s = pos_s + pos_i

        y = torch.cat(y_chunks, dim=-1)
        pos_s = pos_s / num_blocks
        pos_c = (pos_s + pos_b) * 0.5

        return y, pos_c, pos_a

    def _reverse(
        self,
        y: Tensor,
        pos_c: Tensor,
        pos_a: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Compute reverse pass.

        Arguments:
            x: Node features.
            pos_c: Node coordinates. Updated coordinates from forward pass.
            pos_a: Node coordinates. Original coordinates used to compute the
                updated coordinates in the forward pass.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            batch: Node to batch mapping.

        Returns:
            Node features and original coordinates `pos_a` and `pos_b` that were
            used as inputs for the forward pass.
        """

        num_blocks = len(self.blocks)
        y_chunks = torch.chunk(y, num_blocks, dim=-1)
        x_chunks = []
        pos_s = 0

        for i, block in reversed(list(enumerate(self.blocks))):
            if i == 0:
                y_i = sum(x_chunks) / (num_blocks - 1)
            else:
                y_i = y_chunks[i - 1]
            with self.deterministic_contexts[i]():
                d_i, pos_i = block.egnn(y_i, pos_a, edge_index, edge_attr, batch)
                d_i = block.norm.y(d_i)
                d_i = block.drop(d_i)
                pos_i = block.norm.pos(pos_i.flatten(d_i.dim() - 1)).view(pos_i.shape)
            x_i = y_chunks[i] - d_i
            x_chunks.append(x_i)
            pos_s = pos_s + pos_i

        x = torch.cat(list(reversed(x_chunks)), dim=-1)
        pos_s = pos_s / num_blocks
        pos_b = pos_c * 2.0 - pos_s

        return x, pos_a, pos_b

    def reset_parameters(self):
        r"""Reset parameters."""

        for block in self.blocks:
            block.egnn.reset_parameters()


class SimplifiedReversibleEGNN(ReversibleEGNN):
    r"""Simplified variant of the Reversible E(n) Equivariant Graph Neural
    Network.

    This variant uses an auto-regressive update scheme that is more in line
    with [2]_.

    Please refer to :class:`bpp.model.egnn.EGNN` for further details on how to
    use this module.
    """

    def __init__(
        self,
        dims_node: Sequence[int],
        dims_edge: Sequence[int],
        dims_gate: Sequence[int] = (),
        dims_pos: Sequence[int] = (),
        blocks: int = 2,
        num_pos: int = 1,
        dim_pos: int = 3,
        num_encode: int = 1,
        pos_scale: Optional[float] = None,
        norm: bool = False,
        clamp: float = 0.0,
        dropout: tuple[float, float] | float = 0.0,
        dropout_edge: float = 0.0,
        activation: nn.Module = None,
        init: Callable[[nn.Module, str], None] = None,
        norm_cls: type[nn.Module] = nn.BatchNorm1d,
        dropout_cls: type[nn.Module] = nn.Dropout,
        disable: bool = False,
        free_inputs=False,
        free_outputs=False,
    ) -> None:
        r"""
        Arguments:
            dims_node: Dimensions for the node network of a block:
                `dims_node[0]` is the number of input features and
                `dims_node[-1]` is the number of output features. At least 2
                dimensions are required. All remaining dimensions are used for
                the hidden-layers of the network.
            dims_edge: Dimensions for the edge network of a block:
                `dims_edge[0]` is the number of edge features (0 if graphs have
                no edge features), `dims_edge[-1]` is the number of message
                features.  All remaining dimensions are used for the
                hidden-layers of the edge network that computes the messages.
            blocks: Number of blocks in the auto-regressive reversible layer;
                i.e., individual instances of :class:`bpp.model.egnn.EGNN`. At
                least one block is required.
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
                `None`, the penultimate layer of a coordinate update network
                will use a linear activation and omit the scaling factor.
            norm: Whether to use normalization within a block and whether to use
                normalization after a block. If only one value is given, then
                normalization within and after a block will be applied
                accordingly. If two values are given, then the first determines
                whether to apply normalization after a block and the second
                whether to apply normalization within a block.
            clamp: Threshold for clamping coordinate updates (between `-clamp`
                and `clamp`). Unclamped if 0.
            dropout: Dropout probabilities for inputs and within network layers
                of a block. If only one probability is given, then this will be
                used for the inputs and in-between layers. Dropout is omitted if
                0.
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
            dropout_cls: Class instance of dropout module.
            disable: Whether to disable reversible computations and use
                conventional back-propagation method. Used primarily for
                debugging.
            free_inputs: Whether to free inputs. Disabled by default due to
                numerical issues.
            free_outputs: Whether to free outputs. Disabled by default due to
                numerical issues.
        """

        super().__init__(
            dims_node,
            dims_edge,
            dims_gate,
            dims_pos,
            blocks - 1,
            num_pos,
            dim_pos,
            num_encode,
            pos_scale,
            norm,
            clamp,
            dropout,
            dropout_edge,
            activation,
            init,
            norm_cls,
            dropout_cls,
            disable,
            free_inputs,
            free_outputs,
        )

    def _forward(
        self,
        x: Tensor,
        pos_a: Tensor,
        pos_b: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Compute forward pass.

        Arguments:
            x: Node features.
            pos_a: Node coordinates. Used as inputs in EGNN blocks.
            pos_b: Node coordinates. Used for auto-regressive update of
                coordinates.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            batch: Node to batch mapping.

        Returns:
            Updated node features, updated coordinates and original coordinates
            `pos_a`.
        """

        num_blocks = len(self.blocks)
        x_chunks = torch.chunk(x, num_blocks + 1, dim=-1)
        y_chunks = []
        y_i = x_chunks[0]
        pos_s = 0

        for i, block in enumerate(self.blocks):
            args = (y_i, pos_a, edge_index, edge_attr, batch)
            self.deterministic_contexts[i] = deterministic_context(*args)
            d_i, pos_i = block.egnn(*args)
            d_i = block.norm.y(d_i)
            d_i = block.drop(d_i)
            pos_i = block.norm.pos(pos_i.flatten(d_i.dim() - 1)).view(pos_i.shape)
            y_i = x_chunks[i + 1] + d_i
            y_chunks.append(y_i)
            pos_s = pos_s + pos_i

        y_chunks.append(x_chunks[0])

        y = torch.cat(y_chunks, dim=-1)
        pos_s = pos_s / num_blocks
        pos_c = (pos_s + pos_b) * 0.5

        return y, pos_c, pos_a

    def _reverse(
        self,
        y: Tensor,
        pos_c: Tensor,
        pos_a: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Compute reverse pass.

        Arguments:
            x: Node features.
            pos_c: Node coordinates. Updated coordinates from forward pass.
            pos_a: Node coordinates. Original coordinates used to compute the
                updated coordinates in the forward pass.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            batch: Node to batch mapping.

        Returns:
            Node features and original coordinates `pos_a` and `pos_b` that were
            used as inputs for the forward pass.
        """

        num_blocks = len(self.blocks)
        y_chunks = torch.chunk(y, num_blocks + 1, dim=-1)
        x_chunks = [y_chunks[-1]]
        pos_s = 0

        for i, block in enumerate(self.blocks):
            y_i = y_chunks[i - 1]
            with self.deterministic_contexts[i]():
                d_i, pos_i = block.egnn(y_i, pos_a, edge_index, edge_attr, batch)
                d_i = block.norm.y(d_i)
                d_i = block.drop(d_i)
                pos_i = block.norm.pos(pos_i.flatten(d_i.dim() - 1)).view(pos_i.shape)
            x_i = y_chunks[i] - d_i
            x_chunks.append(x_i)
            pos_s = pos_s + pos_i

        x = torch.cat(x_chunks, dim=-1)
        pos_s = pos_s + num_blocks
        pos_b = pos_c * 2.0 - pos_s

        return x, pos_a, pos_b


class DeepReversibleEGNN(nn.Module):
    r"""Deep Reversible E(n) Equivariant Graph Neural Network.

    Implementation of a Deep Reversible E(n) Equivariant Graph Neural Network
    that utilizes reversible layers for saving memory during back-propagation.

    Please refer to :class:`bpp.model.egnn.DeepEGNN` for further details on how
    to use this module.
    """

    def __init__(
        self,
        dims_node: Broadcast[Sequence[int]] = (64, 64),
        dims_edge: Broadcast[Sequence[int]] = (128, 32),
        dims_gate: Broadcast[Sequence[int]] = (),
        dims_pos: Broadcast[Sequence[int]] = (),
        blocks: Broadcast[int] = 2,
        layers: int = 3,
        tied: bool = False,
        num_pos: int = 1,
        dim_pos: int = 3,
        num_encode: Broadcast[int] = 1,
        pos_scale: Optional[float] = None,
        norm: Broadcast[tuple[bool, bool] | bool] = False,
        clamp: Broadcast[float] = 0.0,
        dropout: Broadcast[tuple[float, float]] | float = 0.0,
        dropout_edge: Broadcast[float] = 0.0,
        activation: Broadcast[Optional[nn.Module]] = None,
        init: Broadcast[Optional[Callable[[nn.Module, str], None]]] = None,
        layer_cls: Broadcast[nn.Module] = ReversibleEGNN,
        norm_cls: Broadcast[type[nn.Module]] = nn.BatchNorm1d,
        dropout_cls: Broadcast[type[nn.Module]] = nn.Dropout,
        disable: Broadcast[bool] = False,
    ) -> None:
        r"""
        Arguments:
            dims_node: Dimensions for the node networks. This can be a sequence
                of integers with `dims_node[0]` describing the number of input
                features and all remaining dimensions describing the number of
                features in the hidden layers. Note that the number of output
                features will always be the same as the number of input features
                for this variant of an EGNN model. Alternatively, `dims_node`
                can also be a sequence of sequences of integers, describing the
                aforementioned properties for each layer individually. In this
                case, only the number of input features in the definition of the
                first layer must be provided and for all other layers the number
                of input features is omitted, since it is derived from the
                the previous layer.
            dims_edge: Dimensions for the edge networks. Follows the same
                broadcasting scheme as `dims_node`.
            dims_gate: Dimensions for the message gating network. Follows the
                same broadcasting scheme as `dims_node`. If empty, message
                gating is disabled.
            dims_pos: Dimensions for the coordinate update network. Follows the
                same broadcasting scheme as `dims_node`. If empty, coordinate
                update is disabled.
            blocks: Number of blocks in the auto-regressive reversible layer;
                i.e., individual instances of :class:`bpp.model.egnn.EGNN`. At
                least two blocks are required and the number of node input
                features must be divisible by the number of blocks.
            layers: Number of layers.
            tied: Whether the same layer is repeated. Layer-wise arguments are
                not supported in this case.
            num_pos: Number of node positions.
            dim_pos: Number of coordinate dimensions.
            num_encode: Number of fourier-encoded features.
            pos_scale: Initial scaler for coordinate updates. If `pos_scale` is
                `None`, the penultimate layer of a coordinate update network
                will use a linear activation and omit the scaling factor.
            norm: Whether to use normalization within a block and whether to use
                normalization after a block. If a single boolean or a list of
                single booleans, then normalization will be applied within and
                after the corresponding blocks of a layer according to the
                boolean value. If a tuple of booleans or a list of tuples of
                booleans, then the first boolean will determine whether to use
                normalization after a block and the second will determine
                whether to use normalization within that block.
            clamp: Threshold for clamping coordinate updates (between `-clamp`
                and `clamp`). Unclamped if 0.
            dropout: Dropout probabilities for in-between layers and within
                layers. Values will be broadcasted to all layers, if only two
                probabilities are given, or to all layers and in-between and
                within layers, if only one probability is given. Dropout is
                omitted if 0.
            dropout_edge: Probability for edges to be dropped. No edges will be
                dropped if 0.
            activation: Activation function that is used for the :class:`EGNN`
                layers.
            init: Initializer that is used for network layers. First argument is
                the module that should be initialized, the second is the
                non-linearity, in case of a linear module, that should be used.
                The default non-linearity should adhere to whatever is used for
                `activation`. See :func:`parameter_init`.
            layer_cls: Class instance of the reversible layer.
            norm_cls: Class instance of normalization module.
            dropout_cls: Class instance of dropout module.
            disable: Whether to disable reversible computations and use
                conventional back-propagation method. Used primarily for
                debugging.
        """

        super().__init__()

        if not isinstance(dims_node[0], Sequence):
            features_node = dims_node[0]
            dims_node = dims_node[1:]
        else:
            features_node = dims_node[0][0]
            dims_node = [dims_node[0][1:], *dims_node[1:]]

        if tied:
            layer_instances = 1
            pos_is_broadcasted = not any(isinstance(dim, Sequence) for dim in dims_pos)
        else:
            layer_instances = layers
            pos_is_broadcasted = False

        dims_node = broadcast_arg("dims_node", dims_node, [layer_instances, None])
        dims_edge = broadcast_arg("dims_edge", dims_edge, [layer_instances, None])
        dims_gate = broadcast_arg("dims_gate", dims_gate, [layer_instances, None])
        dims_pos = broadcast_arg("dims_pos", dims_pos, [layer_instances, None])
        blocks = broadcast_arg("blocks", blocks, [layer_instances])
        num_encode = broadcast_arg("num_encode", num_encode, [layer_instances])
        pos_scale = broadcast_arg("pos_scale", pos_scale, [layer_instances])
        norm = broadcast_arg("norm", norm, [layer_instances])
        clamp = broadcast_arg("clamp", clamp, [layer_instances])
        dropout = broadcast_arg("dropout", dropout, [layer_instances, 2])
        dropout_edge = broadcast_arg("dropout_edge", dropout_edge, [layer_instances])
        activation = broadcast_arg("activation", activation, [layer_instances])
        init = broadcast_arg("init", init, [layer_instances])
        layer_cls = broadcast_arg("layer_cls", layer_cls, [layer_instances])
        norm_cls = broadcast_arg("norm_cls", norm_cls, [layer_instances])
        dropout_cls = broadcast_arg("dropout_cls", dropout_cls, [layer_instances])
        disable = broadcast_arg("disable", disable, [layer_instances])

        for i in range(len(dims_node)):

            if blocks[i] < 2:
                raise ValueError(
                    f"minimum number of blocks is 2, but got {blocks} for number "
                    f"of blocks"
                )

            if features_node % blocks[i] != 0:
                raise ValueError(
                    f"number of input features for a node must be divisible by "
                    f"the number of blocks, but got {features_node} for input "
                    f"features and {blocks[i]} for the number of blocks in "
                    f"layer {i}"
                )

            dims_node[i] = (
                features_node // blocks[i],
                *dims_node[i],
                features_node // blocks[i],
            )

        if pos_is_broadcasted:
            dims_pos[-1] = ()

        self.tied = tied
        self.layers = nn.ModuleList()

        for i in range(layers):

            if tied and i != 0:
                layer = self.layers[0]
            else:
                layer = layer_cls[i](
                    dims_node[i],
                    dims_edge[i],
                    dims_gate[i],
                    dims_pos[i],
                    blocks[i],
                    num_pos,
                    dim_pos,
                    num_encode[i],
                    pos_scale[i],
                    norm[i],
                    clamp[i],
                    dropout[i],
                    dropout_edge[i],
                    activation[i],
                    init[i],
                    norm_cls[i],
                    dropout_cls[i],
                    disable[i],
                    free_inputs=False,
                    free_outputs=False,
                )

            self.layers.append(layer)

    def reset_parameters(self):
        r"""Reset parameters."""

        for layer in self.layers:
            layer.reset_parameters()

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

        s = x
        pos_a = pos
        pos_b = pos

        for layer in self.layers:
            s, pos_a, pos_b = layer(
                s,
                pos_a,
                pos_b,
                edge_index,
                edge_attr,
                batch,
            )

        return s, pos_a
