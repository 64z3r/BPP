import torch
from typing import Sequence, Callable
from torch_geometric.data import Data

from .._tensor import assure_2d as _assure_2d


class Sequential:
    """Run multiple transformation in sequence."""

    def __init__(self, transforms: Sequence[Callable[[Data], Data]]) -> None:
        """
        Arguments:
            transforms: List of transformations.
        """

        self.transforms = transforms

    def __call__(self, data: Data) -> Data:
        """Execute transforms in sequence.

        Arguments:
            data: Data object.

        Returns:
            Updated data object.
        """

        for transform in self.transforms:
            data = transform(data)

        return data


class ConcatenateFeatures:
    """Concatenate extra node and edge attributes with their respective default
    feature attributes.

    **Note**: deletes the original node and edge attributes.
    """

    def __init__(
        self,
        node: list[str],
        edge: list[str],
    ) -> None:
        """
        Arguments:
            node: Attribute names of features that should be added to default
                node features.
            edge: Attribute names of features that should be added to default
                edge features.
        """

        self.node = node
        self.edge = edge

    def __call__(self, data: Data) -> Data:
        """Concatenate extra node and edge attributes.

        Argument:
            data: Data object containing node and edge attributes.

        Returns:
            Data object with concatenated node and edge attributes.
        """

        copy = data.clone(None)

        node = [_assure_2d(data.x)]
        edge = [_assure_2d(data.edge_attr)]

        for name in self.node:
            node.append(_assure_2d(getattr(data, name)))
            delattr(copy, name)

        for name in self.edge:
            edge.append(_assure_2d(getattr(data, name)))
            delattr(copy, name)

        copy.x = torch.cat(node, dim=-1).squeeze()
        copy.edge_attr = torch.cat(edge, dim=-1).squeeze()

        return copy


class CollapseTargets:
    """Collapses binding-site targets.

    A protein can have multiple binding-sites; this transformation collapses
    the multi-dimensional binding-site tensor into a one-dimensional tensor.
    """

    def __call__(self, data: Data) -> Data:
        """Collapse binding-site targets.

        Arguments:
            data: Data object containing binding-site targets.

        Returns:
            Data object with collapsed binding-site targets.
        """

        copy = data.clone(None)
        copy.y = data.y.any(axis=-1)

        return copy
