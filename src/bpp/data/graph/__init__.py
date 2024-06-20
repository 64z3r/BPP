import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
from torch import FloatTensor, IntTensor, LongTensor
from graphein.protein import construct_graph
from graphein.protein.config import (
    AltLocsOpts,
    GranularityOpts,
    GraphAtoms,
    ProteinGraphConfig,
)
from networkx import Graph
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

from ..constants import ALLOWABLE_ELEMENTS, ALLOWABLE_RESIDUES
from .operator import (
    Operator,
    GraphOperator,
    NodeOperator,
    EdgeOperator,
    AnnotationKind,
)
from ._pdb import prepare_pdb
from . import annotation, construction

logger = logging.getLogger(__name__)

ELEMENT_LABEL_ENCODER = LabelEncoder().fit(ALLOWABLE_ELEMENTS)
RESIDUE_LABEL_ENCODER = LabelEncoder().fit(ALLOWABLE_RESIDUES)


def _assure_2d(a: ArrayLike, /) -> NDArray:
    """Similar to :fun:`numpy.atleast_2d` this will make sure that any array
    is at least two dimensional. However, unlike :func:`numpy.atleast_2d` this
    will add an additional dimension to the rightmost side.

    Arguments:
        a: An array-like object.
    
    Returns:
        Array with at least two dimensions.
    """

    if a.ndim < 2:
        a = np.atleast_2d(a).T

    return a


@dataclass
class Configuration:
    """ """

    granularity: GraphAtoms | GranularityOpts = "atom"

    insertions: bool = False
    deprotonate: bool = False
    exclude_waters: bool = True
    alternate_locations: AltLocsOpts = "max_occupancy"

    binding_site_cutoff: float = 4
    surface_cutoff: float | None = None

    elements: set[str] = field(
        default_factory=lambda: set(ALLOWABLE_ELEMENTS),
    )
    residues: set[str] = field(
        default_factory=lambda: set(ALLOWABLE_RESIDUES),
    )

    construction: dict[str, Operator] = field(
        default_factory=lambda: {
            "covalent": construction.edge.Covalent,
        },
    )

    annotation: dict[str, Operator] = field(
        default_factory=lambda: {
            "ligand_distances": annotation.node.LigandDistances,
            "surface_distance": annotation.node.SurfaceDistance,
        },
    )

    def __post_init__(self) -> None:

        self.construction = [
            operator(name) for name, operator in self.construction.items()
        ]
        self.annotation = [operator(name) for name, operator in self.annotation.items()]

    @property
    def state(self) -> dict[str, Any]:
        """Return hashable state of configuration.

        Returns:
            Hashable state.
        """

        state = {
            name: getattr(self, name)
            for name in [
                "granularity",
                "insertions",
                "deprotonate",
                "exclude_waters",
                "alternate_locations",
                "binding_site_cutoff",
                "surface_cutoff",
                "elements",
                "residues",
            ]
        }
        state["construction"] = [
            (operator.__class__.__module__, operator.__class__.__name__, vars(operator))
            for operator in self.construction
        ]
        state["annotation"] = [
            (operator.__class__.__module__, operator.__class__.__name__, vars(operator))
            for operator in self.annotation
        ]

        return state


class Featurizer:
    """ """

    def __init__(
        self,
        constructor: "Constructor",
        name: str,
        kind: AnnotationKind,
        getter: Callable[[Graph, str], dict[str, Any]],
    ) -> None:
        """ """

        self.constructor = constructor
        self.name = name
        self.kind = kind
        self.getter = getter

    def get(self, name: str) -> list[Any] | None:
        """ """

        for name in (name, f"_{name}"):
            attributes = self.getter(self.constructor.nxg, name)
            if attributes:
                return list(attributes.values())

    @cached_property
    def features(self) -> NDArray[np.float_]:
        """ """

        feature_names = []
        feature_values = []

        for operator in self.constructor.conf.annotation:
            if operator.name.startswith("_"):
                continue
            if operator.kind is self.kind:
                feature_names.append(operator.name)

        for name in sorted(feature_names):
            values = self.get(name)
            if values is not None:
                logger.debug(f"Adding {name} to {self.name} features")
                feature_values.append(_assure_2d(np.array(values)))

        if not feature_values:
            return np.array([])

        return np.concatenate(feature_values, axis=-1)


class NodeFeaturizer(Featurizer):
    """ """

    def __init__(self, constructor: "Constructor", name: str) -> None:
        """ """

        super().__init__(
            constructor,
            name,
            kind=AnnotationKind.NODE,
            getter=nx.get_node_attributes,
        )

    @cached_property
    def coords(self) -> NDArray[np.float_]:
        """ """

        coords = np.stack(self.get("coords"))
        return coords

    @cached_property
    def surface_distance(self) -> NDArray[np.float_]:
        """ """

        surface_distance = np.stack(self.get("surface_distance"))
        return surface_distance

    @cached_property
    def surface(self) -> NDArray[np.bool_]:
        """ """

        if self.constructor.conf.surface_cutoff is not None:
            surface = self.surface_distance <= self.constructor.conf.surface_cutoff
        else:
            surface = np.full_like(self.surface_distance, True)

        return surface

    @cached_property
    def binding_sites(self) -> NDArray[np.bool_]:
        """ """

        ligand_distances = np.stack(self.get("ligand_distances"))
        binding_sites = ligand_distances <= self.constructor.conf.binding_site_cutoff

        return binding_sites

    @cached_property
    def binding_site_centers(self) -> NDArray[np.float_]:
        """ """

        binding_site_coords = self.binding_sites[:, :, None] * self.coords[:, None, :]
        binding_site_centers = binding_site_coords.mean(axis=0)

        return binding_site_centers


class EdgeFeaturizer(Featurizer):
    """ """

    def __init__(self, constructor: "Constructor", name: str) -> None:
        """ """

        super().__init__(
            constructor,
            name,
            kind=AnnotationKind.EDGE,
            getter=nx.get_edge_attributes,
        )

    @cached_property
    def index(self) -> list[tuple[int, int]]:
        """ """

        return list(self.constructor.nxg.edges)

    @cached_property
    def kinds(self) -> list[str]:
        """ """

        kinds = [operator.name for operator in self.constructor.conf.construction]
        kinds.sort()

        return kinds

    @cached_property
    def labels(self) -> NDArray[np.bool_]:
        """ """

        edge_kinds = self.get("kind")
        edge_labels = np.full((len(edge_kinds), len(self.kinds)), False)

        for labels, kinds in zip(edge_labels, edge_kinds):
            slice = list(map(self.kinds.index, kinds))
            labels[slice] = True

        return edge_labels


class Constructor:
    """ """

    def __init__(self, path: str, conf: Configuration) -> None:
        """ """

        self.path = Path(path)
        self.conf = conf

    @cached_property
    def name(self) -> str:
        """Name of the sample that is processed."""

        return self.path.name

    @cached_property
    def node(self) -> NodeFeaturizer:
        """ """

        return NodeFeaturizer(self, "node")

    @cached_property
    def edge(self) -> EdgeFeaturizer:
        """ """

        return EdgeFeaturizer(self, "edge")

    @cached_property
    def nxg(self) -> Graph:
        """NetworkX graph.

        Intermediate structure from which the final graph is build.
        """

        edge_annotation = []
        node_annotation = []
        graph_annotation = []

        for operator in self.conf.annotation:
            match operator:
                case EdgeOperator():
                    edge_annotation.append(operator)
                case NodeOperator():
                    node_annotation.append(operator)
                case GraphOperator():
                    graph_annotation.append(operator)

        conf = ProteinGraphConfig(
            granularity=self.conf.granularity,
            insertions=self.conf.insertions,
            deprotonate=self.conf.deprotonate,
            exclude_waters=self.conf.exclude_waters,
            alt_locs=self.conf.alternate_locations,
            edge_construction_functions=self.conf.construction,
            edge_metadata_functions=edge_annotation,
            node_metadata_functions=node_annotation,
            graph_metadata_functions=graph_annotation,
        )

        with prepare_pdb(self.path) as pdb_path:
            graph = construct_graph(
                config=conf,
                name=self.name,
                path=pdb_path,
                verbose=False,
            )

        # nodes = []

        # for name, d in graph.nodes(data=True):
        #     if d["element_symbol"] not in self.conf.elements:
        #         continue
        #     if d["residue_name"] not in self.conf.residues:
        #         continue
        #     nodes.append(name)

        # graph = extract_subgraph_from_node_list(graph, nodes)

        graph = nx.convert_node_labels_to_integers(graph)
        graph = nx.to_directed(graph)

        return graph

    def get(self, name: str) -> Any:
        """ """

        return self.nxg.graph[name]

    def construct(self) -> Data:
        """ """

        meta = {
            "name": self.get("name"),
            "chain": self.node.get("chain_id"),
            "element_symbols": self.node.get("element_symbol"),
            "atom_names": self.node.get("atom_type"),
            "ligand_names": self.get("ligand_names"),
            "residue_names": self.node.get("residue_name"),
            "residue_numbers": self.node.get("residue_number"),
        }
        
        data = Data()
        
        data.element = IntTensor(
            ELEMENT_LABEL_ENCODER.transform(self.node.get("element_symbol"))
        )
        data.residue = IntTensor(
            RESIDUE_LABEL_ENCODER.transform(self.node.get("residue_name"))
        )
        data.pos = FloatTensor(self.node.coords)
        data.x = FloatTensor(self.node.features)
        data.y = FloatTensor(self.node.binding_sites)
        data.surface = FloatTensor(self.node.surface)

        data.edge_kind = FloatTensor(self.edge.labels)
        data.edge_attr = FloatTensor(self.edge.features)
        data.edge_index = LongTensor(self.edge.index).t().contiguous()

        return data, meta
