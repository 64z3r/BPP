import logging
import numpy as np
import pandas as pd

from pathlib import Path
from networkx import Graph
from graphein import protein
from Bio.PDB import PDBParser

from ..operator import (
    NodeAnnotationFunctionOperator,
    GraphAnnotationFunctionOperator,
    NodeFunction,
    GraphFunction,
    GraphOperator,
    AnnotationKind,
)
from ._distance import (
    pairwise_min_distance as _pairwise_min_distance,
)
from ._ligand import (
    iter_load_ligands as _iter_load_ligands,
    iter_mol_coords as _iter_mol_coords,
)
from ._surface import (
    get_surface as _get_surface,
)

logger = logging.getLevelName(__name__)


class Expasy(NodeAnnotationFunctionOperator):

    function: NodeFunction = protein.features.nodes.amino_acid.expasy_protein_scale
    id: str = "expasy"

    selection: list[str] | None = None


class HydrogenBondAcceptors(NodeAnnotationFunctionOperator):

    function: NodeFunction = protein.features.nodes.amino_acid.hydrogen_bond_acceptor
    id: str = "hbond_acceptors"

    sum_features: bool = True


class HydrogenBondDonors(NodeAnnotationFunctionOperator):

    function: NodeFunction = protein.features.nodes.amino_acid.hydrogen_bond_donor
    id: str = "hbond_donors"

    sum_features: bool = True


class Meiler(NodeAnnotationFunctionOperator):

    function: NodeFunction = protein.features.nodes.amino_acid.meiler_embedding
    id: str = "meiler"


class BetaCarbonVector(GraphAnnotationFunctionOperator):

    function: GraphFunction = protein.features.nodes.geometry.add_beta_carbon_vector
    id: str = "c_beta_vector"
    kind: AnnotationKind = AnnotationKind.NODE

    scale: bool = True
    reverse: bool = False


class SequenceNeighbourVector(GraphAnnotationFunctionOperator):

    function: GraphFunction = (
        protein.features.nodes.geometry.add_sequence_neighbour_vector
    )
    id: str = "sequence_neighbour_vector"
    kind: AnnotationKind = AnnotationKind.NODE

    def __init__(
        self,
        name: str,
        n_to_c: bool = False,
        scale: bool = True,
        reverse: bool = False,
    ) -> None:
        """ """

        super().__init__(
            name,
            n_to_c=n_to_c,
            scale=scale,
            reverse=reverse,
        )
        self.id = self.id + ("_n_to_c" if n_to_c else "_c_to_n")


class SidechainVector(GraphAnnotationFunctionOperator):

    function: GraphFunction = protein.features.nodes.geometry.add_sidechain_vector
    id: str = "sidechain_vector"
    kind: AnnotationKind = AnnotationKind.NODE

    scale: bool = True
    reverse: bool = False


class NodeAngle(GraphOperator):
    """ """

    def __init__(self, name: str, name_vec_i: str, name_vec_j: str) -> None:
        """ """

        super().__init__(name)
        self.name_vec_i = name_vec_i
        self.name_vec_j = name_vec_j

    def __call__(self, G: Graph) -> None:
        """ """

        for _, d in G.nodes(data=True):
            vec_i = d[self.name_vec_i]
            vec_j = d[self.name_vec_j]
            angle = vec_i @ vec_j
            d[self.name] = angle


class LigandDistances(GraphOperator):
    """ """

    def __init__(
        self,
        name: str = "ligand_distances",
        store_names_as: str | None = "ligand_names",
        exclude_waters: bool = True,
    ) -> None:
        """ """

        super().__init__(name)
        self.store_names_as = store_names_as
        self.exclude_waters = exclude_waters

    def __call__(self, G: Graph) -> None:
        """ """

        ligands = list(
            _iter_load_ligands(
                Path(G.graph["path"]).parent,
                removeHs=self.exclude_waters,
            )
        )

        df = G.graph["raw_pdb_df"]

        atom_coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()

        ligand_atom_distances = pd.DataFrame(index=df.node_id)

        for ligand in ligands:
            coords = np.array(list(_iter_mol_coords(ligand)))
            name = ligand.GetProp("Name")
            distances = _pairwise_min_distance(atom_coords, coords)
            ligand_atom_distances[name] = distances

        for id, d in G.nodes(data=True):
            distances = ligand_atom_distances.loc[id].to_numpy()
            d[self.name] = np.atleast_2d(distances).min(axis=0)

        if self.store_names_as is not None:
            G.graph[self.store_names_as] = ligand_atom_distances.columns.to_list()


class SurfaceDistance(GraphOperator):
    """ """

    def __init__(self, name: str = "surface_distance") -> None:
        """ """

        super().__init__(name)

    def __call__(self, G: Graph) -> None:
        """ """

        parser = PDBParser()
        struct = parser.get_structure("protein", G.graph["path"])[0]
        surface_coords = _get_surface(struct)

        df = G.graph["raw_pdb_df"]

        atom_coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()

        surface_atom_distances = pd.Series(
            _pairwise_min_distance(atom_coords, surface_coords),
            index=df.node_id,
        )

        for id, d in G.nodes(data=True):
            distances = surface_atom_distances.loc[id]
            d["surface_distance"] = np.atleast_1d(distances).min()
