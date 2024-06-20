from typing import Iterable, Literal
from graphein import protein

from ..operator import EdgeConstructorFunctionOperator
from ..operator import GraphFunction


class AromaticSulphur(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_aromatic_sulphur_interactions
    id: str = "aromatic_sulphur"


class Aromatic(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_aromatic_interactions
    id: str = "aromatic"


class CationPi(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_cation_pi_interactions
    id: str = "cation_pi"


class Covalent(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.atomic.add_atomic_edges
    id: str = "covalent"

    tolerance: float = 0.56


class Delaunay(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_delaunay_triangulation
    id: str = "delaunay"

class Disulfide(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_disulfide_interactions
    id: str = "disulfide"


class HydrogenBond(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_hydrogen_bond_interactions
    id: str = "hbond"


class Ionic(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_ionic_interactions
    id: str = "ionic"


class KNN(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_k_nn_edges
    id: str = "knn"

    long_interaction_threshold: int = 30
    k: int = 10
    exclude_edges: Iterable[Literal["inter", "intra"]] = ()
    exclude_self_loops: bool = True


class PeptideBond(EdgeConstructorFunctionOperator):

    function: GraphFunction = protein.edges.distance.add_peptide_bonds
    id: str = "peptide_bond"