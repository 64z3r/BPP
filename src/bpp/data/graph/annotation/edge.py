from networkx import Graph

from ..operator import GraphOperator
from ..operator import AnnotationKind

class EdgeAngle(GraphOperator):
    """ """

    kind: AnnotationKind = AnnotationKind.EDGE 

    def __init__(self, name: str, name_vec_i: str, name_vec_j: str) -> None:
        """ """

        super().__init__(name)
        self.name_vec_i = name_vec_i
        self.name_vec_j = name_vec_j
    
    def __call__(self, G: Graph) -> None:
        """ """

        for node_i, node_j, d in G.edges(data=True):
            vec_i = G.nodes[node_i][self.name_vec_i]
            vec_j = G.nodes[node_j][self.name_vec_j]
            angle = vec_i @ vec_j
            d[self.name] = angle
    