import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Protocol

from networkx import Graph

logger = logging.getLogger(__name__)

class AnnotationKind(Enum):
    """ """

    NONE = 1
    NODE = 2
    EDGE = 4
    GRAPH = 8


class Operator(ABC):
    """ """

    kind: AnnotationKind = AnnotationKind.NONE

    def __init__(self, name: str) -> None:
        """ """

        self.name = name


class GraphOperator(Operator):
    """ """

    @abstractmethod
    def __call__(self, G: Graph) -> None:
        """ """


class NodeOperator(Operator):
    """ """

    kind: AnnotationKind = AnnotationKind.NODE

    @abstractmethod
    def __call__(self, n: str, d: dict[str, Any]) -> None:
        """ """


class EdgeOperator(Operator):
    """ """

    kind: AnnotationKind = AnnotationKind.EDGE

    @abstractmethod
    def __call__(self, u: str, v: str, d: dict[str, Any]) -> None:
        """ """


class GraphFunction(Protocol):
    """ """

    def __call__(
        self,
        g: Graph,
        **kwargs: dict[str, Any],
    ) -> Any: ...


class NodeFunction(Protocol):
    """ """

    def __call__(
        self,
        n: str,
        d: dict[str, Any],
        **kwargs: dict[str, Any],
    ) -> Any: ...


class EdgeFunction(Protocol):
    """ """

    def __call__(
        self,
        u: str,
        v: str,
        d: dict[str, Any],
        **kwargs: dict[str, Any],
    ) -> Any: ...


class FunctionOperatorMetaClass(type(Operator)):
    """ """

    def __new__(meta_cls, name, bases, attrs) -> type:

        exclude_args = set()
        default_args = {}

        for base in bases:
            for parent in base.mro():
                if issubclass(parent, Operator):
                    exclude_args.update(parent.__annotations__)

        for k, v in attrs.items():
            if not k.startswith("_") and k not in exclude_args:
                default_args[k] = v

        attrs["default_args"] = default_args

        if "function" in attrs:
            attrs["function"] = staticmethod(attrs["function"])

        return super().__new__(meta_cls, name, bases, attrs)


class FunctionOperator(Operator, metaclass=FunctionOperatorMetaClass):
    """ """

    function: Callable[..., None]
    id: str

    def __init__(self, name: str, **extra_args: dict[str, Any]) -> None:
        """ """

        super().__init__(name)
        self.extra_args = dict(self.default_args)
        self.extra_args.update(extra_args)


class EdgeConstructorFunctionOperator(FunctionOperator, GraphOperator):
    """ """

    function: GraphFunction

    def __call__(self, G: Graph) -> None:

        self.function(G, **self.extra_args)

        if self.id != self.name:
            self._rename_edge_kind(G)

    def _rename_edge_kind(self, G: Graph) -> None:
        """ """

        for u, v, d in G.edges(data=True):
            if "kind" in d:
                kinds = d["kind"]
                if self.id in kinds:
                    kinds.remove(self.id)
                    kinds.add(self.name)
            else:
                logger.warning(
                    f"Edge ({u}, {v}) has no attribute named kind, "
                    f"cannot rename edge."
                )


class GraphAnnotationFunctionOperator(FunctionOperator, GraphOperator):
    """ """

    function: GraphFunction

    def __call__(self, G: Graph) -> None:

        self.function(G, **self.extra_args)

        if self.id != self.name:
            match self.kind:
                case AnnotationKind.GRAPH:
                    self._rename_graph_attribute(G)
                case AnnotationKind.NODE:
                    self._rename_node_attribute(G)
                case AnnotationKind.EDGE:
                    self._rename_edge_attribute(G)

    def _rename_graph_attribute(self, G: Graph) -> None:
        """ """

        try:
            G.graph[self.name] = G.graph.pop(self.id)
        except KeyError:
            logger.warning(
                f"Graph has no attribute named {self.id}, "
                f"cannot rename to {self.name}."
            )

    def _rename_node_attribute(self, G: Graph) -> None:
        """ """

        for n, d in G.nodes(data=True):
            try:
                d[self.name] = d.pop(self.id)
            except KeyError:
                logger.warning(
                    f"Node {n} has no attribute named {self.id}, "
                    f"cannot rename to {self.name}."
                )

    def _rename_edge_attribute(self, G: Graph) -> None:
        """ """

        for u, v, d in G.edges(data=True):
            try:
                d[self.name] = d.pop(self.id)
            except KeyError:
                logger.warning(
                    f"Edge ({u}, {v}) has no attribute named {self.id}, "
                    f"cannot rename to {self.name}."
                )


class NodeAnnotationFunctionOperator(FunctionOperator, NodeOperator):
    """ """

    function: NodeFunction

    def __call__(self, n: str, d: dict[str, Any]) -> None:

        self.function(n, d, **self.extra_args)

        if self.id != self.name:
            self._rename_node_attribute(n, d)

    def _rename_node_attribute(self, n: str, d: dict[str, Any]) -> None:
        """ """

        try:
            d[self.name] = d.pop(self.id)
        except KeyError:
            logger.warning(
                f"Node {n} has no attribute named {self.id}, "
                f"cannot rename to {self.name}."
            )


class EdgeAnnotationFunctionOperator(FunctionOperator, EdgeOperator):
    """ """

    function: EdgeFunction

    def __call__(self, u: str, v: str, d: dict[str, Any]) -> None:

        self.function(u, v, d, **self.extra_args)

        if self.id != self.name:
            self._rename_edge_attribute(u, v, d)

    def _rename_edge_attribute(self, u: str, v: str, d: dict[str, Any]) -> None:
        """ """

        try:
            d[self.name] = d.pop(self.id)
        except KeyError:
            logger.warning(
                f"Edge ({u}, {v}) has no attribute named {self.id}, "
                f"cannot rename to {self.name}."
            )

