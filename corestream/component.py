from .edge import Edge
from .data_bubble import Vertex
from .abstract_graph import AbstractGraph

class Component(AbstractGraph):
    def __init__(self, startVertex: Vertex, graph: AbstractGraph, prepareEdges : bool):
        super().__init__()
        
        self.m_edges_summarized_by_weight = {}
        self.m_prepare_edges              = prepareEdges
        
        self.addVertex(startVertex)
        if graph.hasSelfLoop(startVertex):
            self.addEdge(startVertex, startVertex, graph.getEdge(startVertex, startVertex))
            
        self.build(startVertex, graph)

    def build(self, vertex: Vertex, graph: AbstractGraph):
        adjacentVertices = graph.adjacencyList(vertex).keys()
        
        for v in adjacentVertices:
            if not super().containsVertex(v):
                self.addVertex(v)
                
                if graph.hasSelfLoop(v):
                    self.addEdge(v, v, graph.getEdge(v, v))
                if not self.containsEdge(vertex, v):
                    self.addEdge(vertex, v, graph.getEdge(vertex, v))
                
                self.build(v, graph)

    def compareByVertices(self, other: "Component"):
        if self.numVertices() != other.numVertices():
            return False
        
        iterator = iter(self)
        
        for v in next(iterator):
            if v not in other.containsVertex(v):
                return False
        return True

    def buildGraph(self):
        pass
    
    def setMEdge(self, a):
        self.m_edges_summarized_by_weight = a
        
    def getMEdge(self):
        return self.m_edges_summarized_by_weight

    def split(self, e: Edge):
        self.removeEdge(e.getVertex1(), e.getVertex2())
        a   = Component(e.getVertex1(), self)
        b   = Component(e.getVertex2(), self)
        res = set()
        res.add(a)
        res.add(b)
        return res