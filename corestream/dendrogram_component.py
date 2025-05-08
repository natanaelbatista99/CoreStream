from .node import Node
from .edge import Edge
from .data_bubble import Vertex
from .component import Component
from .abstract_graph import AbstractGraph

class DendrogramComponent(Component):
    def __init__(self, start_vertex: Vertex, graph: AbstractGraph, prepareEdges: bool):  
        
        super().__init__(start_vertex, graph, prepareEdges)
        
        self.m_set_of_highest_weighted_edges = set()
        self.m_prepare_edges = prepareEdges
        self.m_node = None

        self.addVertex(start_vertex)
        
        if graph.hasSelfLoop(start_vertex):
            self.addEdge(start_vertex, start_vertex, graph.getEdge(start_vertex, start_vertex))
        
        self.build(start_vertex, graph)
         

    def build(self, vertex: Vertex, graph: AbstractGraph):
        adjacent_vertices = graph.adjacencyList(vertex).keys()
        
        for v in adjacent_vertices:
            if not self.containsVertex(v):
                self.addVertex(v)

                if graph.hasSelfLoop(v):
                    self.addEdge(v, v, graph.getEdge(v, v))

                if not self.containsEdge(vertex, v):
                    edge = graph.getEdge(vertex, v)

                    self.addEdge(vertex, v, edge)

                    w = edge.getWeight()
                    if isinstance(w, Edge):
                        w = (edge.getWeight()).getWeight()

                    if self.m_prepare_edges:
                        
                        if w not in self.m_edges_summarized_by_weight:
                            self.m_edges_summarized_by_weight[w] = set()
                            

                        self.m_edges_summarized_by_weight[w].add(edge)
                
                self.build(v, graph)
        
    def setHeighestWeightedEdges(self):        
        if self.m_prepare_edges:
            highest = -1.0
            
            for weight in self.m_edges_summarized_by_weight.keys():
                if weight > highest:
                    highest = weight
            
            if highest == -1:
                self.m_set_of_highest_weighted_edges = None
            else:        
                self.m_set_of_highest_weighted_edges = self.m_edges_summarized_by_weight[highest]
                del self.m_edges_summarized_by_weight[highest]

    def getNextSetOfHeighestWeightedEdges(self):
        if self.m_set_of_highest_weighted_edges is None or len(self.m_set_of_highest_weighted_edges) == 0:
            self.setHeighestWeightedEdges()
        
        res = self.m_set_of_highest_weighted_edges
        self.setHeighestWeightedEdges()  # prepare next step
        
        return res

    def splitComponent(self, e: Edge):
        self.removeEdge(e.getVertex1(), e.getVertex2())

        a = DendrogramComponent(e.getVertex1(), self, False)
        b = DendrogramComponent(e.getVertex2(), self, False)

        res = {a, b}
        return res

    def extendWithSelfEdges(self):
        for v in self.getVertices():
            self_loop = Edge(v, v, v.getCoreDistance())
            self.addEdge(v, v, self_loop)

            w = self_loop.getWeight()
            if w not in self.m_edges_summarized_by_weight:
                self.m_edges_summarized_by_weight[w] = set()

            self.m_edges_summarized_by_weight[w].add(self_loop)

    def setNodeRepresentitive(self, node: Node):
        self.m_node = node

    def getNode(self):
        return self.m_node
    
    def getMEdge(self):
        return self.m_edges_summarized_by_weight

    def String(self):
        sb = []

        for v in self.get_vertices():
            sb.append(str(v))

        return f"[{''.join(sb)}]"