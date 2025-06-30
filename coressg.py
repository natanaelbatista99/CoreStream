import river
import copy
import math
import typing
import time
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys

from river import stream
from abc import ABCMeta
from collections import defaultdict, deque
from sklearn.neighbors import NearestNeighbors
from river import base, utils
from sklearn.preprocessing import MinMaxScaler

class Vertex():

    s_idCounter = 0

    def __init__(self, db , id=None):
        self.m_id = id if id is not None else Vertex.s_idCounter
        Vertex.s_idCounter += 1
        self.m_db = db
        if self.m_db is not None:
            self.m_db.setVertexRepresentative(self)
        self.m_visited = False
        self.m_coreDistanceObject = None
        self.m_lrd = -1
        self.m_coreDist = 0
        
    def getDataBubble(self):
        return self.m_db

    def getCoreDistance(self):
        if self.m_coreDist == 0:
            return -1
        return self.m_coreDist

    def setCoreDist(self, coreDistValue):
        self.m_coreDist = coreDistValue

    def setCoreDistance(self, coreDistObj):
        self.m_coreDistanceObject = coreDistObj

    def String(self):
        return f"({self.m_db.getRep()})"

    def getGraphVizVertexString(self):
        return f"vertex{self.m_id}"

    def getGraphVizString(self):
        return f"{self.getGraphVizVertexString()} [label='{self}';cdist={self.getCoreDistance()}]"

    def getDistanceToVertex(self, other):
        return self.m_db.getCenterDistance(other.getDataBubble())

    def getDistance(self, vertex):
        if self.m_db.getStaticCenter() is None or vertex.getDataBubble().getStaticCenter() is None:
            return self.getDistanceToVertex(vertex)
        
        x1 = self.distance(self.m_db.getRep(), vertex.getDataBubble().getRep()) - (self.m_db.getExtent() + vertex.getDataBubble().getExtent())
        x2 = self.m_db.getNnDist(1)
        x3 = vertex.getDataBubble().getNnDist(1)
        
        if x1 >= 0:
            return x1 + x2 + x3
        
        return max(x2, x3)

    def distance(self, v1, v2):
        distance = 0
        for i in range(len(v1)):
            d = v1[i] - v2[i]
            distance += d * d
        return math.sqrt(distance)

    def setCoreDistChanged(self):
        self.m_changeCoreDist = True

    def resetCoreDistChanged(self):
        self.m_changeCoreDist = False

    def hasCoreDistChanged(self):
        return self.m_changeCoreDist

    def visited(self):
        return self.m_visited

    def setVisited(self):
        self.m_visited = True

    def resetVisited(self):
        self.m_visited = False;

    def getID(self):
        return self.m_id

    def compareID(self,  other: "Vertex"):
        return self.m_id == other.m_id

class DataBubble(metaclass=ABCMeta):
  
    def __init__(self, x, timestamp, decaying_factor):

        self.x               = x
        self.last_edit_time  = timestamp
        self.creation_time   = timestamp
        self.decaying_factor = decaying_factor

        self.N              = 1
        self.linear_sum     = x
        self.squared_sum    = {i: (x_val * x_val) for i, x_val in x.items()}
        self.m_staticCenter = len(self.linear_sum);

    def calc_norm_cf1_cf2(self):
        # |CF1| and |CF2| in the paper
        x1     = 0
        x2     = 0
        res    = 0
        weight = self._weight()
        
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            val_ss = self.squared_sum[key]
            
            x1 = 2 * val_ss * weight
            x2 = 2 * val_ls**2
            
            diff = (x1 - x2) / (weight * (weight - 1))
            
            res += math.sqrt(diff) if diff > 0 else 0
            
        # return |CF1| and |CF2|
        return res
    
    def getN(self):
        return self.N

    def calc_weight(self):
        return self._weight()

    def _weight(self):
        return self.N
        
    def getRep(self):  
        weight = self._weight()
        center = {key: (val) / weight for key, val in self.linear_sum.items()}
        
        return center

    def getExtent(self):        
        res = self.calc_norm_cf1_cf2()
        
        return res

    def insert(self, x):
        self.N += 1
        
        for key, val in x.items():
            self.linear_sum[key]  += val
            self.squared_sum[key] += val * val
            

    def merge(self, cluster):
        self.N += cluster.N
        for key in cluster.linear_sum.keys():
            try:
                self.linear_sum[key] += cluster.linear_sum[key]
                self.squared_sum[key] += cluster.squared_sum[key]
            except KeyError:
                self.linear_sum[key] = cluster.linear_sum[key]
                self.squared_sum[key] = cluster.squared_sum[key]
        if self.last_edit_time < cluster.creation_time:
            self.last_edit_time = cluster.creation_time

    def getNnDist(self, k):
        return ((k/self.N)**(1.0/len(self.linear_sum)))*self.getExtent()

    def fading_function(self, time):
        return 2 ** (-self.decaying_factor * time)
    
    def setVertexRepresentative(self, v : Vertex): 
        self.m_vertexRepresentative = v

    def getVertexRepresentative(self):
        return self.m_vertexRepresentative
    
    def getStaticCenter(self):
        return self.m_staticCenter

    def setStaticCenter(self):
        m_static_center = self.getRep().copy()
        return m_static_center


class Neighbour():
    def __init__(self, vertex = Vertex, dist=None):
        if dist is not None:
            self.m_coreDist = dist
        if vertex is not None:
            self.m_vertex = vertex
            self.m_coreDist = dist

    def getDistance(self):
        return self.m_coreDist

    def String(self):
        return "value = {:.2f}".format(self.m_coreDist)

    def getVertex(self):
        return self.m_vertex
    
    def setCoredist(self, nn_dist):
        self.m_coreDist += nn_dist


class Edge():

    def __init__(self, v1 : Vertex, v2 : Vertex, dist : float):
        self.m_vertex1 = v1
        self.m_vertex2 = v2
        self.m_weight = dist
    
    def __str__(self):
        return str(self.m_weight)

    def compareTo(self, other):
        return self.m_weight < other.m_weight

    def getWeight(self):
        return self.m_weight

    def getVertex1(self):
        return self.m_vertex1

    def getVertex2(self):
        return self.m_vertex2

    def getAdjacentVertex(self, v):
        if v != self.m_vertex1:
            return self.m_vertex1
        else:
            return self.m_vertex2

    def setVertex1(self, v : Vertex):
        self.m_vertex1 = v

    def setVertex2(self, v : Vertex):
        self.m_vertex2 = v

    def setVertices(self, v1, v2):
        self.m_vertex1 = v1
        self.m_vertex2 = v2

    def graphVizString(self):
        return "M.add_edge(\"" + self.m_vertex1.getGraphVizVertexString() + "\",\"" + self.m_vertex2.getGraphVizVertexString() + "\",weight= " + str(self.m_weight) +")"

    def setEdgeWeight(self, weight):
        self.m_weight = weight

class AbstractGraph():
    def __init__(self):
        self.m_graph = {}
        self.m_globalIDCounter = 0

    def addVertex(self, vertex):
        if vertex in self.m_graph:
            return False
        self.m_graph[vertex] = {}
        return True

    def addEdge(self, vertex1 : Vertex, vertex2: Vertex, edge_weight):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise Exception("One vertex or both are missing")
        
        edge = None    
        
        
        for key, value in self.m_graph[vertex2].items():
            if key == vertex1:
                edge = Edge(vertex1, vertex2, edge_weight)
                break
        
        if edge is None:
            edge = Edge(vertex1, vertex2, edge_weight)
        
        self.addEdge1(vertex1, vertex2, edge)

    def addEdge1(self, vertex1, vertex2, edge : Edge):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise Exception("One vertex or both are missing")
        
        self.m_graph[vertex1][vertex2] = edge
        self.m_graph[vertex2][vertex1] = edge

    def removeEdge(self, vertex1, vertex2):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise Exception("One vertex or both are missing")
        
        del(self.m_graph[vertex1][vertex2])
        del(self.m_graph[vertex2][vertex1])

    def removeEdge2(self, edge):
        self.removeEdge(edge.getVertex1(), edge.getVertex2())

    def removeVertex(self, vertex):
        del self.m_graph[vertex]

    def buildGraph(self):
        pass

    def getEdge(self, vertex1, vertex2):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise Exception("One vertex or both are missing")
        
        for v,w in self.adjacencyList(vertex1).items():
            if v == vertex2:
                return w
        
        return None

    def getVertices(self):
        return self.m_graph.keys()

    def getEdges(self):
        edges = set()
        for v in self.getVertices():
            for e in self.adjacencyList(v).values():
                edges.add(e)
        return edges

    def getAdjacentEdges(self, vertex):
        return self.m_graph[vertex]
    
    def containsVertex(self, vertex):
        return vertex in self.m_graph

    def containsEdge(self, vertex1, vertex2):
        if not self.containsVertex(vertex1) or not self.containsVertex(vertex2):
            raise Exception("One vertex or both are missing")
        for v in self.adjacencyList(vertex1).keys():
            if v == vertex2:
                return True                
        return False
    def containsEdge2(self, edge : Edge):
        if (self.containsVertex(edge.getVertex1()) and self.containsVertex(edge.getVertex2())):
            return self.containsEdge(edge.getVertex1(), edge.getVertex2())
        return False

    def __iter__(self):
        return iter(self.m_graph)

    def numVertices(self):
        return len(self.m_graph)

    def isEmpty(self):
        return not bool(self.m_graph)

    def getNextID(self):
        self.m_global_id_counter += 1
        return self.m_global_id_counter

    def adjacencyList(self, vertex):
        return self.m_graph[vertex]

    def getGraphVizString(self):
        edges = set()

        vertices = sorted(self.m_graph, key=lambda x: x.id)

        sb = []
        sb.append("graph {\n")

        for v in vertices:
            sb.append("\t" + v.get_graph_viz_string() + "\n")
            edges.update(self.adjacency_list(v).values())

        edges_sorted = sorted(edges, key=lambda x: (x.v1.id, x.v2.id))

        for e in edges_sorted:
            sb.append("\t" + e.graph_viz_string() + "\n")

        sb.append("}")
        return "".join(sb)

    def getAdjacencyMatrixAsArray(self):
        matrix = [[0.0 for _ in range(len(self.m_graph))] for _ in range(len(self.m_graph))]
        df = "{:.4f}"

        sorted_by_id = sorted(self.m_graph, key=lambda x: x.id)

        for row in range(len(sorted_by_id)):
            for column in range(len(sorted_by_id)):
                v1 = sorted_by_id[row]
                v2 = sorted_by_id[column]
                edge = self.m_graph[v1].get_edge_to(v2)
                if edge:
                    matrix[row][column] = edge.weight

        return matrix

    def extendWithSelfEdges(self):
        for v in self.m_graph:
            self_loop = Edge(v, v, v.getCoreDistance())
            self.addEdge(v, v, self_loop)
    
    def controlNumEdgesCompleteGraph(self):
        vertex_iterator = iter(self)
        edges = set()
        for v in vertex_iterator:
            edges.update(self.adjacencyList(v).values())
        return len(edges) == int(self.numVertices() * (self.numVertices() - 1) / 2)
    
    #aqui pode ter um possível erro preciso revisar
    def hasSelfLoop(self, vertex: Vertex):
        if vertex not in self.m_graph:
            return Exception("Vertex does not exist!")
        return  vertex in self.adjacencyList(vertex).keys()


        
class MutualReachabilityGraph():
    def __init__(self, G, dbs : DataBubble, minPts):
        super().__init__()
        self.m_minPts = minPts
        self.G = G

        for db in dbs:
            v = Vertex(db)
            db.setVertexRepresentative(v)
            self.G.add_node(v)

        self.knng = KNearestNeighborsGraph(G)
        
        start = time.time()
        print("\nComputando coreDistanceDB")
        self.computeCoreDistance(G, minPts)
        end = time.time()
        print(">tempo para computar coreDistanceDB",end - start, end=' ')

    def getKnngGraph(self):
        return self.knng

       
    def buildGraph(self):
        for v1 in self.G:            
            for v2 in self.G:
                if v1 != v2:
                    mrd = self.getMutualReachabilityDistance(v1, v2)
                    self.G.add_edge(v1, v2, weight = mrd)                       
            
            

    def computeCoreDistance(self, vertices, minPts):
        for current in vertices:
            if current.getDataBubble()._weight() >= minPts:
                current.setCoreDist(current.getDataBubble().getNnDist(minPts))
            else:
                neighbours = self.getNeighbourhood(current, vertices)
                countPoints = current.getDataBubble().getN()
                neighbourC = None

                for n in neighbours:
                    countPoints += n.getVertex().getDataBubble().getN()
                    if self.knng.getEdge(current, n.getVertex()) is None:
                        self.knng.setEdge(current, n.getVertex())
                    if countPoints >= minPts:
                        neighbourC = n
                        break

                countPoints -= neighbourC.getVertex().getDataBubble().getN()
                current.setCoreDist(current.getDistance(neighbourC.getVertex()) + neighbourC.getVertex().getDataBubble().getNnDist(minPts - countPoints))

    def getNeighbourhood(self, vertex, vertices):
        neighbours = []
        for v in vertices:
            if v != vertex:
                neighbour = Neighbour(v, vertex.getDistance(v))
                neighbours.append(neighbour)
        neighbours.sort(key=lambda x: x.getDistance(), reverse=False)
        return neighbours

    def getMutualReachabilityDistance(self, v1, v2):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))

class MinimalSpaningTree(AbstractGraph):
    def __init__(self, graph):
        super().__init__()
        self.m_inputGraph = graph

    def buildGraph(self):
        for i, (u,v,w) in enumerate(self.m_inputGraph.edges(data='weight')):
            self.addVertex(u)
            self.addVertex(v)
            self.addEdge(u,v,w)
    
    def getEdgeWithMinWeight(self, available):
        fromVertex = toVertex = edge = None
        dist = float('inf')
        
        for v in available:            
            for e in self.m_inputGraph.adjacencyList(v).values():
                other = e.getAdjacentVertex(v)
                if e.getWeight() < dist and other not in available:
                    fromVertex = v
                    toVertex = other
                    edge = e
                    dist = e.getWeight()

        return fromVertex, toVertex, edge

    @staticmethod
    def getEmptyMST():
        return MinimalSpaningTree(None)

    def getTotalWeight(self):
        edges = set()
        for v in self.m_graph.getVertices():
            for e in self.adjacencyList[v].values():
                edges.add(e)

        res = 0
        for e in edges:
            res += e.getWeight()
        return res


class KNearestNeighborsGraph(AbstractGraph):
    def __init__(self, vertices):
        super().__init__()
        for v in vertices:
            super().addVertex(v)
        
    def setEdge(self, v1, v2):
        distance = v1.getDistance(v2)
        super().addEdge(v1, v2, distance)

    def buildGraph(self):
        pass


class CoreSG(AbstractGraph):
    def __init__(self, mst: MinimalSpaningTree, knng: KNearestNeighborsGraph):
        super().__init__()
        for v in mst.getVertices():
            super().addVertex(v)
        self.addKnng(knng)
        self.addMst(mst)

    def buildGraph(self):
        pass
    
    def getGraphNetworkx(self):
        G = nx.Graph()
        for e in self.getEdges():
            v1 = e.getVertex1()
            v2 = e.getVertex2()            
            G.add_node(v1)
            G.add_node(v2)
            G.add_edge(v1, v2, weight = e.getWeight())
        return G
            

    def addKnng(self, knng: KNearestNeighborsGraph):
        edges = knng.getEdges()        
        for e in edges:
            self.addEdge(e.getVertex1(), e.getVertex2(), e.getWeight())

    def addMst(self, mst: MinimalSpaningTree):
        for e in mst.getEdges():
            if self.getEdge(e.getVertex1(), e.getVertex2()) is None:
                self.addEdge(e.getVertex1(), e.getVertex2(), e.getWeight())

    def computeHierarchieMinPts(self, minPts: int):
        self.computeCoreDistance(minPts)
        edgesGraph = self.getEdges()
        for e in edgesGraph:
            self.removeEdge2(e)
            self.addEdge(e.getVertex1(), e.getVertex2(), self.getMutualReachabilityDistance(e.getVertex1(), e.getVertex2()))

    def computeCoreDistance(self, minPts: int):
        vertices = self.getVertices()
        for current in vertices:
            if current.getDataBubble().getN() >= minPts:
                nnDist = current.getDataBubble().getNnDist(minPts)
                current.setCoreDist(nnDist)
            else:
                neighbours = self.getNeighbourhoodMinPtsNN(current)
                countPoints = current.getDataBubble().getN()
                neighbourC = None
                for n in neighbours:
                    countPoints += n.getVertex().getDataBubble().getN()
                    if countPoints >= minPts:
                        neighbourC = n
                        break
                countPoints -= neighbourC.getVertex().getDataBubble().getN()
                current.setCoreDist(neighbourC.getDistance() + neighbourC.getVertex().getDataBubble().getNnDist(minPts - countPoints))

    def getNeighbourhoodMinPtsNN(self, vertex):
        neighbours = []
        vertices = self.getAdjacentEdges(vertex).keys()
        for v in vertices:
            if v != vertex:
                neighbour = Neighbour(v, vertex.getDistance(v))
                neighbours.append(neighbour)
        neighbours.sort(key=lambda x: x.getDistance(), reverse=False)
        return neighbours

    def getMutualReachabilityDistance(self, v1: Vertex, v2: Vertex):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))


class Updating:
    def __init__(self, mrg: MutualReachabilityGraph, mst : MinimalSpaningTree, csg : CoreSG):
        self.m_mrg = mrg
        self.m_mst = mst
        self.m_csg = csg
        self.m_globalReplacementEdge = None
    
    def getMST(self):
        return self.m_mst
    
    def getMRG(self):
        return self.m_mrg
    
    def getCSG(self):
        return self.m_csg

class Component(AbstractGraph):
    def __init__(self, startVertex: Vertex, graph: AbstractGraph, prepareEdges : bool):
        super().__init__()
        
        self.m_edges_summarized_by_weight = {}
        self.m_prepare_edges = prepareEdges
        
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
        a = Component(e.getVertex1(), self)
        b = Component(e.getVertex2(), self)
        res = set()
        res.add(a)
        res.add(b)
        return res

class Node:
    s_label = 0

    def __init__(self, c):
        self.m_vertices = set(c)
        self.m_children = []
        self.m_delta = True
        self.m_label = Node.s_label
        Node.s_label += 1
        self.m_parent = None
        self.m_scaleValue = 0
        

    def computeStability(self) -> float:
        if self.m_parent is None:
            return float('nan')

        eps_max = self.m_parent.m_scaleValue
        eps_min = self.m_scaleValue        
            
        
        self.m_stability = len(self.m_vertices) * ((1 / eps_min) - (1 / eps_max))

        return self.m_stability

    def addChild(self, child: "Node"):
        self.m_children.append(child)

    def getChildren(self):
        return self.m_children

    def setParent(self, parent):
        self.m_parent = parent

    def getParent(self):
        return self.m_parent

    def setScaleValue(self, scaleValue):
        self.m_scaleValue = scaleValue

    def getScaleValue(self):
        return self.m_scaleValue

    def getVertices(self):
        return self.m_vertices

    def setDelta(self):
        self.m_delta = True

    def resetDelta(self):
        self.m_delta = False

    def isDiscarded(self):
        return not self.m_delta

    def getStability(self) -> float:
        return self.m_stability

    def getPropagatedStability(self) -> float:
        return self.m_propagatedStability

    def setPropagatedStability(self, stability):
        self.m_propagatedStability = stability

    @staticmethod
    def resetStaticLabelCounter():
        Node.s_label = 0

    #def __str__(self):
    #    return self.getDescription()

    def getDescription(self):
        return f'N={len(self.m_vertices)},SV={self.m_scaleValue},SC={self.m_stability}'

    def getOutputDescription(self):
        return f'{len(self.m_vertices)},{self.m_scaleValue},{self.m_stability}'

    def getGraphVizNodeString(self):
        return f'node{self.m_label}'

    def getGraphVizEdgeLabelString(self):
        return f'[label="{self.m_scaleValue}"];'

    def getGraphVizString(self):
        return f'{self.getGraphVizNodeString()} [label="Num={len(self.m_vertices)}[SV,SC,D]:{{{self.m_scaleValue}; {self.m_stability}; {self.m_delta}}}""];'

    def setVertices(self, vertices ):
        self.m_vertices = set(vertices)

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

class Dendrogram:
    def __init__(self, mst: MinimalSpaningTree, min_cluster_size: int):        
        assert len(mst.getVertices()) > 0
        Node.resetStaticLabelCounter()

        self.m_components = []
        self.m_minClusterSize = min_cluster_size
        first = None
        it = iter(mst.getVertices())
        if next(it):
            first = next(it)

        assert first is not None

        self.m_mstCopy = DendrogramComponent(first, mst, True)
        
        
        self.m_root = Node(self.m_mstCopy.getVertices())
        
        self.spurious_1 = 0
        self.spurious_gr2 = 0
        
        self.m_mstCopy.setNodeRepresentitive(self.m_root)
        self.m_components.append(self.m_mstCopy)

    def build(self):
        self.experimental_build()

    def splitting(self, c, edges):
        to_remove = []
        for edge in edges:
            i = 0
            while i < len(c):
                current = c[i]
                if current.containsEdge2(edge):
                    to_remove.append(edge)
                    v1, v2 = edge.getVertex1(), edge.getVertex2()
                    if v1 == v2:
                        current.removeEdge(edge)
                    else:
                        c.pop(i)
                        c.extend(current.splitComponent(edge))
                    break
                else:
                    i += 1
        return to_remove
    
    def compare(self, n1: Node):
        return n1.getScaleValue()

    def clusterSelection(self):
        selection = []

        # Step 1
        leaves = self.getLeaves(self.m_root)
        
        for leaf in leaves:
            leaf.setPropagatedStability(leaf.computeStability())

        # Special case
        if len(leaves) == 1 and leaves[0] == self.m_root:
            selection.append(self.m_root)
            return selection

        queue = []
        for leaf in leaves:
            if leaf.getParent() is not None and leaf.getParent() not in queue:
                queue.append(leaf.getParent())
        
        queue.sort(key=self.compare)

        # Step 2
        while queue and queue[0] != self.m_root:
            current = queue[0]
            current_stability = current.computeStability()
            s = sum(child.getPropagatedStability() for child in current.getChildren())
            if current_stability < s:
                current.setPropagatedStability(s)
                current.resetDelta()
            else:
                current.setPropagatedStability(current_stability)
            queue.remove(current)
            if current.getParent() not in queue and current.getParent() is not None:
                queue.append(current.getParent())
            queue.sort(key=self.compare)

        # get clustering selection
        selection_queue = self.m_root.getChildren().copy()
        self.m_root.resetDelta()

        while selection_queue:
            current = selection_queue.pop(0)
            if not current.isDiscarded():
                selection.append(current)
            else:
                selection_queue.extend(current.getChildren())

        return selection

    @staticmethod
    def getLeaves(node: Node):
        res = []
        queue = [node]

        while queue:
            n = queue.pop(0)

            if len(n.getChildren()) > 0:
                queue.extend(n.getChildren())
            elif len(n.getChildren()) == 0:
                res.append(n)
            #queue.pop(0)

        return res
    
    def experimental_build(self):
    
        # Get set of edges with the highest weight
        
        next = self.m_mstCopy.getNextSetOfHeighestWeightedEdges()

        # return if no edge available
        if next is None:
            return

        # repeat until all edges are processed
        while next is not None:
                        
            # copy edges into "queue"
            highestWeighted = []
            highestWeighted.extend(next)

            # Mapping of a component onto it's subcomponents, resulting from splitting
            splittingCandidates = {}

            # search components which contains one of the edges which the highest weight
            for edge in highestWeighted:
                i = 0
                while i < len(self.m_components):
                    current = self.m_components[i]

                    if current.containsEdge2(edge):
                        tmp = [current]
                        splittingCandidates[current] = tmp

                        self.m_components.pop(i)
                    else:
                        i += 1

            # Split these components
            for current in splittingCandidates.keys():  # Nodes
                if len(highestWeighted) == 0:
                    break

                currentNode = current.getNode()  # get the DendrogramComponent node to access the internal DBs
                
                highest = highestWeighted[0].getWeight()
                if isinstance(highest, Edge):
                    highest = (highestWeighted[0].getWeight()).getWeight()
                    
                
                current.getNode().setScaleValue(highest)  # epsilon
        
                subComponents = splittingCandidates[current]  # get TMP

                # Info: call by reference with "subComponent".
                # Effect: After calling splitting(), the subComponent List contains the subcomponents
                # which are created by removing the edges from the whole component
                toRemove = self.splitting(subComponents, highestWeighted)

                for item in toRemove:
                    highestWeighted.remove(item)

                spuriousList = []

                # Computing the Matrix Dh for HAI
                # computingMatrixDhHAI(numberDB, current, splittingCandidates.get(current));
                for c in splittingCandidates[current]:  # scrolls through the list of splitting Components (HAI)
                    numPoints = 0.0
                    count = 0

                    for v in c.getVertices():
                        numPoints += v.getDataBubble()._weight()
                        count += 1

                    if numPoints >= self.m_minClusterSize or count >= self.m_minClusterSize:
                        spuriousList.append(c)

                if len(spuriousList) == 1:
                    self.spurious_1 += 1

                    # cluster has shrunk
                    replacementComponent = spuriousList.pop(0)

                    replacementComponent.setNodeRepresentitive(currentNode)

                    assert len(spuriousList) == 0

                    # add component to component list for further processing
                    self.m_components.append(replacementComponent)

                elif len(spuriousList) > 1:
                    
                    self.spurious_gr2 += 1

                    for c in spuriousList:
                        # generate new child node with currentNode as parent node
                        child = Node(c.getVertices())

                        child.setParent(currentNode)

                        # add child to parent
                        currentNode.addChild(child)
                        c.setNodeRepresentitive(child)

                        # add new components to component list for further processing
                        self.m_components.append(c)
                        
                        child.setScaleValue(highest)

            # update set of heighest edges
            next = self.m_mstCopy.getNextSetOfHeighestWeightedEdges()

class CoreSSG(base.Clusterer):
    
    class BufferItem:
        def __init__(self, x, timestamp, covered):
            self.x = x
            self.timestamp = (timestamp,)
            self.covered = covered

    def __init__(
        self,
        m_minPoints            = 10,
        decaying_factor: float = 0.25,
        beta: float            = 0.75,
        mu: float              = 2.0,
        epsilon: float         = 0.02,
        n_samples_init: int    = 1000,
        stream_speed: int      = 2795,
        dataset                = ''
        
    ):
        super().__init__()
        self.timestamp       = -1
        self.initialized     = False
        self.decaying_factor = decaying_factor
        self.beta            = beta
        self.mu              = mu
        self.epsilon         = epsilon
        self.n_samples_init  = n_samples_init
        self.stream_speed    = stream_speed
        self.mst             = None
        self.mst_mult        = None
        self.m_minPoints     = m_minPoints
        self.base_dir_result = os.path.join("results/" + str(dataset) + "/")
        # number of clusters generated by applying the variant of DBSCAN algorithm
        # on p-micro-cluster centers and their centers
        self.n_clusters = 0
        self.clusters: typing.Dict[int, "DataBubble"] = {}
        self.p_micro_clusters: typing.Dict[int, "DataBubble"] = {}
        self.o_micro_clusters: typing.Dict[int, "DataBubble"] = {}
        
        #mudei o método pq estava dando erro e não estavamos precisando no momento
        self._time_period = math.ceil((1 / self.decaying_factor)) # * math.log((self.mu * self.beta) / (self.mu * self.beta - 1))) + 1
        self._init_buffer: typing.Deque[typing.Dict] = deque()
        self._n_samples_seen = 0
        self.m_update = None

        # check that the value of beta is within the range (0,1]
        if not (0 < self.beta <= 1):
            raise ValueError(f"The value of `beta` (currently {self.beta}) must be within the range (0,1].")

    def _build(self):
        print("num db: ", len(self.p_micro_clusters))
        print("minPts: ", self.m_minPoints)

        for db in self.p_micro_clusters.values():
            db.setVertexRepresentative(None)
            db.setStaticCenter()
            
        if len(self.p_micro_clusters) < self.m_minPoints:
            print("no building possible since num_potential_dbs < minPoints")
            return
    
    def _initial(self, X):
        visited = np.zeros(self._n_samples_seen, dtype=int)
        db_reps = []

        start = time.time()
        nbrs  = NearestNeighbors(n_neighbors = self.m_minPoints, metric='euclidean').fit(X)

        distances, knn = nbrs.kneighbors(X)
        #print(distances.shape)

        i = 0
        for item in distances:
            if not visited[i]:
                ids_dbs = []

                visited[i] = 1

                j = 0

                for neighbour_dist in item:
                    if not visited[knn[i][j]]:
                        if neighbour_dist < self.epsilon:
                            ids_dbs.append(knn[i][j]) 
                    j += 1

                if len(ids_dbs) > self.mu:
                    db = DataBubble(
                        x = dict(zip([dim for dim in range(len(X[i]))], X[i])),
                        timestamp = self.timestamp,
                        decaying_factor = self.decaying_factor,
                    )

                    for neighbour_id in ids_dbs:
                        if not visited[neighbour_id]:
                            visited[neighbour_id] = 1

                            db.insert(dict(zip([dim for dim in range(len(X[neighbour_id]))], X[neighbour_id])))

                    self.p_micro_clusters.update({len(self.p_micro_clusters): db})
                    db_reps.append(list(db.getRep().values()))

                else:
                    visited[i] = 0
            i+=1

        end = time.time()
        print("tempo para gerar os DBs", end - start, end=' ')

        print("\nNum dbs: ", len(self.p_micro_clusters))

        start = time.time()

        nbrs = NearestNeighbors(n_neighbors = (self.m_minPoints//2) + 1, metric='euclidean').fit(db_reps)

        distances, knn = nbrs.kneighbors(db_reps)

        coreDistances_final = np.zeros((len(self.p_micro_clusters),self.m_minPoints), dtype=float)
        knng                = np.zeros((len(self.p_micro_clusters),self.m_minPoints), dtype = int)

        i = 0
        for item in distances:
            j = 0

            db_current   = self.p_micro_clusters[i]
            count_points = db_current.getN()

            for neighbour_dist in item:
                if j == 0:
                    for x in range(db_current.getN()):
                        coreDistances_final[i][x] = db_current.getNnDist(x + 1)
                        knng[i][x] = i
                else:
                    id_db = knn[i][j]

                    db_neighbour = self.p_micro_clusters[id_db]

                    for x in range(db_neighbour.getN()):
                        if (count_points + x) == self.m_minPoints:
                            break

                        coreDistances_final[i][count_points + x] = self.ditanceDataBubbles(db_current, db_neighbour, neighbour_dist) + db_neighbour.getNnDist(x + 1)
                        knng[i][count_points + x] = id_db

                    count_points += db_neighbour.getN()

                    if (count_points) >= self.m_minPoints:
                        break
                j += 1
            i+=1

        end = time.time()
        print("tempo para gerar o coreDistance",end - start, end=' ')

        np.savetxt(os.path.join(self.base_dir_result, "meu_array.csv"), coreDistances_final, delimiter=',')
        np.savetxt(os.path.join(self.base_dir_result, "meu_array_knn.csv"), knng, delimiter=',')

        print(knng.shape)

        ####################
        # Plot databubbles #
        ####################

        df_data_bubbles = pd.DataFrame(columns=['x', 'y', 'radio', 'color', 'cluster'], index = [x for x in range(len(self.p_micro_clusters))])

        i            = 0
        count_points = 0
        min_db       = 9999
        max_db       = -1

        for db in self.p_micro_clusters.values():
            df_data_bubbles.iloc[i]['x']       = db.getRep()[0]
            df_data_bubbles.iloc[i]['y']       = db.getRep()[1]
            df_data_bubbles.iloc[i]['radio']   = db.getExtent()
            df_data_bubbles.iloc[i]['color']   = 'green'
            df_data_bubbles.iloc[i]['cluster'] = 1

            i += 1
            count_points += db.getN()
            
            min_db = min(min_db, db.getN())
            max_db = max(max_db, db.getN())

        df_data_bubbles.to_csv(os.path.join(self.base_dir_result, "df_data_bubbles.csv"), sep=',', encoding='utf-8', index=False)
        
        print('N points in DBs: ', count_points)
        print('N points outliers: ', 38600 - count_points)
        print('Min db: ', min_db)
        print('Max db: ', max_db)

    def ditanceDataBubbles(self, db_current, db_neighbour, distance):
        x1 = distance - (db_current.getExtent() + db_neighbour.getExtent())
        x2 = db_current.getNnDist(1)
        x3 = db_neighbour.getNnDist(1)
        
        if x1 >= 0:
            return x1 + x2 + x3
        
        return max(x2, x3)

    def learn_one(self, x, sample_weight=None):
        self._n_samples_seen = len(x)        
            
        print("entrando no initial()")
        self._initial(x)
        
        print("entrando no build()")
        self._build()
        
        self.initialized = True
        
        del self._init_buffer

    def predict_one(self, x, sample_weight=None):
        print("self.initialized 1", self.initialized)
        # This function handles the case when a clustering request arrives.
        # implementation of the DBSCAN algorithm proposed by Ester et al.
        if not self.initialized:
            # The model is not ready
            return 0
        
        print("self.initialized 2", self.initialized)
        #if self.m_update is None:
        #    self.initialized = False
        #    return 0
        
        print("\nto aqui no predict_one pra fazer o dendrogram")
        start      = time.time()        
        dendrogram = Dendrogram(self.mst, self.m_minPoints)
        dendrogram.build()
        end        = time.time()
        print("tempo para fazer Dendrogram:",end - start, end=' ')
        
        start     = time.time()        
        selection = dendrogram.clusterSelection()
        end       = time.time()
        print("\ntempo para selecionar clusters:",end - start, end=' ')

        countDB         = len(self.mst.getVertices())
        matrixPartition = [ -1 for x in range(countDB)]
        cont            = 1 
        
        print("\nfor do selecion cluster")
        for n in selection:            
            it = iter(n.getVertices())
            for el in it:
                v                  = el.getID()
                matrixPartition[v] = cont
            cont+=1

        return matrixPartition

def set_stdout_stderr(result_dataset_path, dataset_name):
    stdout_path = os.path.join(result_dataset_path, f"stdout_{dataset_name}.txt")
    stderr_path = os.path.join(result_dataset_path, f"log_{dataset_name}.txt")
    sys.stdout  = open(stdout_path, 'w')
    sys.stderr  = open(stderr_path, 'w')

def main():
    dataset_name        = "dataset_38k"
    result_dataset_path = os.path.join("results", dataset_name)
    os.makedirs(result_dataset_path, exist_ok=True)

    # Redirects stdout and stderr to files
    set_stdout_stderr(result_dataset_path, dataset_name)

    data = pd.read_csv("datasets/" + dataset_name + ".csv", sep=',')
    data = data.to_numpy()
    print('> ', len(data))

    denstream = CoreSSG(200, decaying_factor = 0.001, mu = 0, epsilon = 0.7, dataset = dataset_name)
    denstream = denstream.learn_one(data)

if __name__ == '__main__':
    main()