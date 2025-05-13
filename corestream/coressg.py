import networkx as nx
import math

from sklearn.neighbors import KDTree
from .data_bubble import Vertex
from .neighbour import Neighbour
from .abstract_graph import AbstractGraph
from .minimal_spaning_tree import MinimalSpaningTree
from .k_nearest_neighbors_graph import KNearestNeighborsGraph

class CoreSSG(AbstractGraph):
    def __init__(self, mst: MinimalSpaningTree, knng: KNearestNeighborsGraph, timestamp):
        super().__init__()
        
        self.timestamp = timestamp
        
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

    def computeHierarchyMpts(self, mpts: int):
        self.computeCoreDistance(mpts)
        edgesGraph = self.getEdges()
        
        for e in edgesGraph:
            self.removeEdge2(e)
            self.addEdge(e.getVertex1(), e.getVertex2(), self.getMutualReachabilityDistance(e.getVertex1(), e.getVertex2()))

    def computeCoreDistance(self, mpts: int):
        vertices_list = list(self.getVertices())
        coords        = [[v for k, v in vertex.getDataBubble().getRep(self.timestamp).items()] for vertex in vertices_list]
        
        kdtree = KDTree(coords)

        # mpts: valor fixo para a distância do m-ésimo vizinho
        _, indices = kdtree.query(coords, k = math.floor((mpts + 1) / 2))

        # Atualiza os objetos Vertex no grafo com a core distance
        for i, knn in enumerate(indices):
            current      = vertices_list[i]
            mpts_objects = 0.0
            neighbour_c  = None

            for j, k in enumerate(knn):
                k_neighbour   = vertices_list[k]
                weight        = k_neighbour.getDataBubble()._weight(self.timestamp)
                mpts_objects += weight

                if (current == k_neighbour) and (weight >= mpts):
                    current.setCoreDistance(current.getDataBubble().getNnDist(mpts, self.timestamp))
                    break
                elif (mpts_objects >= mpts):
                    mpts_objects -= weight
                    neighbour_c   = k_neighbour
                    break

            if neighbour_c != None:
                # EXTENT OF CURRENT AND NEIGHBOUR c
                extent_current       = current.getDataBubble().getExtent(self.timestamp)
                extent_neighbour_c   = neighbour_c.getDataBubble().getExtent(self.timestamp)
                
                overlapping          = current.getDistanceRep(neighbour_c) - (extent_current + extent_neighbour_c)
                
                knn_dist_neighbour_c = neighbour_c.getDataBubble().getNnDist(mpts - mpts_objects, self.timestamp)
                
                if(overlapping >= 0.0):
                    current.setCoreDistance(current.getDistanceRep(neighbour_c) - extent_neighbour_c + knn_dist_neighbour_c)
                else:
                    overlapping *= -1
                    
                    if knn_dist_neighbour_c <= overlapping:
                        current.setCoreDistance(extent_current)
                    else:
                        current.setCoreDistance(extent_current + knn_dist_neighbour_c - overlapping)

    def getMutualReachabilityDistance(self, v1: Vertex, v2: Vertex):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))