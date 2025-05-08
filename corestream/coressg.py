import networkx as nx

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
        vertices = self.getVertices()
        
        for current in vertices:
            if current.getDataBubble()._weight(self.timestamp) >= mpts:
                nnDist = current.getDataBubble().getNnDist(mpts, self.timestamp)
                current.setCoreDist(nnDist)
            else:
                neighbours  = self.getNeighbourhoodMptsNN(current)
                countPoints = current.getDataBubble()._weight(self.timestamp)
                neighbourC  = None

                for n in neighbours:
                    weight      = n.getVertex().getDataBubble()._weight(self.timestamp)
                    countPoints += weight
                        
                    if countPoints >= mpts:
                        countPoints -= weight
                        neighbourC   = n
                        break
                
                extentCurrent    = current.getDataBubble().getExtent(self.timestamp)
                extentNeighbourC = neighbourC.getVertex().getDataBubble().getExtent(self.timestamp)
                
                overlapping = current.getDistanceRep(neighbourC.getVertex()) - (extentCurrent + extentNeighbourC)
                
                knnDistNeighbourC = neighbourC.getVertex().getDataBubble().getNnDist(mpts - countPoints, self.timestamp)
                
                if(overlapping >= 0.0):
                    current.setCoreDist(current.getDistanceRep(neighbourC.getVertex()) - extentNeighbourC + knnDistNeighbourC)
                else:
                    overlapping *= -1
                    
                    if knnDistNeighbourC <= overlapping:
                        current.setCoreDist(current.getDataBubble().getExtent(self.timestamp))
                    else:
                        current.setCoreDist(current.getDataBubble().getExtent(self.timestamp) + knnDistNeighbourC - overlapping)
                
    def getNeighbourhoodMptsNN(self, vertex):
        neighbours = []
        vertices   = self.getAdjacentEdges(vertex).keys()
        
        for v in vertices:
            if v != vertex:
                neighbour = Neighbour(v, vertex.getDistanceRep(v) - v.getDataBubble().getExtent(self.timestamp))
                neighbours.append(neighbour)
        
        neighbours.sort(key=lambda x: x.getDistance(), reverse=False)
        
        return neighbours

    def getMutualReachabilityDistance(self, v1: Vertex, v2: Vertex):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))