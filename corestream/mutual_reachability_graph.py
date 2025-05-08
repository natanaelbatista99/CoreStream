
import time
import seaborn as sns
import matplotlib.pyplot as plt

from .abstract_graph import AbstractGraph
from .data_bubble import Vertex, DataBubble
from .k_nearest_neighbors_graph import KNearestNeighborsGraph
from .neighbour import Neighbour

class MutualReachabilityGraph(AbstractGraph):
    def __init__(self, G, dbs : DataBubble, mpts, timestamp):
        super().__init__()
        self.m_mpts  = mpts
        self.G         = G
        self.timestamp = timestamp

        for db in dbs:
            v = Vertex(db, timestamp)
            db.setVertexRepresentative(v)
            self.G.add_node(v)
            
            self.addVertex(v)

        self.knng = KNearestNeighborsGraph(G)
        
        start = time.time()
        self.computeCoreDistance(G, mpts)
        end   = time.time()
        print("> Time coreDistanceDB", end - start, end='\n')

    def getKnngGraph(self):
        return self.knng
    
    def buildAbsGraph(self):
        
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        plot_kwds = {'s' : 1, 'linewidths':0}
        
        plt.figure(figsize = (16,12))
        
        linhas = []
        
        for i, (u,v,w) in enumerate(self.G.edges(data='weight')):
            self.addVertex(u)
            self.addVertex(v)
            self.addEdge(u,v,w)
            
            plt.gca().add_patch(plt.Circle((u.getDataBubble().getRep(self.timestamp)[0],u.getDataBubble().getRep(self.timestamp)[1]), u.getDataBubble().getExtent(self.timestamp), color='blue', fill=False))
            plt.gca().add_patch(plt.Circle((v.getDataBubble().getRep(self.timestamp)[0],v.getDataBubble().getRep(self.timestamp)[1]), v.getDataBubble().getExtent(self.timestamp), color='blue', fill=False))
            
            linhas.append(((u.getDataBubble().getRep(self.timestamp)[0],u.getDataBubble().getRep(self.timestamp)[1]) ,(v.getDataBubble().getRep(self.timestamp)[0],v.getDataBubble().getRep(self.timestamp)[1]), w))
            
            plt.text(u.getDataBubble().getRep(self.timestamp)[0], u.getDataBubble().getRep(self.timestamp)[1], str(u.getID()), fontsize=18, ha='center', va='center')
            plt.text(v.getDataBubble().getRep(self.timestamp)[0], v.getDataBubble().getRep(self.timestamp)[1], str(v.getID()), fontsize=18, ha='center', va='center')
        
        # Loop através da lista de linhas
        for (x1, y1), (x2, y2), numero in linhas:
            # Trace a linha
            plt.plot([x1, x2], [y1, y2], marker='o', linestyle='-', markersize=5, label=str(numero))

            # Adicione o número como texto no meio da linha
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(numero), fontsize=12, ha='center', va='center')

        # Mostre o gráfico
        plt.scatter(data[0:5000, 0], data[0:5000, 1], **plot_kwds)
        plt.show()
       
    def buildGraph(self):
        for v1 in self.G:            
            for v2 in self.G:
                if v1 != v2:
                    mrd = self.getMutualReachabilityDistance(v1, v2)
                    self.G.add_edge(v1, v2, weight = mrd)
                    self.addEdge(v1, v2, mrd)

    def computeCoreDistance(self, vertices, mpts):
        for current in vertices:
            if current.getDataBubble()._weight(self.timestamp) >= mpts:
                current.setCoreDist(current.getDataBubble().getNnDist(mpts, self.timestamp))
            else:
                neighbours  = self.getNeighbourhood(current, vertices)
                countPoints = current.getDataBubble()._weight(self.timestamp)
                neighbourC  = None

                for n in neighbours:
                    weight       = n.getVertex().getDataBubble()._weight(self.timestamp)
                    countPoints += weight
                    
                    if self.knng.getEdge(current, n.getVertex()) is None:
                        self.knng.setEdge(current, n.getVertex())
                        
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
                
    def getNeighbourhood(self, vertex, vertices):
        neighbours = []
        
        for v in vertices:
            if v != vertex:
                neighbour = Neighbour(v, vertex.getDistanceRep(v) - v.getDataBubble().getExtent(self.timestamp))
                neighbours.append(neighbour)
                
        neighbours.sort(key=lambda x: x.getDistance(), reverse=False)
        
        return neighbours

    def getMutualReachabilityDistance(self, v1, v2):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))