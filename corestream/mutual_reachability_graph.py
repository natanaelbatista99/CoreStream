
import time
import math
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from .abstract_graph import AbstractGraph
from .data_bubble import Vertex, DataBubble
from .k_nearest_neighbors_graph import KNearestNeighborsGraph
from .neighbour import Neighbour

class MutualReachabilityGraph(AbstractGraph):
    def __init__(self, G, dbs : DataBubble, mpts, timestamp):
        super().__init__()
        self.mpts    = mpts
        self.G         = G
        self.timestamp = timestamp

        for db in dbs:
            v = Vertex(db, timestamp)
            db.setVertexRepresentative(v)
            self.G.add_node(v)
            self.addVertex(v)

        self.knng = KNearestNeighborsGraph(G)
        
        start = time.time()
        self.computeCoreDistance()
        end   = time.time()
        print("> Time coreDistanceDB", end - start, end='\n')

    def getKnngGraph(self):
        return self.knng
       
    def buildGraph(self):
        seen = set()

        for idx1, v1 in enumerate(self.G.nodes):
            for idx2, v2 in enumerate(self.G.nodes):
                pair = (idx1, idx2)

                if idx1 >= idx2 or pair in seen:
                    continue
                
                seen.add(pair)
                mrd = self.getMutualReachabilityDistance(v1, v2)
                self.G.add_edge(v1, v2, weight = mrd)
        
        del seen

    def computeCoreDistance(self):
        vertices_list = list(self.G.nodes)
        coords        = [[v for k, v in vertex.getDataBubble().getRep(self.timestamp).items()] for vertex in vertices_list]
        
        kdtree = KDTree(coords)

        # mpts: valor fixo para a distância do m-ésimo vizinho
        _, indices = kdtree.query(coords, k = math.floor((self.mpts + 1) / 2))

        # Atualiza os objetos Vertex no grafo com a core distance
        for i, knn in enumerate(indices):
            current      = vertices_list[i]
            mpts_objects = 0.0
            neighbour_c  = None

            for j, k in enumerate(knn):
                k_neighbour   = vertices_list[k]
                weight        = k_neighbour.getDataBubble()._weight(self.timestamp)
                mpts_objects += weight

                if (current != k_neighbour) and self.knng.getEdge(current, k_neighbour) is None:
                    self.knng.setEdge(current, k_neighbour)

                if (current == k_neighbour) and (weight >= self.mpts):
                    current.setCoreDistance(current.getDataBubble().getNnDist(self.mpts, self.timestamp))
                    break
                elif (mpts_objects >= self.mpts):
                    mpts_objects -= weight
                    neighbour_c   = k_neighbour
                    break

            if neighbour_c != None:
                # EXTENT OF CURRENT AND NEIGHBOUR c
                extent_current       = current.getDataBubble().getExtent(self.timestamp)
                extent_neighbour_c   = neighbour_c.getDataBubble().getExtent(self.timestamp)
                
                overlapping          = current.getDistanceRep(neighbour_c) - (extent_current + extent_neighbour_c)
                
                knn_dist_neighbour_c = neighbour_c.getDataBubble().getNnDist(self.mpts - mpts_objects, self.timestamp)
                
                if(overlapping >= 0.0):
                    current.setCoreDistance(current.getDistanceRep(neighbour_c) - extent_neighbour_c + knn_dist_neighbour_c)
                else:
                    overlapping *= -1
                    
                    if knn_dist_neighbour_c <= overlapping:
                        current.setCoreDistance(extent_current)
                    else:
                        current.setCoreDistance(extent_current + knn_dist_neighbour_c - overlapping)

    def getMutualReachabilityDistance(self, v1, v2):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))
    
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