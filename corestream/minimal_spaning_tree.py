import seaborn as sns
import matplotlib.pyplot as plt

from .abstract_graph import AbstractGraph

class MinimalSpaningTree(AbstractGraph):
    def __init__(self, graph):
        super().__init__()
        self.m_inputGraph = graph
        #self.mst_to_hdbscan = []

    def buildGraph(self):
        for i, (u,v,w) in enumerate(self.m_inputGraph.edges(data='weight')):
            self.addVertex(u)
            self.addVertex(v)
            self.addEdge(u,v,w)
        #    self.mst_to_hdbscan.append([u.getID() , v.getID() , w]) 
        #print(self.mst_to_hdbscan)
    
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

    def buildAbsGraph(self, timestamp):
        
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        plot_kwds = {'s' : 1, 'linewidths':0}
        
        plt.figure(figsize = (16,12))
        
        linhas = []
        
        for edge in self.getEdges():
            u = edge.getVertex1()
            v = edge.getVertex2()

            plt.gca().add_patch(plt.Circle((u.getDataBubble().getRep(timestamp)[0],u.getDataBubble().getRep(timestamp)[1]), u.getDataBubble().getExtent(timestamp), color='blue', fill=False))
            plt.gca().add_patch(plt.Circle((v.getDataBubble().getRep(timestamp)[0],v.getDataBubble().getRep(timestamp)[1]), v.getDataBubble().getExtent(timestamp), color='blue', fill=False))
            
            linhas.append(((u.getDataBubble().getRep(timestamp)[0],u.getDataBubble().getRep(timestamp)[1]) ,(v.getDataBubble().getRep(timestamp)[0],v.getDataBubble().getRep(timestamp)[1])))
            
            plt.text(u.getDataBubble().getRep(timestamp)[0], u.getDataBubble().getRep(timestamp)[1], str(u.getID()), fontsize=10, ha='center', va='center')
            plt.text(v.getDataBubble().getRep(timestamp)[0], v.getDataBubble().getRep(timestamp)[1], str(v.getID()), fontsize=10, ha='center', va='center')
        
        # Loop através da lista de linhas
        for (x1, y1), (x2, y2) in linhas:
            # Trace a linha
            plt.plot([x1, x2], [y1, y2], marker='o', linestyle='-', markersize=5)

        # Mostre o gráfico
        plt.scatter(data[0:5000, 0], data[0:5000, 1], **plot_kwds)
        plt.show()