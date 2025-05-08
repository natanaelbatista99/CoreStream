import os
from .edge import Edge
from .data_bubble import Vertex

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

    def getGraphVizString(self, timestamp, mpts):
        
        m_directory = os.path.join(os.getcwd(), "results/graphviz")
        
        try:
            sub_dir = os.path.join(m_directory, "graphviz_t" + str(timestamp))

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            with open(os.path.join(sub_dir, "graphviz_Mpts_" + str(mpts) + ".txt"), 'w') as writer:
                writer.write("graph {\n")

                edges = set()

                vertices = sorted(self.m_graph, key=lambda x: x.getID())
                
                for v in vertices:
                    writer.write("\t" + v.getGraphVizString() + "\n")
                    edges.update(self.adjacencyList(v).values())

                edges_sorted = sorted(edges, key=lambda x: x.getWeight())

                for e in edges_sorted:
                    writer.write("\t" + e.graphVizString() + "\n")

                writer.write("}\n")

        except FileNotFoundError as e:
            print(e)

    def getAdjacencyMatrixAsArray(self):
        matrix = [[0.0 for _ in range(len(self.m_graph))] for _ in range(len(self.m_graph))]
        df = "{:.4f}"

        sorted_by_id = sorted(self.m_graph, key=lambda x: x.id)

        for row in range(len(sorted_by_id)):
            for column in range(len(sorted_by_id)):
                v1   = sorted_by_id[row]
                v2   = sorted_by_id[column]
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
        edges           = set()
        
        for v in vertex_iterator:
            edges.update(self.adjacencyList(v).values())
            
        return len(edges) == int(self.numVertices() * (self.numVertices() - 1) / 2)
    
    #aqui pode ter um possível erro preciso revisar
    def hasSelfLoop(self, vertex: Vertex):
        if vertex not in self.m_graph:
            return Exception("Vertex does not exist!")
        
        return  vertex in self.adjacencyList(vertex).keys()