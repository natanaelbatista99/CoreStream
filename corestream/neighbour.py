from .data_bubble import Vertex

class Neighbour():
    def __init__(self, vertex = Vertex, dist=None):
        if dist is not None:
            self.m_coreDist = dist
        if vertex is not None:
            self.m_vertex   = vertex
            self.m_coreDist = dist

    def getDistance(self):
        return self.m_coreDist

    def String(self):
        return "value = {:.2f}".format(self.m_coreDist)

    def getVertex(self):
        return self.m_vertex
    
    def setCoredist(self, nn_dist):
        self.m_coreDist += nn_dist