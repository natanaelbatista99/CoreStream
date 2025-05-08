import math
from abc import ABCMeta

class Vertex():

    s_idCounter = 0

    def __init__(self, db, timestamp, id=None):
        self.m_id          = id if id is not None else Vertex.s_idCounter
        Vertex.s_idCounter += 1
        self.m_db          = db
        self.timestamp     = timestamp
        
        if self.m_db is not None:
            self.m_db.setVertexRepresentative(self)
            
        self.m_visited            = False
        self.m_coreDistanceObject = None
        self.m_lrd                = -1
        self.m_coreDist           = 0
        
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
        #return f"{self.getGraphVizVertexString()} [label = " {self.String}  cdist={self.getCoreDistance()}"];"
        return "{} [label=\"{}\"]" .format(self.getGraphVizVertexString(),self.getGraphVizVertexString())

    def getDistanceToVertex(self, other):
        return self.m_db.getCenterDistance(other.getDataBubble())
    
    def getDistanceRep(self, vertex):
        x1 = self.distance(self.m_db.getRep(self.timestamp), vertex.getDataBubble().getRep(self.timestamp))
        
        return x1

    def getDistance(self, vertex):
        if self.m_db.getStaticCenter() is None or vertex.getDataBubble().getStaticCenter() is None:
            return self.getDistanceToVertex(vertex)
        
        x1 = self.distance(self.m_db.getRep(self.timestamp), vertex.getDataBubble().getRep(self.timestamp)) - (self.m_db.getExtent(self.timestamp) + vertex.getDataBubble().getExtent(self.timestamp))
        x2 = self.m_db.getNnDist(1, self.timestamp)
        x3 = vertex.getDataBubble().getNnDist(1, self.timestamp)
        
        if x1 >= 0:
            return x1 + x2 + x3
        
        return max(x2, x3)
    
        #return self.distance(self.m_db.getRep(self.timestamp), vertex.getDataBubble().getRep(self.timestamp))

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
        self.m_visited = False

    def getID(self):
        return self.m_id

    def compareID(self,  other: "Vertex"):
        return self.m_id == other.m_id

class DataBubble(metaclass=ABCMeta):
    
    s_idCounter = 0
  
    def __init__(self, x, timestamp, decaying_factor):

        self.x = x
        
        self.db_id              = DataBubble.s_idCounter
        DataBubble.s_idCounter += 1
        
        self.last_edit_time  = timestamp
        self.creation_time   = timestamp
        self.decaying_factor = decaying_factor

        self.N              = 1
        self.linear_sum     = x
        self.squared_sum    = {i: (x_val * x_val) for i, x_val in x.items()}        
        self.m_staticCenter = len(self.linear_sum)
    
    def getID(self):
        return self.db_id

    def setID(self, id):
        self.db_id = id
    
    def getN(self):
        return self.N

    def _weight(self, timestamp):
        return self.N * self.fading_function(timestamp - self.last_edit_time)
        
    def getRep(self, timestamp):
        ff     = self.fading_function(timestamp - self.last_edit_time)
        weight = self._weight(timestamp)
        center = {key: (val * ff) / weight for key, val in self.linear_sum.items()}
        
        return center

    def getExtentDB(self, timestamp):        
        x1  = 0
        x2  = 0
        res = 0
        
        ff     = self.fading_function(timestamp - self.last_edit_time)
        weight = self._weight(timestamp)
        
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            val_ss = self.squared_sum[key]
            
            x1  = 2 * (val_ss * ff) * weight
            x2  = 2 * (val_ls * ff)**2
            tmp = (x1 - x2)
            
            if tmp <= 0.0:
                tmp = 1/10 * 1/10
            
            diff = (tmp / (weight * (weight - 1))) if (weight * (weight - 1)) > 0.0 else 0.1
            
            res += math.sqrt(diff) if diff > 0 else 0

        return (res / len(self.linear_sum)) * 1.4 #redius factor
        #return res

    def getExtent(self, timestamp):        
        x1  = 0
        x2  = 0
        res = 0
        
        ff     = self.fading_function(timestamp - self.last_edit_time)
        weight = self._weight(timestamp)
    
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            val_ss = self.squared_sum[key]
            
            # raio Micro-Cluster
            x1  = (val_ss * ff) / weight
            x2  = ((val_ls * ff) / weight)**2
            tmp = (x1 - x2)
            
            res += math.sqrt(tmp) if tmp > 0 else (1/10 * 1/10)
            
        return (res / len(self.linear_sum)) * 1.8  #redius factor
        #return res

    def insert(self, x, timestamp):
        
        if self.last_edit_time != timestamp:
            self.fade(timestamp)
            
        self.last_edit_time = timestamp
        
        self.N += 1
        
        for key, val in x.items():
            try:
                self.linear_sum[key]  += val
                self.squared_sum[key] += val * val
            except KeyError:
                self.linear_sum[key]  = val
                self.squared_sum[key] = val * val
    
    def fade(self, timestamp):
        ff = self.fading_function(timestamp - self.last_edit_time)
        
        self.N *= ff
        
        for key, val in self.linear_sum.items():
            self.linear_sum[key]  *= ff
            self.squared_sum[key] *= ff
    
    def merge(self, cluster):
        self.N += cluster.N
        
        for key in cluster.linear_sum.keys():
            try:
                self.linear_sum[key]  += cluster.linear_sum[key]
                self.squared_sum[key] += cluster.squared_sum[key]
            except KeyError:
                self.linear_sum[key]  = cluster.linear_sum[key]
                self.squared_sum[key] = cluster.squared_sum[key]
                
        if self.last_edit_time < cluster.creation_time:
            self.last_edit_time = cluster.creation_time

    def getNnDist(self, k, timestamp):
        return ((k / self.N)**(1.0 / len(self.linear_sum))) * self.getExtent(timestamp)

    def fading_function(self, time):
        return 2 ** (-self.decaying_factor * time)
    
    def setVertexRepresentative(self, v : Vertex):
        self.m_vertexRepresentative = v

    def getVertexRepresentative(self):
        return self.m_vertexRepresentative
    
    def getStaticCenter(self):
        return self.m_staticCenter

    def setStaticCenter(self, timestamp):
        m_static_center = self.getRep(timestamp).copy()
        return m_static_center