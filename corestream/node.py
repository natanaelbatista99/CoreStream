class Node:
    s_label = 0

    def __init__(self, c, timestamp):
        self.m_vertices   = set(c)
        self.m_children   = []
        self.m_delta      = True
        self.m_label      = Node.s_label
        self.m_id         = Node.s_label
        Node.s_label      += 1
        self.m_parent     = None
        self.m_scaleValue = 0
        self.m_stability  = 0.0
        self.timestamp    = timestamp
        
    def getID(self):
        return self.m_id
    
    def getInternalPoints(self):
        numPoints = 0.0

        for v in self.getVertices():
            numPoints += v.getDataBubble()._weight(self.timestamp)
            
        return numPoints
    
    def computeStabilityNew(self) -> float:
        self.m_stability = 0.0
        eps_max          = self.m_scaleValue
        
        for child in self.getChildren():
            self.m_stability += child.getInternalPoints() * ((1.0 / child.getScaleValue()) - (1.0 / eps_max))

        #print("<<< ", self.m_stability)
        return self.m_stability
    
    def computeStability(self) -> float:
        if self.m_parent is None:
            return float('nan')

        eps_max = self.m_parent.m_scaleValue
        eps_min = self.m_scaleValue
        
        if eps_max == 0:
            eps_max = 0.0000000001
        if eps_min == 0:
            eps_min = 0.0000000001
        
        # É o somatório dos pesos vezes a densidade minima (quando o Cluster foi criado) + a densidade máxima (quando o DB saiu do cluster)
        self.m_stability = len(self.m_vertices) * ((1 / eps_min) - (1 / eps_max))

        return self.m_stability

    def addChild(self, child: "Node"):
        self.m_children.append(child)

    def getChildren(self):
        return self.m_children
    
    def getChildrenMinClusterSize(self, min_cluster_size):
        children = []
        
        for child in self.getChildren():
            if child.getInternalPoints() >= min_cluster_size:
                children.append(child)
            
        return children

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
        return "node" + str(self.m_label)

    def getGraphVizEdgeLabelString(self):
        return "[label=\"{:.2f}\"];".format(self.m_scaleValue)

    def getGraphVizString(self):
        return "{} [label=\"Num={}[SV,SC,D,P]:{{ {:.4f}; {:.10f}; {}; ".format(self.getGraphVizNodeString(), len(self.m_vertices), self.m_scaleValue, self.m_stability, self.m_delta)

    def setVertices(self, vertices ):
        self.m_vertices = set(vertices)