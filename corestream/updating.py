from .mutual_reachability_graph import MutualReachabilityGraph
from .minimal_spaning_tree import MinimalSpaningTree
from .coressg import CoreSSG

class Updating:
    def __init__(self, mrg: MutualReachabilityGraph, mst : MinimalSpaningTree, csg : CoreSSG):
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