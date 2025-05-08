import os
import numpy as np
import matplotlib.pyplot as plt

from .edge import Edge
from .node import Node
from logging import warn
from .minimal_spaning_tree import MinimalSpaningTree
from .dendrogram_component import DendrogramComponent

class Dendrogram:
    def __init__(self, mst: MinimalSpaningTree, min_cluster_size: int, mpts: int,timestamp):        
        assert len(mst.getVertices()) > 0
        Node.resetStaticLabelCounter()

        self.m_components     = []
        self.m_minClusterSize = min_cluster_size
        self.m_mpts           = mpts
        first                 = None
        it                    = iter(mst.getVertices())
        self.timestamp        = timestamp
        
        if next(it):
            first = next(it)

        assert first is not None

        self.m_mstCopy = DendrogramComponent(first, mst, True)
        
        self.m_root = Node(self.m_mstCopy.getVertices(), self.timestamp)
        
        self.spurious_1   = 0
        self.spurious_gr2 = 0
        
        self.m_mstCopy.setNodeRepresentitive(self.m_root)
        self.m_components.append(self.m_mstCopy)
        self.len_mst = len(mst.getVertices())

    def build(self):
        # Get set of edges with the highest weight
        
        next = self.m_mstCopy.getNextSetOfHeighestWeightedEdges()

        # return if no edge available
        if next is None:
            return

        # repeat until all edges are processed
        while next is not None:
                        
            # copy edges into "queue"
            highestWeighted = []
            highestWeighted.extend(next)

            # Mapping of a component onto it's subcomponents, resulting from splitting
            splittingCandidates = {}

            # search components which contains one of the edges which the highest weight
            for edge in highestWeighted:
                i = 0
                while i < len(self.m_components):
                    current = self.m_components[i]

                    if current.containsEdge2(edge):
                        tmp = [current]
                        splittingCandidates[current] = tmp

                        self.m_components.pop(i)
                    else:
                        i += 1

            # Split these components
            for current in splittingCandidates.keys():  # Nodes
                if len(highestWeighted) == 0:
                    break

                currentNode = current.getNode()  # get the DendrogramComponent node to access the internal DBs
                
                highest = highestWeighted[0].getWeight()
                if isinstance(highest, Edge):
                    highest = (highestWeighted[0].getWeight()).getWeight()
                    
                
                current.getNode().setScaleValue(highest)  # epsilon
        
                subComponents = splittingCandidates[current]  # get TMP

                # Info: call by reference with "subComponent".
                # Effect: After calling splitting(), the subComponent List contains the subcomponents
                # which are created by removing the edges from the whole component
                toRemove = self.splitting(subComponents, highestWeighted)

                for item in toRemove:
                    highestWeighted.remove(item)

                spuriousList = []

                # Computing the Matrix Dh for HAI
                # computingMatrixDhHAI(numberDB, current, splittingCandidates.get(current));
                for c in splittingCandidates[current]:  # scrolls through the list of splitting Components (HAI)
                    numPoints = 0.0
                    count     = 0

                    for v in c.getVertices():
                        numPoints += v.getDataBubble()._weight(self.timestamp)
                        count     += 1

                    if numPoints >= self.m_minClusterSize or count >= self.m_minClusterSize:
                        spuriousList.append(c)

                if len(spuriousList) == 1:
                    self.spurious_1 += 1

                    # cluster has shrunk
                    replacementComponent = spuriousList.pop(0)

                    replacementComponent.setNodeRepresentitive(currentNode)

                    assert len(spuriousList) == 0

                    # add component to component list for further processing
                    self.m_components.append(replacementComponent)

                elif len(spuriousList) > 1:
                    
                    self.spurious_gr2 += 1

                    for c in spuriousList:
                        # generate new child node with currentNode as parent node
                        child = Node(c.getVertices(), self.timestamp)

                        child.setParent(currentNode)

                        # add child to parent
                        currentNode.addChild(child)
                        c.setNodeRepresentitive(child)

                        # add new components to component list for further processing
                        self.m_components.append(c)
                        
                        child.setScaleValue(highest)

            # update set of heighest edges
            next = self.m_mstCopy.getNextSetOfHeighestWeightedEdges()

    def splitting(self, c, edges):
        to_remove = []
        for edge in edges:
            i = 0
            while i < len(c):
                current = c[i]
                if current.containsEdge2(edge):
                    to_remove.append(edge)
                    v1, v2 = edge.getVertex1(), edge.getVertex2()
                    if v1 == v2:
                        current.removeEdge(edge)
                    else:
                        c.pop(i)
                        c.extend(current.splitComponent(edge))
                    break
                else:
                    i += 1
        return to_remove
    
    def getLeaves(self, node):
        res   = []
        queue = [node]

        while queue:
            n = queue.pop(0)
            
            count = 0
            
            for child in n.getChildrenMinClusterSize(self.m_minClusterSize):
                count += 1
                queue.append(child)
            
            if count == 0:
                res.append(n)

        return res

    # Compare to Nodes
    def compareNode(self, n: Node):
        return (n.getScaleValue(), n.getInternalPoints())
    
    def clusterSelection(self):
        selection = []

        # Step 1
        leaves = self.getLeaves(self.m_root)
        
        for leaf in leaves:
            #print("leaf ", leaf.getInternalPoints())
            leaf.setPropagatedStability(leaf.computeStability())
            #print("SC: ", leaf.getStability())

        # Special case
        if len(leaves) == 1 and leaves[0] == self.m_root:
            selection.append(self.m_root)
            
            return selection

        queue = []
        
        # add the Parent of the leaves
        for leaf in leaves:
            if leaf.getParent() is not None and leaf.getParent() not in queue:
                queue.append(leaf.getParent())
        
        queue.sort(key=self.compareNode)
        
        #for i in queue:
        #    print("ord: ", i.getInternalPoints())
        
        # Step 2
        while queue:
            current           = queue[0]
            current_stability = current.computeStability()
            
            children_sum_stability = 0.0
            
            #print("Pai ", current.getInternalPoints())
            
            #for child in current.getChildrenMinClusterSize(self.m_minClusterSize):
                #print("Child ", child.getInternalPoints())
            #    children_sum_stability += child.getPropagatedStability()
                
            children_sum_stability = sum(child.getPropagatedStability() for child in current.getChildrenMinClusterSize(self.m_minClusterSize))
            
            #print("> Stability Pai: ", current_stability)
            #print("> Stability Children: ", children_sum_stability)
            
            if current_stability < children_sum_stability:
                current.setPropagatedStability(children_sum_stability)
                current.resetDelta()
            else:
                current.setPropagatedStability(current_stability)
            
            for c in current.getChildrenMinClusterSize(self.m_minClusterSize):
                if c.getPropagatedStability() == 0:
                    c.resetDelta()
            
            queue.remove(current)
            
            if current.getParent() not in queue and current.getParent() is not None:
                queue.append(current.getParent())
        
            queue.sort(key=self.compareNode)

        # get clustering selection
        selection_queue = self.m_root.getChildrenMinClusterSize(self.m_minClusterSize).copy()
        self.m_root.resetDelta()

        while selection_queue:
            current = selection_queue.pop(0)
            
            if not current.isDiscarded():
                #print("Node: ", current.getInternalPoints())
                selection.append(current)
            else:
                selection_queue.extend(current.getChildrenMinClusterSize(self.m_minClusterSize))
                
        #self.condensedTreePlot(selection)
        return selection
    
    def getLeavesDfs(self, node):
        res = []
        
        if len(node.getChildrenMinClusterSize(self.m_minClusterSize)) == 0:
            res.append(node)
            return res

        for n in node.getChildrenMinClusterSize(self.m_minClusterSize):
            res.extend(self.getLeavesDfs(n))

        return res      
        
    def condensedTreePlot(self, selection, select_clusters = True, selection_palette = None, label_clusters = False):
        
        cluster_x_coords = {}

        leaves = self.getLeavesDfs(self.m_root)
        leaf_position = 0.0

        # set coords X from leaves
        for leaf in leaves:
            cluster_x_coords[leaf] = leaf_position
            leaf_position += 1
        
        # add the x and y coordinates for the clusters
        queue = []
        
        queue.extend(leaves)
        
        queue.sort(key = self.compareNode)
        
        cluster_y_coords = {self.m_root: 0.0}

        while queue:
            n        = queue[0]
            children = n.getChildrenMinClusterSize(self.m_minClusterSize)
            
            if len(children) > 1:
                left_child = children[0]
                right_child = children[1]
                
                mean_coords_children = (cluster_x_coords[left_child] + cluster_x_coords[right_child]) / 2.0
                cluster_x_coords[n] = mean_coords_children
                
                cluster_y_coords[left_child] = 1.0 / left_child.getScaleValue()
                cluster_y_coords[right_child] = 1.0 / right_child.getScaleValue()
                
            if n.getParent() is not None and n.getParent() not in queue:
                queue.append(n.getParent())

            queue.remove(n)
            queue.sort(key = self.compareNode)

        #print("> 1º While")

        # set scaling to plot
        root    = self.m_root
        scaling = 0
        
        for c in self.m_root.getChildren():
            scaling += len(c.getVertices())
        
        cluster_bounds = {}

        bar_centers = []
        bar_heights = []
        bar_bottoms = []
        bar_widths  = []

        # set bar configuration
        queue.clear()
        
        queue = [self.m_root]

        while queue:
            c = queue[0]
            
            cluster_bounds[c] = [0, 0, 0, 0]
            
            n_children = c.getChildren()
            
            if len(n_children) == 0:
                queue.remove(c)
                continue            
            
            current_size = 0           
            
            max_lambda = []
            for a in n_children:
                current_size += len(a.getVertices())
                max_lambda.append(1.0 / a.getScaleValue())
            
            current_lambda   = cluster_y_coords[c]
            cluster_max_size = current_size
            
            cluster_max_lambda = max_lambda[-1]
            
            cluster_min_size = 0
            
            for b in n_children:
                if (1.0 / b.getScaleValue())  == cluster_max_lambda:
                    cluster_min_size += len(b.getVertices())
            
            #2000
            max_rectangle_per_icicle = 20
            total_size_change        = float(cluster_max_size - cluster_min_size)
            step_size_change         = total_size_change / max_rectangle_per_icicle
            
            cluster_bounds[c][0] = cluster_x_coords[c] * scaling - (current_size / 2.0)
            cluster_bounds[c][1] = cluster_x_coords[c] * scaling + (current_size / 2.0)
            cluster_bounds[c][2] = cluster_y_coords[c]
            cluster_bounds[c][3] = cluster_max_lambda
            
            last_step_size   = current_size
            last_step_lambda = current_lambda
            
            
            for i in n_children:
                
                if (1.0 / i.getScaleValue())  != current_lambda and (last_step_size - current_size > step_size_change or (1.0 / i.getScaleValue()) == cluster_max_lambda):
                    bar_centers.append(cluster_x_coords[c] * scaling)
                    bar_heights.append((1.0 / i.getScaleValue()) - last_step_lambda)
                    bar_bottoms.append(last_step_lambda)
                    bar_widths.append(last_step_size)
                    last_step_size   = current_size
                    last_step_lambda = current_lambda
                else:
                    current_size -= len(i.getVertices())
                    
                current_lambda = 1.0 / i.getScaleValue()
            
            if c.getChildrenMinClusterSize(self.m_minClusterSize) is not None:
                queue.extend(c.getChildrenMinClusterSize(self.m_minClusterSize))

            queue.remove(c)

        #print("> 2º While")

        # set lines to plot
        line_xs = []
        line_ys = []

        queue_dendrogram = []
        queue_dendrogram.append(self.m_root)

        while queue_dendrogram:
            n = queue_dendrogram[0]
            children = n.getChildrenMinClusterSize(self.m_minClusterSize)

            for n_child in children:
                sign = 1

                if (cluster_x_coords[n_child] - cluster_x_coords[n]) < 0:
                    sign = -1

                line_xs.append((cluster_x_coords[n] * scaling, cluster_x_coords[n_child] * scaling + sign * (len(n_child.getVertices()) / 2.0)))
                line_ys.append((cluster_y_coords[n_child],cluster_y_coords[n_child]))
            
            if len(children) != 0:
                queue_dendrogram.extend(children)
                
            queue_dendrogram.remove(n)

        #print("> 3º While")
        
        fig, ax = plt.subplots(figsize=(16, 10))

        # Bars max(bar_widths)
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, self.len_mst))
        sm.set_array([x  for x in bar_widths ])
        bar_colors = [sm.to_rgba(x) for x in bar_widths]
        
        ax.bar(
            bar_centers,
            bar_heights,
            bottom=bar_bottoms,
            width=bar_widths,
            color=bar_colors,
            align='center',
            linewidth=0
        )
        
        for i in range(len(line_xs)):
            ax.plot(*[[line_xs[i][0], line_xs[i][1]], [line_ys[i][0], line_ys[i][1]]], color='black', linewidth=1)
        
        cluster_bounds2 = cluster_bounds
        
        if select_clusters:
            try:
                from matplotlib.patches import Ellipse
            except ImportError:
                raise ImportError('You must have matplotlib.patches available to plot selected clusters.')

            chosen_clusters = selection
            
            # Extract the chosen cluster bounds. If enough duplicate data points exist in the
            # data the lambda value might be infinite. This breaks labeling and highlighting
            # the chosen clusters.
            cluster_bounds = np.array([ cluster_bounds[c] for c in chosen_clusters ])
            
            if not np.isfinite(cluster_bounds).all():
                warn('Infinite lambda values encountered in chosen clusters.'
                     ' This might be due to duplicates in the data.')

            # Extract the plot range of the y-axis and set default center and height values for ellipses.
            # Extremly dense clusters might result in near infinite lambda values. Setting max_height
            # based on the percentile should alleviate the impact on plotting.
            plot_range    = np.hstack([bar_heights, bar_bottoms])
            plot_range    = plot_range[np.isfinite(plot_range)]
            mean_y_center = np.mean([np.max(plot_range), np.min(plot_range)])
            max_height    = np.diff(np.percentile(plot_range, q=[10,90]))

            for c in chosen_clusters:
                c_bounds = cluster_bounds2[c]
                #print("c_bounds: ", c_bounds)
                width  = (c_bounds[1] - c_bounds[0])
                height = (c_bounds[3] - c_bounds[2])
                center = (
                    np.mean([c_bounds[0], c_bounds[1]]),
                    np.mean([c_bounds[3], c_bounds[2]]),
                )
                
                # Set center and height to default values if necessary
                if not np.isfinite(center[1]):
                    center = (center[0], mean_y_center)
                if not np.isfinite(height):
                    height = max_height

                # Ensure the ellipse is visible
                min_height = 0.1*max_height
                if height < min_height:
                    height = min_height

                if selection_palette is not None and \
                        len(selection_palette) >= len(chosen_clusters):
                    oval_color = selection_palette[i]
                else:
                    oval_color = 'r'

                box = Ellipse(
                    center,
                    2.0 * width,
                    1.2 * height,
                    facecolor='none',
                    edgecolor=oval_color,
                    linewidth=2
                )

                if label_clusters:
                    ax.annotate(str(i), xy=center,
                                  xytext=(center[0] - 4.0 * width, center[1] + 0.65 * height),
                                  horizontalalignment='left',
                                  verticalalignment='bottom')

                ax.add_artist(box)

        cb = plt.colorbar(sm, ax=ax)
        cb.ax.set_ylabel('Number of Data Bubbles', fontsize=36)
                                    
        # Cantos do plot
        ax.set_xticks([])
        for side in ('right', 'top', 'bottom'):
            ax.spines[side].set_visible(False)

        ax.invert_yaxis()

        ax.set_ylabel('$\lambda$ value', fontsize=30)

        # Legend
        ax.set_title("Dendrogram", fontsize=34, pad=24)
        #ax.legend(bbox_to_anchor=(0, 1.03, 1, 0.2), loc="lower left", borderaxespad=0, fontsize=28)

        #plt.show()
        
        m_directory = os.path.join(os.getcwd(), "results/dendrograms/dendrograms_t" + str(self.timestamp))
        
        if not os.path.exists(m_directory):
            os.makedirs(m_directory)
                
        fig.savefig("results/dendrograms/dendrograms_t" + str(self.timestamp) + "/mpts_" + str(self.m_mpts) + ".png")
        plt.close()
    
    def getGraphVizString(self):
        newline = "\n"
        tab     = "\t"
        
        sb    = ["graph{" + newline]
        queue = [self.m_root]
        size  = len(queue)
        
        i = 0

        while i < size:
            n = queue[i]
            sb.append(tab + n.getGraphVizString())
            
            numPoint = n.getInternalPoints()
            
            sb.append(str(numPoint) + "}> ")
            
            for v in n.getVertices():
                sb.append(str(v.getID()) + ",")
            
            sb.append("\"];" + newline)
            #sb.append(str(numPoint) + "}\"];" + newline)
            
            children = n.getChildrenMinClusterSize(self.m_minClusterSize)
            #children = n.getChildren()
            
            for child in children:
                numPoint = child.getInternalPoints()
                
                size += 1
                
                queue.append(child)
                    
                sb.append(tab + n.getGraphVizNodeString() + " -- " + child.getGraphVizNodeString())
                sb.append(newline)
            
            i += 1

        sb.append("}")
        
        print(''.join(sb))