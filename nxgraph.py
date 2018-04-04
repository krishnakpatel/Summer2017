



class node(object):
    #__slots__ = ('type')

    def __init__(self, type):
        self.type = type

class diGraph(object):

    #def __init__(self, nodes, edges):
    #for future? when reconstructing from a saved file?
    def __init__(self):
        self.nodes = []
        self.edges = []

#functions: add edges, add nodes, search edges, count types, print graph, other useful things??
    def add_edge(self, edge):
        self.edges.append(edge)

    def add_node(self, node):
        self.nodes.append(node)

    def print(self):
        for e in self.edges:
            print("Edge from %s to %s" % (e.start.type, e.end.type))


class edge(object):

    def __init__(self, start, end):
        self.start = start
        self.end = end

#in order of what should automatically happen
graph = diGraph()

r = node("r")
graph.add_node(r)

j1 = node("j")
graph.add_node(j1)
e1 = edge(r, j1)
graph.add_edge(e1)

c = node("c")
graph.add_node(c)
e2 = edge(j1,c)
graph.add_edge(e2)

R = node("r")
graph.add_node(R)
e3 = edge(j1,R)
graph.add_edge(e3)

j2 = node("j")
graph.add_node(j2)
e4 = edge(R,j2)
graph.add_edge(e4)
e5 = edge(c, j2)
graph.add_edge(e5)

graph.print()
