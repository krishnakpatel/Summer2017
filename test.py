import networkx as nx

class nodes(object):

    def __init__(self, num):
        self.num = num
        self.into = []
        self.out = []
        self.important = False
        self.pressure = None

n1 = nodes(1)
n2 = nodes(2)
n3 = nodes(3)
n4 = nodes(4)
list = [n1, n2, n3, n4]
print(list)
list.remove(n3)
print(list)

graph = nx.MultiDiGraph()

graph.add_node(1)
graph.add_node(2)
graph.add_edge(1, 2, key=0, type='c', value=10)
graph.add_edge(1, 2, key=1, type='r', value=6)



l = graph.get_edge_data(1,2,0)
z = l['value']
a = graph.edges()
one = a[1]
print('Nodes are', one[0], 'and', one[1])
print(graph.edge)
graph.add_edge(3, 4, key=1, type='c', value=10)
print(graph.edge)
print(graph.node)

print('hi')

for edge in graph.edges_iter(data=True):
    print(edge)

print(0.03 + 0.005)