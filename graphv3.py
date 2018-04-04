import networkx as nx

graph = nx.MultiDiGraph()

# class node(object):

    # def __init__(self, num):
        # self.num = num

# class edge(object):

    # def __init__(self, type, value):
        # self.type = type
        # self.value = value

# n1 = node(1)
# graph.add_node(n1)
#  e1 = edge('c', 10)
# n2 = node(2)
# graph.add_node(n2)
# graph.add_edge(n1, n2, object=e1)  # is the way you added an edge object correct
# e2 = edge('r', 6)
# graph.add_edge(n1, n2, object=e2)

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



#print(graph.number_of_edges())
#print(graph.number_of_nodes())

#a = list(graph.nodes())
#for node in a:
   # print(node)

