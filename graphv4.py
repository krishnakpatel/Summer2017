import networkx as nx
import matplotlib.pyplot as plt

circuit = nx.MultiDiGraph()

for x in range(1,8):
    circuit.add_node(x)

circuit.add_edges_from([(1, 2,{'type': 'r', 'value': 25}), (1, 2,{'type': 'r', 'value': 50}), (1, 5,{'type': 'r', 'value': 77}), (1, 3,{'type': 'r', 'value': 85}),(2, 7,{'type': 'r', 'value': 63}), (6, 7,{'type': 'r', 'value': 50}), (3, 4,{'type': 'r', 'value': 200}), (4, 5,{'type': 'r', 'value': 100}), (4, 6,{'type': 'r', 'value': 85})])

pos = nx.random_layout(circuit)
nx.draw_networkx_nodes(circuit,pos,node_size=300)
nx.draw_networkx_edges(circuit,pos,width=1.0)
nx.draw_networkx_labels(circuit,pos)

#nx.draw(circuit, pos)

# show graph
plt.show()