# add functionality: modifying values

import networkx as nx
from xml.etree.ElementTree import ElementTree, Element, SubElement


# as soon as the program loads
def initialize():
    global circ
    global num_node
    circ = nx.MultiDiGraph()
    num_node = 0


# component generation 
def click(type, value):
    circ.add_node(num_node)
    circ.add_node(num_node+1)
    # ask for value?
    circ.add_edges_from([(num_node, num_node+1, {'type': type, 'value': value})])
    num_node += 2


# click on a resistor
def resistor(value):
    click('r', value)


# click on a capacitor
def capacitor(value):
    click('c', value)


# click on an inductor
def inductor(value):
    click('i', value)


# just adding wire
def wire():
    click('w', 0)


# joining two components
def connect(edge1, edge2):
    old_node = edge2[0]
    edge2[0] = edge1[1]
    to_change = circ.edges(old_node)
    for edge in to_change:
        edge[0] = edge1[1]
    circ.remove_edge(old_node)


# splitting components apart---edge being dragged away must be second argument
def cleave(stationary, dragged):
    if stationary[1] == dragged[0]:
        dragged[0] = num_node+1
    elif stationary[0] == dragged[1]:
        dragged[1] = num_node+1
    num_node += 1


# delete a component
def delete(edge):
    start = circ.edges(edge[0])
    end = circ.edges(edge[1])
    if not start and not end:
        circ.remove_node(edge[0])
        circ.remove_node(edge[1])
        circ.remove_edge(edge)


# save a graph that has been created
# def save(file_name):
def save():
    file = open('test.xml', 'wb')
    root = Element('circuit')
    document = ElementTree(root)
    nodes = SubElement(root, 'max_num_node')
    nodes.text = str(num_node)
    edges_list = SubElement(root, 'edges_list')
    edges = circ.edge
    for start, dicts in edges.items():
        if bool(dicts):
            s = SubElement(edges_list, 'start', {'at': str(start)})
            for end, keys in dicts.items():
                e = SubElement(s, 'end', {'at': str(end)})
                string = ''
                for key, data in keys.items():
                    for t, v in data.items():
                        if not isinstance(v, str):
                            v = str(v)
                        string += v
                    string += ' '
                e.text = string
    document.write(file, encoding='UTF-8', xml_declaration=True)
    file.close()


# open from a saved file
def open_saved(file_name):
    file = open(file_name, 'rb')
    tree = ElementTree()
    tree.parse(file)
    root = tree.getroot()
    num_node = root.findtext('max_num_node')
    for start in root.find('edges_list'):
        node1 = start.attrib['at']
        for end in start:
            node2 = end.attrib['at']
            edges_str = ''.join(end.itertext())
            edges = edges_str.split( )
            if bool(edges):
                circ.add_edges_from([(node1, node2, {'type': edges[0], 'value': edges[1:]})])
                # how to integrate this to show a visual picture
                # you'll know that the nodes of this segment are all connected w/ the same start node


# properly closes the graph
def close():
    save()
    circ.clear()

# only for plot/delete this later
import matplotlib.pyplot as plot
initialize()
circ.add_node(num_node+1, important=False, into=[])
for node in circ.nodes_iter(data=True):
    if node[1]['into'] == 56:
        node[1]['important'] = True
# circ.add_edges_from([(1, 2, {'type': 'r', 'value': 25}), (1, 2, {'type': 'r', 'value': 50}), (1, 5, {'type': 'r', 'value': 77}), (1, 3,{'type': 'r', 'value': 85}),(2, 7,{'type': 'r', 'value': 63}), (6, 7,{'type': 'r', 'value': 50}), (3, 4,{'type': 'r', 'value': 200}), (4, 5,{'type': 'r', 'value': 100}), (4, 6,{'type': 'r', 'value': 85})])
# circ.add_edges_from([(1, 2, {'type': 'r', 'value': 3}), (2, 3, {'type': 'r', 'value': 1}), (3, 4, {'type': 'r', 'value': 4}), (4, 5,{'type': 'r', 'value': 3}),(5, 6,{'type': 'r', 'value': 1}), (3, 5,{'type': 'r', 'value': 7}), (2, 6,{'type': 'r', 'value': 6}), (6, 1,{'type': 'r', 'value': 1})])
circ.add_edges_from([(1, 2, {'type': 'w', 'value': 0}), (2, 3, {'type': 'r', 'value': 50}), (2, 5, {'type': 'r', 'value': 77}), (2, 5,{'type': 'r', 'value': 85}),(2, 7,{'type': 'r', 'value': 63}), (5, 7,{'type': 'r', 'value': 50}), (7, 3,{'type': 'r', 'value': 200}), (3, 1,{'type': 'r', 'value': 100})])
edge = circ.edges()[1]
edge2 = circ.edges()[0]
for edge in circ.edges_iter(data=True):
    if edge['pos'] == 0:
        edge['important'] = False
for node in circ.nodes_iter(data=True):
    edges = circ.edges(node[0])
    for edge in edges:
       # if edge[0] == node[0]:
        #    node[1]['out'].append(edge)
      #  else:
            node[1]['into'].append(edge)
pos = nx.random_layout(circ)
nx.draw_networkx_nodes(circ,pos,node_size=700)
nx.draw_networkx_edges(circ,pos,width=1.0)
nx.draw_networkx_labels(circ,pos)
save()
circ.clear()
open_saved('test.xml')
plot.show()



# only for plot/delete this later
# import matplotlib.pyplot as plot
# c = circuit()
# circ.add_edges_from([(1, 2, {'type': 'r', 'value': 25}), (1, 2, {'type': 'r', 'value': 50}), (1, 5, {'type': 'r', 'value': 77}), (1, 3,{'type': 'r', 'value': 85}),(2, 7,{'type': 'r', 'value': 63}), (6, 7,{'type': 'r', 'value': 50}), (3, 4,{'type': 'r', 'value': 200}), (4, 5,{'type': 'r', 'value': 100}), (4, 6,{'type': 'r', 'value': 85})])
# circ.add_edges_from([(1, 2, {'type': 'r', 'value': 3}), (2, 3, {'type': 'r', 'value': 1}), (3, 4, {'type': 'r', 'value': 4}), (4, 5,{'type': 'r', 'value': 3}),(5, 6,{'type': 'r', 'value': 1}), (3, 5,{'type': 'r', 'value': 7}), (2, 6,{'type': 'r', 'value': 6}), (6, 1,{'type': 'r', 'value': 1})])
# circ.add_edges_from([(1, 2, {'type': 'w', 'value': 0}), (2, 3, {'type': 'r', 'value': 50}), (2, 5, {'type': 'r', 'value': 77}), (2, 5,{'type': 'r', 'value': 85}),(2, 7,{'type': 'r', 'value': 63}), (5, 7,{'type': 'r', 'value': 50}), (7, 3,{'type': 'r', 'value': 200}), (3, 1,{'type': 'r', 'value': 100})])
# pos = nx.random_layout(circ)
# nx.draw_networkx_nodes(circ,pos,node_size=700)
# nx.draw_networkx_edges(circ,pos,width=1.0)
# nx.draw_networkx_labels(circ,pos)
# save()
# circ.clear()
# open_saved('test.xml')
# plot.show()