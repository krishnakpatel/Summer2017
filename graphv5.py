
#add functionality: modifying values
import matplotlib.pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element, SubElement, tostring


# as soon as the program loads

circ = nx.MultiDiGraph()

node_num = 1
key = 0

def click(type):
    circ.add_node(node_num)
    circ.add_node(node_num+1)
    # print(What is it's value?)
    circ.add_edge(node_num, (node_num+1),key=key+1,type=type)
    node_num += 2
    key += 1

# click on a resistor
def resistor():
    click('r')

# click on an inductor
def inductor():
    click('i')

# click on a capacitor
def capacitor():
    click('c')

#two components dragged together(ends join together)
def join(edge1, edge2):
    old_node = edge2[0]
    edge2[0] = edge1[1]
    to_change = circ.edges(old_node)
    for edge in to_change:
        edge[0] = edge1[1]
    circ.remove_edge(old_node)
    # figure out which node is the last node of edge1
        #edge1[1]
    # figure out which node is the first node of edge2
        #edge2[0]
    # change edge2's first node to the last node of edge1
    # find any other edges connected to edge2's start
    # change all of them to the edge1's last node
    # delete the first node of edge2

# two components dragged apart (ends split) the one being dragged MUST be edge2
def split(stationary, dragged):
    # figure which node they have in common

    if stationary[1] == dragged[0]:
        dragged[0] = node_num+1
        node_num += 1
        return
    elif stationary[0] == dragged[1]:
        dragged[1] = node_num+1
        node_num += 1
    # if it's edge2's first node
        # change edge2's first node to num_node+1
        # num_node++
    # if it's edge2's second node
        # change edge2's second node to num_node+1
        # num_node++

# delete a component
def delete(edge):
    start = circ.edges(edge[0])
    end = circ.edges(edge[1])
    if not start:
        if not end:
            circ.remove_node(edge[0])
            circ.remove_node(edge[1])
            circ.remove_edge(edge)
    # check that it has start & end nodes that no other nodes are connected to
    # remove edge
    # remove start & end nodes


for x in range(1,8):
    circ.add_node(x)

circ.add_edges_from([(1, 2, {'type': 'r', 'value': 25}), (1, 2, {'type': 'r', 'value': 50}), (1, 5, {'type': 'r', 'value': 77}), (1, 3,{'type': 'r', 'value': 85}),(2, 7,{'type': 'r', 'value': 63}), (6, 7,{'type': 'r', 'value': 50}), (3, 4,{'type': 'r', 'value': 200}), (4, 5,{'type': 'r', 'value': 100}), (4, 6,{'type': 'r', 'value': 85})])



# CTRL + S/click on save
def save():
    # added functionality to ask for file name (Save As...)
    file = open('test.xml','wb')
    root = Element('circuit')
    document = ElementTree(root)
    nodes = SubElement(root, 'max_node_num')
    nodes.text = str(node_num)
    edges_list = SubElement(root,'edges_list')
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

    #list = [e for e in circuit.edges_iter(data=True)]

    #for elem in list:
        #edge = SubElement(edges_list, 'edge', {'data':elem})


  #  print(tostring(file, encoding='UTF-8', xml_declaration=True))
  #  print(tostring(root, encoding='UTF-8'))
    document.write(file, encoding='UTF-8', xml_declaration=True)
    #print(prettify(root))
    #file.write(tostring(root))
    file.close()


save()
circ.clear()


# open from saved file
# read in selected HML file use ElemenTree HML parser call create_graph
def open_saved(file_name):
    file = open(file_name, 'rb')
    tree = ElementTree()
    tree.parse(file)

   # root = tree.getroot()
   # for child in root:
   #     print(child.tag, child.attrib)
    #tree = ET.parse(file_name)
    root = tree.getroot()
    node_num = root.findtext('max_node_num')
    for start in root.find('edges_list'):
        node1 = start.attrib['at']
        for end in start:
            node2 = end.attrib['at']
            edges_str = ''.join(end.itertext())
            edges = edges_str.split( )
            if bool(edges):
                first = edges[0]
                sec = edges[1:]
                circ.add_edges_from([(node1, node2, {'type': first, 'value': sec})])



       # l = l.replace('\\', '')
       # result = bytes(l, "utf-8").decode("unicode_escape")
        #result = l.translate({ord('\\'):None})

        #circuit.add_edges_from(result)

#    print(node_num)

   # print(tostring(root))
   # print(root.tag)
  #  for child in root:
        # print(child.tag)
      #  for c in child:
            # print(c.tag, c.attrib)



open_saved('test.xml')


pos = nx.random_layout(circ)
nx.draw_networkx_nodes(circ,pos,node_size=700)
nx.draw_networkx_edges(circ,pos,width=1.0)
nx.draw_networkx_labels(circ,pos)


# show graph
plt.show()

