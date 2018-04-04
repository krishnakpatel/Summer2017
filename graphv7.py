# add functionality: modifying values
# can use pos to save position when graph is saved
# can create another object for pos
# can store a dictionary of position for easy access

import networkx as nx
from xml.etree.ElementTree import ElementTree, Element, SubElement


class circuit(object):

    # as soon as the program loads
    def __init__(self):
        self.circ = nx.MultiDiGraph()
        self.num_node = 0
        self.num_key = 0

    # component generation
    # key must be included if constructing something in parallel
    # private function
    def click(self, type, value, **key):
        self.circ.add_node(self.num_node, important=False, into=[], out=[], pos='', value=0)
        self.circ.add_node(self.num_node+1, important=False, into=[], out=[], pos='', value=0)
        if 'key' in key:
            k = key['key']
        else:
            k = self.num_key + 1
            self.num_key += 1
        self.circ.add_edge(self.num_node, self.num_node+1, key=k, important=False, type=type, value=value, pos=' ')
        self.num_node += 2

    # click on a resistor
    def resistor(self, value):
        self.click('r', value)

    # click on a capacitor
    def capacitor(self, value):
        self.click('c', value)

    # click on an inductor
    def inductor(self, value):
        self.click('i', value)

    # just adding wire
    def wire(self):
        self.click('w', 0)

    # mark a node as important
    def node_important(self, pos):
        for node in self.circ.nodes_iter(data=True):
            if node['pos'] == pos:  # essentially, probably will need to be a function later
                node['important'] = True

    # mark a node as unimportant
    def node_unimportant(self, pos):
        for node in self.circ.nodes_iter(data=True):
            if node['pos'] == pos:
                node['important'] = False

    # set value of a node
    def node_value(self, value, pos):
        for node in self.circ.nodes_iter(data=True):
            if node['pos'] == pos:
                node['value'] = value

    # mark an edge as important
    def edge_important(self, pos):
        for edge in self.circ.edges_iter(data=True):
            if edge[2]['pos'] == pos:
                edge[2]['important'] = True

    # mark an edge as unimportant
    def edge_unimportant(self, pos):
        for edge in self.circ.edges_iter(data=True):
            if edge[2]['pos'] == pos:
                edge[2]['important'] = False

    # joining two components
    # edge1[1] & edge2[0] must be the nodes connecting
    def connect(self, e1, e2):  # tbh probably going to get two positions passed in
        # just so it works rn
        for edge in self.circ.edges_iter(data=True):
            if edge[0] == e1[0] and edge[1] == e1[1]:
                edge1 = edge
            elif edge[0] == e2[0] and edge[1] == e2[1]:
                edge2 = edge
        # CLEAN THIS UP
        self.circ.add_edge(edge1[1], edge2[1], key=edge2['key'], important=edge2['important'], type=edge2['type'],
                           value=edge2['value'], pos=edge2['pos'])
        for edge in self.circ.edges(edge2[0]):
            # if node to be removed is the first node in edge
            if edge[0] == edge2[0]:
                self.circ.add_edge(edge1[1], edge[1], key=edge['key'], important=edge['important'], type=edge['type'],
                                   value=edge['value'], pos=edge['pos'])
            # if node to be removed is the second node in edge
            else:
                self.circ.add_edge(edge[0], edge1[1], key=edge['key'], important=edge['important'], type=edge['type'],
                                   value=edge['value'], pos=edge['pos'])
            self.circ.remove_edge(edge[0], edge[1], key=edge['key'])

    # splitting components apart---edge being dragged away must be second argument
    def cleave(self, stationary, dragged):
        if stationary[1] == dragged[0]:
            self.circ.add_edge(self.num_node+1, dragged[1], key=dragged['key'], important=dragged['important'],
                               type=dragged['type'], value=dragged['value'], pos=dragged['pos'])
        elif stationary[0] == dragged[1]:
            self.circ.add_edge(dragged[0], self.num_node + 1, key=dragged['key'], important=dragged['important'],
                               type=dragged['type'], value=dragged['value'], pos=dragged['pos'])
        self.num_node += 1
        self.circ.remove_edge(dragged[0], dragged[1], key=dragged['key'])

    # delete a component
    def delete(self, edge):
        start = self.circ.edges(edge[0])
        end = self.circ.edges(edge[1])
        if not start and not end:
            self.circ.remove_node(edge[0])
            self.circ.remove_node(edge[1])
            self.circ.remove_edge(edge[0],edge[1])

    # update into & out of nodes for calculation
    def update_nodes(self):
        for node in self.circ.nodes_iter(data=True):
            edges = self.circ.edges(node)
            for edge in edges:
                if edge[0] == node[0]:
                    node[1]['out'].append(edge)
                else:
                    node[1]['into'].append(edge)

    # save a graph that has been created
    # def save(file_name):
    def save(self):
        file = open('test.xml', 'wb')
        root = Element('circuit')
        document = ElementTree(root)
        nodes = SubElement(root, 'max_num_node')
        nodes.text = str(self.num_node)
        edges_list = SubElement(root, 'edges_list')
        edges = self.circ.edge
        for start, dicts in edges.items():
            if bool(dicts):
                s = SubElement(edges_list, 'start', {'at': str(start)})
                for end, keys in dicts.items():
                    e = SubElement(s, 'end', {'at': str(end)})
                    string = ''
                    for key, data in keys.items(): # maybe you should stop ignoring keys
                        for t, v in data.items():
                            if not isinstance(v, str):
                                v = str(v)
                            string += v
                            string += ','
                        string += ' '
                    e.text = string
        document.write(file, encoding='UTF-8', xml_declaration=True)
        file.close()

    # open from a saved file
    def open_saved(self,file_name):
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
                    for e in edges:
                        dict = e.split(',')
                        self.circ.add_edges_from([(node1, node2, {'important': dict[0], 'type': dict[1], 'value': dict[2],
                                                              'pos': dict[3]})])
                    # how to integrate this to show a visual picture
                    # you'll know that the nodes of this segment are all connected w/ the same start node

    # properly closes the graph
    def close(self):
        self.save()
        self.circ.clear()

g = circuit()
g.resistor(10)
g.capacitor(5)
g.resistor(16)
g.connect((0, 1), (2, 3))
g.connect((4, 5), (6, 7))

