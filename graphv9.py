
import networkx as nx
from xml.etree.ElementTree import ElementTree, Element, SubElement

class flow_r(object):

    def __init__(self, p_in, p_out, resistance):
        self.p_0 = p_in
        self.p_d = p_out
        self.r = resistance

class flow_c(object):

    def __init__(self, p_in, p_out, capacitance):
        self.p_0 = p_in
        self.p_d = p_out
        self.c = capacitance

class flow_i(object):

    def __init__(self, p_in, p_out, inductance):
        self.p_0 = p_in
        self.p_d = p_out
        self.i = inductance

class circuit(object):

    # as soon as the program loads
    def __init__(self):
        self.circ = nx.MultiDiGraph()
        self.num_node = 0
        self.num_key = 0

    # update into & out of nodes for calculation
    def update_nodes(self):
        for node in self.circ.nodes_iter(data=True):
            edges = self.circ.edges(node)
            for edge in edges:
                if edge[0] == node[0]:
                    node[1]['out'].append(edge)
                else:
                    node[1]['into'].append(edge)

    def find_node(self, tuple, dict):
        for key, value in dict.items():
            if value == tuple:
                return key

    # open from a saved file
    def open_saved(self, file_name):
        file = open(file_name, 'rb')
        tree = ElementTree()
        tree.parse(file)
        root = tree.getroot()
        node_dict = {}
        num_node = 1
        nodes_str = root.findtext('nodes')
        for node in nodes_str.split():
            list = node.split(',')
            node_dict[num_node] = (list[0], list[1])
            self.circ.add_node(num_node, into=[], out=[], p=list[2])
            num_node += 1
        # fix this
        wires = root.findtext('wires')
        for wire in wires.split():
            list = wire.split(',')
            self.circ.add_edge(self.find_node((list[0], list[1]), node_dict), self.find_node((list[2], list[3]),
                                node_dict), type='w', value=0) # go back and catch errors based on input
        resistors = root.findtext('resistors')
        for r in resistors.split():
            list = r.split(',')
            self.circ.add_edge(self.find_node((list[0], list[1]), node_dict), self.find_node((list[2], list[3]),
                                                                                             node_dict), type='r',
                               value=list[4])  # go back and catch errors based on input
        capacitors = root.findtext('capacitors')
        for c in capacitors.split():
            list = c.split(',')
            self.circ.add_edge(self.find_node((list[0], list[1]), node_dict), self.find_node((list[2], list[3]),
                                                                                             node_dict), type='c',
                               value=list[4])  # go back and catch errors based on input
        inductors = root.findtext('inductors')
        for i in inductors.split():
            list = i.split(',')
            self.circ.add_edge(self.find_node((list[0], list[1]), node_dict), self.find_node((list[2], list[3]),
                                                                                             node_dict), type='i',
                               value=list[4])  # go back and catch errors based on input



# something that condenses nodes
       # for wire in wires_str.split( ):
        #    list = wire.split(',')
       #     t1 = (list[0], list[1])
      #      t2 = (list[2], list[3])
       #     added = False
     #       for key, value in node_dict:
      #          for v in value:
      #              if v == t1 or v == t2:
      #                  v.append(t1)
      #                  v.append(t2)
    #                    added = True
      #      if not added:
    #            node_dict[num_node] = [t1, t2]
    #            num_node += 1
     #   for x in range(1, num_node): # off by 1? double check
    #        self.circ.add_node(x, into=[], out=[])


        # load in first set of tuples
        # search for first one in the dictionary
        # search for second one in dictionary
        # if either are values in the dictionary, add both to that key
        # else add them as newest node


     #   num_node = root.findtext('max_num_node')
     #   for start in root.find('edges_list'):
    #        node1 = start.attrib['at']
     #       for end in start:
     #           node2 = end.attrib['at']
    #            edges_str = ''.join(end.itertext())
     #           edges = edges_str.split( )
      #          if bool(edges):
      #              for e in edges:
      #                  dict = e.split(',')
     #                   self.circ.add_edges_from([(node1, node2, {'important': dict[0], 'type': dict[1],
     #                                                             'value': dict[2], 'pos': dict[3]})])

    def is_connected(self):
        for node in self.circ.nodes_iter():
            nodes = self.circ.nodes()
            for n in self.circ.neighbors_iter(node):
                nodes.remove(n)
                for x in self.circ.neighbors_iter(n):
                    nodes.remove(n)
                    if not nodes:
                        return
                    else:
                        print('not connected')




    # properly closes the graph
    def close(self):
        self.save()
        self.circ.clear()

c = circuit()
c.open_saved('test_input.xml')