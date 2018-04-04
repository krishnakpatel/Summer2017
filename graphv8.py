
from xml.etree.ElementTree import ElementTree, Element, SubElement

class pressure_r_equation(object):

    def __init__(self, into, out, r):
        self.p_into = into
        self.p_out = out
        self.r = r
        self.result = 0

    def calculate(self):
        try:
            result = ((self.p_into - self.p_out)/self.r)
            return result
        finally:
            print('cannot calculate pressure at this node')

    # def print(self):
            # where the heck is this equation obejct supooed toita be storyed ti


class nodes(object):

    def __init__(self, pressure):
        self.into = []
        self.out = []
        # self.important = False
        self.pressure = pressure

    #def change_num(self, val):


class edges(object):

    def __init__(self, st, end, type, val):
        self.type = type
        self.value = val
        # self.important = False
        self.start = st
        self.end = end


class graph(object):

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.num_nodes = 0

    def add_edge(self, n1, n2, type, value):
        if (n1 > self.num_nodes):
            start = nodes(n1)
            self.nodes[self.num_nodes] = start
            self.num_nodes += 1
        else:
            start = self.nodes[n1-1]
        if (n2 > self.num_nodes):
            end = nodes(n2)
            self.num_nodes
            self.num_nodes += 1
        else:
            end = self.nodes[n2-1]
        edge = edges(start, end, type, value)
        self.edges.append(edge)

    # update into & out of nodes for calculation
    def update_nodes(self):
        for node in self.nodes:
            for edge in self.edges:
                if edge.start.num == node.num:
                    node.out.append(edge)
                elif edge.end.num == node.num:
                    node.into.append(edge)

   # def remove_edge(self):

    def open_saved(self,file_name):
        file = open(file_name, 'rb')
        tree = ElementTree()
        tree.parse(file)
        root = tree.getroot()
        root_dict = {}
        wires_str = root.findtext('wires')
        wires = wires_str.split( )
        node_num = 0
        for wire in wires:
            list = wire.split(',')
            #for x, y in list:
                #if (found):
                    #add there
                #if (not found):
                    #root_dict[node_num] = tuple(x, y)



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

    def is_connected(self):
        for node in self.nodes:
            check_list = self.nodes
            for edge in node.into:
                if edge.start in check_list: # modify based on how you end up storing things
                    check_list.remove(edge.start)
            for edge in node.out:
                if edge.end in check_list:
                    check_list.remove(edge.end)
            if check_list:
                print('graph is not fully connected')


graph = graph()
# load into graph
# create a dictionary
# condense the dictionary
# create edges/nodes
graph.add_edge(1, 2, 'r', 10)
graph.add_edge(2, 3, 'c', 2)
graph.add_edge(2, 4, 'r', 10)
graph.is_connected()
graph.update_nodes()
