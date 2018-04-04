
import networkx as nx
from xml.etree.ElementTree import ElementTree, Element, SubElement

class flow_r(object):

    def __init__(self, p_in, p_out, resistance, num):
        self.p_0 = p_in
        self.p_d = p_out
        self.r = resistance
        self.num = num
        # self.flow = eval(((self.p_0 - self.p_d)/self.r))

    def calculate(self):
        return eval(((self.p_0 - self.p_d)/self.r))

    def print_eq(self):
        print("Q" + str(self.num) + " = " + "(" + str(self.p_0) + " - " + str(self.p_d) + ")" + "/" + str(self.r))

class flow_c(object):

    def __init__(self, q_in, q_out, capacitance, num):
        self.q_in = q_in
        self.q_out = q_out
        self.c = capacitance
        self.num = num

    def print_eq(self):
        print("dP" + str(self.num) + "/dt = (" + str(self.q_in) + " - " + str(self.q_out) + ")/" + str(self.c))

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
        self.wires = []
        self.capacitors = []

    # update into & out of nodes for calculation
    def update_nodes(self):
        for node in self.circ.nodes_iter(data=True):
            edges = self.circ.edges(keys=True)
            for edge in edges:
                if edge[0] == node[0]:
                    node[1]['out'].append(edge)
                elif edge[1] == node[0]:
                    node[1]['into'].append(edge)

    # must be called before calling update_capacitor_flow & update nodes
    def prune(self):
        wires = []
        node_dict = {}
        # find all wires
        for edge in self.circ.edges(keys=True, data=True):
            if edge[3]['type'] == 'w':
                wires.append(edge)
                self.circ.remove_edge(edge[0], edge[1], edge[2])
        # create the node_dict
        for wire in wires:
            node_1 = wire[0] in node_dict
            node_2 = wire[1] in node_dict
            if node_1 and node_2:
                use = node_dict[wire[0]]
                replace = node_dict[wire[1]]
                for key in node_dict:
                    if node_dict[key] == replace:
                        node_dict[key] = use
            elif node_1:
                node_dict[wire[1]] = node_dict[wire[0]]
            elif node_2:
                node_dict[wire[0]] = node_dict[wire[1]]
            else:
                node_dict[wire[0]] = wire[0]
                node_dict[wire[1]] = wire[0]
        # replace nodes
        for edge in self.circ.edges(keys=True, data=True):
            if edge[0] in node_dict:
                self.circ.remove_edge(edge[0], edge[1], edge[2])
                self.circ.add_edge(node_dict[edge[0]], edge[1], key=edge[2], type=edge[3]['type'], value=edge[3]['value'], flow=edge[3]['flow'])
            elif edge[1] in node_dict:
                self.circ.remove_edge(edge[0], edge[1], edge[2])
                self.circ.add_edge(edge[0], node_dict[edge[1]], key=edge[2], type=edge[3]['type'], value=edge[3]['value'], flow=edge[3]['flow'])
        # delete nodes
        out_degrees = self.circ.out_degree()
        in_degrees = self.circ.in_degree()
        to_delete = [n for n in out_degrees if out_degrees[n] == 0 and in_degrees[n] == 0]
        self.circ.remove_nodes_from(to_delete)


    def update_capacitor_flow(self):
        for edge in self.circ.edges_iter(keys=True, data=True):
            if edge[3]['type'] == 'c':
                nodes = self.circ.node
                edges = self.circ.edge
                into = nodes[edge[0]]['into']
                out = nodes[edge[0]]['out']
                out.remove((edge[0], edge[1], edge[2]))
                q_in = []
                q_out = []
                if len(into) >= 1: # is there the possibility that it would be 0??
                    for obj in into:
                        # q_in.append(edges[obj[0]][obj[1]][obj[2]]['flow'].flow)
                        q_in.append(edges[obj[0]][obj[1]][obj[2]]['flow'])
                if len(out) >= 1:
                    for obj in out:
                        # q_out.append(edges[obj[0]][obj[1]][obj[2]]['flow'].flow)
                        q_out.append(edges[obj[0]][obj[1]][obj[2]]['flow'])
                # figure out how to add the flows
                f = flow_c(q_in, q_out, edge[3]['value'], edge[2])
                edge[3]['flow'] = f

    # open from a saved file
    def open_saved(self, file_name):
        file = open(file_name, 'rb')
        tree = ElementTree()
        tree.parse(file)
        root = tree.getroot()
        node_dict = {}
        for node in root.find('nodes'):
            list = ''.join(node.itertext()).split(',') # test
            node_dict[(list[0], list[1])] = node.attrib['num']
            self.circ.add_node(node.attrib['num'], into=[], out=[], p=list[2])
        wires = root.findtext('wires')
        for wire in wires.split( ):
            list = wire.split(',')
            self.circ.add_edge(node_dict[(list[0], list[1])], node_dict[(list[2], list[3])],
                                                     type='w', value=0)
        for resistor in root.find('resistors'):
            list = ''.join(resistor.itertext()).split(',')  # read in p_in, p_out vectors
            f = flow_r('p_in', 'p_out', list[4], resistor.attrib['num']) #node 1 & node2 ????? saved ahead of time
            self.circ.add_edge(node_dict[(list[0], list[1])], node_dict[(list[2], list[3])]
                                            , key=resistor.attrib['num'],type='r', value=list[4], flow=f)
        for capacitor in root.find('capacitors'):
            list = ''.join(capacitor.itertext()).split(',')
            self.circ.add_edge(node_dict[(list[0], list[1])], node_dict[(list[2], list[3])],
                                        key=capacitor.attrib['num'], type='c', value=list[4], flow=None)  # go back and catch errors based on input
       # inductors = root.findtext('inductors')
       # for i in inductors.split():
       #     list = i.split(',')
        #    self.circ.add_edge(self.find_node((list[0], list[1]), node_dict), self.find_node((list[2], list[3]),
        #                                                                                     node_dict), type='i',
        #                       value=list[4])  # go back and catch errors based on input


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

    def boundary_conditions(self, pressure_files):
        boundary_c = {}
        for node in pressure_files:
            pressures = {}
            file = open(pressure_files[node], "r")
            for line in file:
                entry = line.split()
                pressures[entry[0]] = entry[1]
            boundary_c[node] = pressures
        return boundary_c



c = circuit()
c.open_saved('Parallel.xml')
c.prune()
c.update_nodes()
c.update_capacitor_flow()
for edge in c.circ.edges(data=True):
    print(edge)
    edge[2]['flow'].print_eq()