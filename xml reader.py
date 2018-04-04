import networkx as nx
from xml.etree.ElementTree import ElementTree


class flow_r(object):

    def __init__(self, start, end, value):
        self.node_in = start # somehow import the dictionary of pressures if it exists
        self.node_out = end
        self.resistance = value

    def calc_flow(self, time, solution_var, special_nodes):
        p_in = 0
        p_out = 0
        if self.node_in in solution_var:
            p_in = solution_var[self.node_in]
        elif bool(self.node_in['p']):
            p_in = self.node_in['p'][time]
        else:
            # error handling
            print('error')

        if self.node_out in solution_var:
            p_out = solution_var[self.node_out]
        elif bool(self.node_out['p']):
            p_in = self.node_out['p'][time]
        else:
            # error handling
            print('error')

        return ((p_in - p_out)/self.resistance)


class flow_c(object):

    def __init__(self, value, proximal, distal):
        self.capacitance = value
        self.prox_node = proximal
        self.distal_node = distal

    def calc(self, time, solution_var, special_nodes):
        q_in = self.prox_node.calc_in(time, solution_var, special_nodes)
        q_out = self.distal_node.calc_out(time, solution_var, special_nodes)
        return ((q_in - q_out)/self.capacitance)


class circiut(object):

    def __init__(self):
        self.circ = nx.MultiDiGraph()
        self.metric_dict = {'E': 10**18, 'P': 10**15, 'T': 10**12, 'G': 10**9, 'M': 10**6, 'k': 1000, 'h': 100,
                            'da': 10, 'd': 0.1, 'c': 0.01, 'm': 0.001, 'µ': 10**(-6), 'n': 10**(-9), 'p': 10**(-12),
                            'f': 10**(-18), '': 1}
        self.node_dict = {}
        self.num_nodes = 0
        self.ODEs = []

    def find_node(self, node):
        if node not in self.node_dict:
            self.num_nodes += 1
            self.node_dict[node] = self.num_nodes
            self.circ.add_node(self.num_nodes, into=[], out=[], p=None)
        return self.node_dict[node]

    def load_node(self, comp):
        file_name = comp.attrib['name']
        node = comp.find('Node').text
        pressures = {}
        file = open(file_name, "r")
        for line in file:
            entry = line.split()
            pressures[entry[0]] = entry[1]
        self.circ.node[self.find_node(node)]['p'] = pressures

    def component_extract(self, comp):
        value = comp.attrib['value'] * self.metric_dict[comp.attrib['metricPrefix']]
        nodes = comp.findall('Node')
        if not len(nodes) == 2:
            # send out error message
            print('error')
        n = []
        for x in nodes:
            n.append(self.find_node(x.text))
        return n, value

    def calc_c_flow(self, node, type, time, solution_vars, special_nodes):
        flow = 0
        for edge in self.circ.node[node][type]:
            flow += edge[3]['calculate'](edge, time, solution_vars, special_nodes) # proper syntax???
        return flow

    def calc_c(self, edge, time, solution_vars, special_nodes):
        q_in = self.calc_c_flow(edge[0], 'into', time, solution_vars, special_nodes)
        q_out = self.calc_c_flow(edge[1], 'out', time, solution_vars, special_nodes)
        return (q_in - q_out)/(edge[3]['value'])

    # def calc_i(self, edge):

    def calc_r(self, edge, time, solution_vars, special_nodes):
        p_in = 0
        p_out = 0
        if edge[0] in special_nodes:
            p_in = solution_vars[special_nodes.index(edge[0])]
        elif bool(self.circ.node[edge[0]]['p']):
            p_in = self.circ.node[edge[0]]['p'][time]
        else:
            print('error')
            # send error message
        if edge[1] in special_nodes:
            p_in = solution_vars[special_nodes.index(edge[1])]
        elif bool(self.circ.node[edge[1]]['p']):
            p_in = self.circ.node[edge[1]]['p'][time]
        else:
            print('error')
            # send error message
        return (p_in - p_out)/(edge[3]['value'])

    def load_wires(self, pos_list):
        start = self.find_node(str('(' + pos_list[0] + ', ' + pos_list[1] + ')'))
        end = self.find_node(str('(' + pos_list[2] + ', ' + pos_list[3] + ')'))
        self.circ.add_edge(start, end, type='w')

    def load_inductor(self, component):
        extracted = self.component_extract(component)
        self.circ.add_edge(extracted[0], extracted[1], type='i', value=extracted[2], calculate=None)

    def load_capacitor(self, component):
        extracted = self.component_extract(component)
        self.circ.add_edge(extracted[0], extracted[1], type='c', value=extracted[2], calculate=self.calc_c)

    def load_resistor(self, component):
        extracted = self.component_extract(component)
        self.circ.add_edge(extracted[0], extracted[1], type='r', value=extracted[2], calculate=self.calc_r)

    def open_saved(self, file_name):
        file = open(file_name, 'rb')
        tree = ElementTree()
        tree.parse(file)
        root = tree.getroot()
        # read in components
        for c in root.findall('Component'):
            type = c.attrib['type']
            if type == 'Inductor':
                self.load_inductor(c)
            elif type == 'Capacitor':
                self.load_capacitor(c)
            elif type == 'Resistor':
                self.load_resistor(c)
            elif type == 'Boundary Face':
                self.load_node(c)
            else:
                # send error message
                print('error')
        # read in wires
        for wire in root.findall('wire'):
            pos = wire.find('wirePos').text.split(', ')
            self.load_wires(pos)

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
            data_1 = bool(self.circ.node[wire[0]]['p'])
            data_2 = bool(self.circ.node[wire[1]]['p'])
            if node_1 and node_2:
                if data_1:
                    use = node_dict[wire[0]]
                    replace = node_dict[wire[1]]
                elif data_2:
                    use = node_dict[wire[1]]
                    replace = node_dict[wire[0]]
                else:
                    # send out error message
                    print('error')
                for key in node_dict:
                    if node_dict[key] == replace:
                        node_dict[key] = use
            elif node_1 and not data_2:
                node_dict[wire[1]] = node_dict[wire[0]]
            elif node_2 and not data_1:
                node_dict[wire[0]] = node_dict[wire[1]]
            elif not data_1 and data_2:
                node_dict[wire[0]] = wire[1]
            elif data_1 and not data_2 or not data_1 and not data_2:
                node_dict[wire[1]] = wire[0]
            else:
                # send error message
                print('error')
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

    # update into & out of nodes for calculation
    def update_nodes(self):
        for node in self.circ.nodes_iter(data=True):
            edges = self.circ.edges(keys=True)
            # what about using self.circ.edge[node] but what about the second node?
            for edge in edges:
                if edge[0] == node[0]:
                    node[1]['out'].append(edge)
                elif edge[1] == node[0]:
                    node[1]['into'].append(edge)

    def update_odes(self):
        for edge in self.circ.edges_iter(keys=True, data=True):
            if edge[3]['type'] == 'c' or edge[3]['type'] == 'i':
                self.ODEs.append(edge)

    def dydt(self, time, solution_vars, special_nodes):
        calculated = []
        for edge in self.ODEs:
            calculated.append(edge[3]['flow'].calc(time, solution_vars, special_nodes))
        return calculated

    def rk4(self, dydt, t_0, h, n):
        special_nodes = []
        values = []
        # load special nodes
        for edge in self.circ.edges_iter(data=True):
            if edge['type'] == 'c':
                if not self.circ.node[edge[0]].out_degree() == 0:
                    special_nodes.append(self.circ.node[edge[0]])
                elif not self.circ.node[edge[1]].out_degree() == 0:
                    special_nodes.append(self.circ.node[edge[0]])
                else:
                    # error message here
                    print('error')
        # construct list for first round
        initial_values = []
        for node in special_nodes:
            initial_values.append(node['p'][0])
        values.append(initial_values)
        half_h = h*0.5
        t = [t_0]
        for i in range(0, n):
            t.append(t_0 + (i+1)*h)
            k1 = [h*dy for dy in dydt(t[i+1], values[i])]
            wtemp = [ww + 0.5*kk1 for ww, kk1 in zip(values[i], k1)]
            k2 = [h*dy for dy in dydt(t[i+1]+half_h, wtemp, special_nodes)]
            wtemp = [ww + 0.5*kk2 for ww, kk2 in zip(values[i], k2)]
            k3 = [h*dy for dy in dydt(t[i+1]+half_h, wtemp, special_nodes)]
            wtemp = [ww + kk3 for ww, kk3 in zip(values[i], k3)]
            k4 = [h*dy for dy in dydt(t[i+1]+h, wtemp, special_nodes)]
            values.append([ww + (1/6.)*(kk1+2.*kk2+2.*kk3+kk4) for ww, kk1, kk2, kk3, kk4 in zip(initial_values, k1, k2, k3, k4)])

        return values, t





r = circiut()
r.open_saved('test.xml')
r.prune()
r.update_nodes()
r.update_odes()
r.rk4(r.dydt, 0, 0.01, 2000)
print('hi')