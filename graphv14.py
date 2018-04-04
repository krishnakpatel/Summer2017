import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from xml.etree.ElementTree import ElementTree


class Circiut(object):

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

    def load_node(self, comp): # resolve difference for linear interpolation & kinda re-implementing find name
        file_name = comp.attrib['name']
        node = comp.find('Node').text
        # nodes = comp.findall('Node')
        # node = None
        # for n in nodes:
        #    if n.text in self.node_dict:
        #        node = n.text
        # if node == None: # should never happen at this point, right? because wires are loaded in first now
        #    node = nodes[0].text
        if file_name == '':
            value = float(comp.attrib['value'])
            self.circ.node[self.find_node(node)]['p'] = value
        else:
            time = []
            pressure = []
            pressures = {} # keep here for efficiency x and y just for linear interpolation
            file = open(file_name, "r")
            for line in file:
                entry = line.split()
                time.append(float(entry[0]))
                pressure.append(float(entry[1]))
                pressures[entry[0]] = np.float64(entry[1])
            self.circ.node[self.find_node(node)]['p'] = pressures
            self.circ.node[self.find_node(node)]['x'] = time
            self.circ.node[self.find_node(node)]['y'] = pressure

    def component_extract(self, comp):
        value = comp.attrib['value'] * self.metric_dict[comp.attrib['metricPrefix']]
        nodes = comp.findall('Node')
        if not len(nodes) == 2:
            # send out error message
            print('error component')
        n = []
        for x in nodes:
            n.append(self.find_node(x.text))
        return n, value

    def calc_c_flow(self, node, type, time, solution_vars, special_nodes, edge):
        flow = 0
        for e in self.circ.node[node][type]:
            if not e == edge: # check to make sure it is not a capacitor????
                flow += self.circ.edge[e[0]][e[1]][e[2]]['calculate'](e, time, solution_vars, special_nodes)
        return flow

    def calc_c(self, edge, time, solution_vars, special_nodes):
        # determine which node is the important one & that is the one you look at into and out for
        node = 0
        if edge[0] in special_nodes:
            node = edge[0]
        elif edge[1] in special_nodes:
            node = edge[1]
        else:
            # isn't this an error on your part not a user error if you get here
            print('error capacitor')
        q_in = self.calc_c_flow(node, 'into', time, solution_vars, special_nodes, edge)
        q_out = self.calc_c_flow(node, 'out', time, solution_vars, special_nodes, edge)
        return (q_in - q_out)/float(edge[3]['value'])


    def get_flows(self, type, node):
        for edge in node[1][x]:


    def calc_pressure(self, node):
        # ignore capacitors
        # get flow list from resistor
        # inductor from it's flow value
        floats_into = 0
        floats_out = 0
        flows_into = []
        flows_out = []
        for x in ['into', 'out']:
            for edge in node[1][x]:
                if not:
                    if i:
                        get ['flow']
                else:
                    if you are going through the 'into' list:
                        list = [None, 'p'[time], 'value']
                    else:
                        list = ['p'[time], None, 'value']
        somehow figure out which one is None


    def is_unknown(self, node_dict, time, solution_vars, special_nodes):
        if node_dict['p'] != None:
            return sel

    def calc_i(self, edge, time, solution_vars, special_nodes): #IDENTIFY WHICH NODE IS THE UNKNOWN NODE
    # make sure to extract the full edges in the functions from the into and out lists like in calc_c_flow

        p_in =
        p_out =
        return (p_in-p_out)/float(edge[3]['value'])

    def calc_r(self, edge, time, solution_vars, special_nodes):
        p_in = 0
        p_out = 0
        if edge[0] in special_nodes:
            p_in = solution_vars[special_nodes.index(edge[0])]
        elif type(self.circ.node[edge[0]]['p']) == dict and len(self.circ.node[edge[0]]['p']) > 1:
            if time in self.circ.node[edge[0]]['p']:
                p_in = self.circ.node[edge[0]]['p'][time]
            else:
                p_in = self.boundary_conditions(time, self.circ.node[edge[0]])
        elif isinstance(self.circ.node[edge[0]]['p'], float): # this doesn't need to be altered, right?
            p_in = self.circ.node[edge[0]]['p']
        else:
            print('error resistor')
            # send error message
        if edge[1] in special_nodes:
            p_out = solution_vars[special_nodes.index(edge[1])]
        elif type(self.circ.node[edge[1]]['p']) == dict and len(self.circ.node[edge[1]]['p']) > 1:
            if time in self.circ.node[edge[1]]['p']:
                p_out = self.circ.node[edge[1]]['p'][time]
            else:
                p_in = self.boundary_conditions(time, self.circ.node[edge[1]])
        elif isinstance(self.circ.node[edge[1]]['p'], float):
            p_out = self.circ.node[edge[1]]['p']
        else:
            print('error resistor 2')
            # send error message
        return (p_in - p_out)/(float(edge[3]['value']))

    def load_wires(self, pos_list):
        start = self.find_node(str('(' + pos_list[0] + ', ' + pos_list[1] + ')'))
        end = self.find_node(str('(' + pos_list[2] + ', ' + pos_list[3] + ')'))
        self.circ.add_edge(start, end, type='w')

    def load_inductor(self, component):
        extracted, value = self.component_extract(component)
        self.circ.add_edge(extracted[0], extracted[1], type='i', value=value, calculate=None, flow=None) # load in initial flow values later

    def load_capacitor(self, component):
        extracted, value = self.component_extract(component)
        self.circ.add_edge(extracted[0], extracted[1], type='c', value=value, calculate=self.calc_c)

    def load_resistor(self, component):
        extracted, value = self.component_extract(component)
        self.circ.add_edge(extracted[0], extracted[1], type='r', value=value, calculate=self.calc_r)

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
                print('error open save')
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
            if self.circ.node[wire[0]]['p'] is not None and self.circ.node[wire[1]]['p'] is not None:
                # user error
                print('error prune')
            elif self.circ.node[wire[0]]['p'] is not None:
                if node_1 and node_2:
                    use = node_dict[wire[0]]
                    replace = node_dict[wire[1]]
                    node_dict[replace] = use
                    for key in node_dict:
                        if node_dict[key] == replace:
                            node_dict[key] = use
                elif node_2:
                    use = wire[0]
                    replace = node_dict[wire[1]]
                    node_dict[replace] = use
                    for key in node_dict:
                        if node_dict[key] == replace:
                            node_dict[key] = use
                else:
                    node_dict[wire[1]] = wire[0]
            elif self.circ.node[wire[1]]['p'] is not None:
                if node_1 and node_2:
                    use = node_dict[wire[1]]
                    replace = node_dict[wire[0]]
                    node_dict[replace] = use
                    for key in node_dict:
                        if node_dict[key] == replace:
                            node_dict[key] = use
                elif node_1:
                    use = wire[1]
                    replace = node_dict[wire[0]]
                    node_dict[replace] = use
                    for key in node_dict:
                        if node_dict[key] == replace:
                            node_dict[key] = use
                else:
                    node_dict[wire[0]] = wire[1]
            else:
                if node_1 and node_2:
                    use = node_dict[wire[0]]
                    replace = node_dict[wire[1]]
                    node_dict[replace] = use
                    for key in node_dict:
                        if node_dict[key] == replace:
                            node_dict[key] = use
                else:
                    node_dict[wire[1]] = wire[0]

        # replace nodes
        for edge in self.circ.edges(keys=True, data=True):
            if edge[0] in node_dict and edge[1] in node_dict:
                self.circ.remove_edge(edge[0], edge[1], edge[2])
                self.circ.add_edge(node_dict[edge[0]], node_dict[edge[1]], type=edge[3]['type'], value=edge[3]['value'], calculate=edge[3]['calculate'])
            elif edge[0] in node_dict: # should you modify key
                self.circ.remove_edge(edge[0], edge[1], edge[2])
                self.circ.add_edge(node_dict[edge[0]], edge[1], key=edge[2], type=edge[3]['type'], value=edge[3]['value'], calculate=edge[3]['calculate'])
            elif edge[1] in node_dict: # should you modify key
                self.circ.remove_edge(edge[0], edge[1], edge[2])
                self.circ.add_edge(edge[0], node_dict[edge[1]], key=edge[2], type=edge[3]['type'], value=edge[3]['value'], calculate=edge[3]['calculate'])
        # delete nodes
        out_degrees = self.circ.out_degree()
        in_degrees = self.circ.in_degree()
        to_delete = [n for n in out_degrees if out_degrees[n] == 0 and in_degrees[n] == 0]
        self.circ.remove_nodes_from(to_delete)

    # update into & out of nodes for calculation
    def update_nodes(self):
        for node in self.circ.nodes_iter(data=True):
            edges = self.circ.edges(keys=True, data=True)
            # what about using self.circ.edge[node] but what about the second node?
            for edge in edges:
                # if not edge[3]['type'] == 'c': <--- why did I do that?
                if edge[0] == node[0]:
                    node[1]['out'].append(edge)
                elif edge[1] == node[0]:
                    node[1]['into'].append(edge)

    # linear interpolation
    def boundary_conditions(self, time, node):
        this = np.interp(float(time), node['x'], node['y'], period=1)
        print(this)
        return this

    def go_into(self, node):
        if len(node[1]['out']) == 0:
            return
        this = node[1]['out']
        for edge in this:
            self.go_into((edge[1], self.circ.node[edge[1]]))
            if edge[3]['type'] == 'c' or edge[3]['type'] == 'i':
                self.ODEs.append(edge)

    def update_odes(self):
        # find start node
        n = 0
        for node in self.circ.nodes_iter(data=True):
            if len(node[1]['into']) == 0 and len(node[1]['out']) > 0: # and ensure the edge out of it isn't a capacitor??
                n = node
                break
        # recurse through graph from there
        self.go_into(n)

    def dydt(self, time, solution_vars, special_nodes):
        calculated = []
        for edge in self.ODEs:
            calculated.append(edge[3]['calculate'](edge, time, solution_vars, special_nodes))
        return calculated

    def rk4(self, dydt, t_0, h, n):
        special_components = []
        initial_values = []
        for obj in self.ODEs:
            if obj[3]['type'] == 'c':
                special_components.append(obj[0])
                if type(self.circ.node[obj[0]]['p']) == dict:
                    initial_values.append(self.circ.node[obj[0]]['p']['0']) # 0 or 0.0
                else:
                    initial_values.append(self.circ.node[obj[0]]['p'])
            else:
                special_components.append(obj) # append the inductor
                initial_values.append(obj[3][['flow']])
                # find initial flow to the inductor

        values = np.array([initial_values,])
        half_h = np.float64(h*0.5)
        t = np.array([t_0])

        # np arrays don't seem to make a difference, but are there to protect the math so if at any point python wouldn't be able
        # to handle it, it remains protected
        for i in range(-1, n):
            if np.mod(i, 100) == 0:
                print('CurrentTime %d/%d ' % (i, n))

            t = np.append(t, [(t[0] + ((i+1)*h))])

            #t.append(t_0 + (i+1)*h)
            k1 = np.array([h*dy for dy in dydt(str(t[i+1]), values[i], special_components)])
            wtemp = np.array([ww + 0.5*kk1 for ww, kk1 in zip(values[i], k1)])

            # print(t[i+1]+half_h)

            time = np.float64(t[i+1] + half_h)
            k2 = np.array([h*dy for dy in dydt(str(time), wtemp, special_components)])
            wtemp = np.array([ww + 0.5*kk2 for ww, kk2 in zip(values[i], k2)])
            k3 = np.array([h*dy for dy in dydt(str(t[i+1]+half_h), wtemp, special_components)])
            wtemp = np.array([ww + kk3 for ww, kk3 in zip(values[i], k3)])

            # print(t[i+1]+h)

            k4 = np.array([h*dy for dy in dydt(str(t[i+1]+h), wtemp, special_components)])
            values = np.append(values, [[ww + (1/6.)*(kk1+2.*kk2+2.*kk3+kk4) for ww, kk1, kk2, kk3, kk4 in zip(values[i], k1, k2, k3, k4)]], axis=0)

        return values, t


r = Circiut()
r.open_saved('test.xml')
r.prune()
r.update_nodes()
r.update_odes()
solution, time = r.rk4(r.dydt, 0, np.float64(0.001), 3000)

# Plot the solution for verification
test_sol = np.zeros(3001 + 1)
test_sol[0] = 90
for i in range(0, 3001):
    test_sol[i + 1] = solution[i + 1][0]

plt.plot(time, test_sol)
plt.title('Pressure vs. time in RCR circuit')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (mmHg)')
plt.show()

