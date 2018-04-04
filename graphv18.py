import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from xml.etree.ElementTree import ElementTree
import linecache as lnc


# consider creating a cache so that previously calculated flows/pressures (on resistors/capacitors/inductors) are not calculated twice at the same point in time
# make sure all of your math is protected float math!

class Circiut(object):

	def __init__(self):
		# consider also making special_nodes and solution_vars global and time
		self.circ = nx.MultiDiGraph()
		self.metric_dict = {'E': 10**18, 'P': 10**15, 'T': 10**12, 'G': 10**9, 'M': 10**6, 'k': 1000, 'h': 100,
							'da': 10, 'd': 0.1, 'c': 0.01, 'm': 0.001, 'µ': 10**(-6), 'n': 10**(-9), 'p': 10**(-12),
							'f': 10**(-18), '': 1}
		self.node_dict = {}
		self.num_nodes = 0
		self.ODEs = []
		self.cycle = 0

	def find_node(self, node):
		if node not in self.node_dict:
			self.num_nodes += 1
			self.node_dict[node] = self.num_nodes
			self.circ.add_node(self.num_nodes, into=[], out=[], p=None)
		return self.node_dict[node]

	def load_node(self, file_name, nodes): # resolve difference for linear interpolation & kinda re-implementing find name
		if not len(nodes) >= 2:
			# send out error message
			print('error node')
		node = -1
		if nodes[0] not in self.node_dict and nodes[1] not in self.node_dict:
			self.num_nodes += 1
			self.node_dict[nodes[0]] = self.num_nodes
			self.node_dict[nodes[1]] = self.num_nodes
			node = self.num_nodes
			self.circ.add_node(self.num_nodes, into=[], out=[], p=None)
		elif nodes[0] in self.node_dict:
			node = self.node_dict[nodes[0]]
		else:
			node = self.node_dict[nodes[1]]
		# can't specify constant values without file anymore
		if file_name == ' ':
		   self.circ.node[node]['p'] = float(nodes[2])
		elif file_name != '':
			time = []
			pressure = []
			pressures = {} # keep here for efficiency x and y just for linear interpolation
			file = open(file_name, "r")
			for line in file:
				entry = line.split()
				time.append(float(entry[0]))
				pressure.append(float(entry[1]))
				pressures[entry[0]] = np.float64(entry[1])
				# probably don't need to save x and y either
			self.circ.node[node]['p'] = pressures
			self.circ.node[node]['x'] = time
			self.circ.node[node]['y'] = pressure

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

	def calc_q(self, edge, time, solution_vars, special_nodes):
		if type(edge[3]['q']) == dict and len(edge[3]['q']) > 1:
			if time in edge[3]['q']:
				return edge[3]['q'][time]
			else:
				return np.interp([float(time)], edge[3]['x'], edge[3]['y'], period=1)[0]
		elif isinstance(edge[3]['q'], float): # this doesn't need to be altered, right?
			return edge[3]['q']
		else:
			print('error calc_q')

	def calc_c_flow(self, node, type, time, solution_vars, special_nodes, edge):
		flow = 0
		for e in self.circ.node[node][type]:
			if e[3]['type'] == 'i':
				flow += self.calc_c_i(e, time, solution_vars, special_nodes)
			elif e[3]['type'] == 'r':
				f = self.calc_r(e, time, solution_vars, special_nodes)
				if isinstance(f, list):
					if f[0] is None:
						self.calc_pressure(e[0], time, special_nodes, solution_vars)
					else: #f[1] is None
						self.calc_pressure(e[1], time, special_nodes, solution_vars)
					f = self.calc_r(e, time, solution_vars, special_nodes)
				flow += f
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

	def both_sides(self, into, out, floats_into, floats_out):
		# double check this math!
		term_1 = float(out[2])*into[2]*(floats_into-floats_out)
		term_2 = float(into[2])*out[1] + float(into[0])*out[2]
		term_3 = float(out[2])*into[1] + float(out[0])*into[2]
		return (term_1 + term_2)/float(term_3)

	def gather_abc(self, flows, unknown):
		a = 0
		b = 0
		c = 1
		if unknown =='a':
			for flow in flows:
				c *= flow[2]
				a += flow[2]
				b_temp = flow[1]
				for f in flows:
					if f is not flow:
						b_temp *= f[2]
				b -= b_temp
		else:
			for flow in flows:
				c *= flow[2]
				b -= flow[2]
				a_temp = flow[1]
				for f in flows:
					if f is not flow:
						a_temp *= f[2]
				a += a_temp
		return [a, b, c]

	def get_flows(self, type, node, time, special_nodes, solution_vars):
		floats = 0.
		flows = []
		for edge in self.circ.node[node][type]:
			if edge[3]['type'] == 'i': # must be in solution_vars
				floats += solution_vars[special_nodes.index(edge)]
			elif edge[3]['type'] == 'r':
				# this should never be a float right?
				flows.append(edge[3]['calculate'](edge, time, solution_vars, special_nodes))
		return floats, flows

	def calc_pressure(self, node, time, special_nodes, solution_vars):
		# ignore capacitors
		# get flow list from resistor
		# inductor from it's flow value
		floats_into, flows_into = self.get_flows('into', node, time, special_nodes, solution_vars)
		floats_out, flows_out = self.get_flows('out', node, time, special_nodes, solution_vars)
		abc_in = self.gather_abc(flows_into, 'b')
		abc_out = self.gather_abc(flows_out, 'a')
		# double check how to check if its empty
		if bool(flows_into) and bool(flows_out):
			pressure = self.both_sides(abc_in, abc_out, floats_into, floats_out)
		elif bool(flows_into):
			term_1 = float(floats_out)-floats_into
			term_2 = term_1*float(abc_in[2])
			term_3 = term_2 + float(abc_in[0])
			pressure = term_3/((-1.0)*abc_in[1])
		elif bool(flows_out):
			term_1 = float(floats_into)-floats_out
			term_2 = term_1*float(abc_out[2])
			term_3 = term_2 - float(abc_out[1])
			pressure = term_3/float(abc_out[0])
		self.circ.node[node]['calculated_p'] = (self.cycle, pressure)
		return pressure

	def calc_c_i(self, edge, time, solution_vars, special_nodes):
		index = special_nodes.index(edge)
		return solution_vars[index]


	def calc_i(self, edge, time, solution_vars, special_nodes): #IDENTIFY WHICH NODE IS THE UNKNOWN NODE
	# make sure to extract the full edges in the functions from the into and out lists like in calc_c_flow
		p_in = self.boundary_conditions(time, edge[0], special_nodes, solution_vars)
		p_out = self.boundary_conditions(time, edge[1], special_nodes, solution_vars)
		# what if both are None? can both be None? change to 2 if statements? <- I don't think the program is capable of dealing with that
		if p_in is None:
			p_in = self.calc_pressure(edge[0], time, special_nodes, solution_vars)
		elif p_out is None:
			p_out = self.calc_pressure(edge[1], time, special_nodes, solution_vars)
		return (p_in-p_out)/float(edge[3]['value'])


	def calc_r(self, edge, time, solution_vars, special_nodes):
		p_in = self.boundary_conditions(time, edge[0], special_nodes, solution_vars)
		p_out = self.boundary_conditions(time, edge[1], special_nodes, solution_vars)
		# can they both be None? NO THE PROGRAM WILL BREAK/ISN'T CAPABLE
		if p_in is None or p_out is None: # make sure that this can never be returned to a capacitor
			return [p_in, p_out, float(edge[3]['value'])]
		return (p_in - p_out)/(float(edge[3]['value']))

	def load_wires(self, pos_list):
		start = self.find_node(str('(' + pos_list[0] + ', ' + pos_list[1] + ')'))
		end = self.find_node(str('(' + pos_list[2] + ', ' + pos_list[3] + ')'))
		self.circ.add_edge(start, end, type='w')

	def load_inductor(self, component):
		extracted, value = self.component_extract(component)
		self.circ.add_edge(extracted[0], extracted[1], type='i', value=value, calculate=self.calc_i) # load in initial flow values later

	def load_capacitor(self, component):
		extracted, value = self.component_extract(component)
		self.circ.add_edge(extracted[0], extracted[1], type='c', value=value, calculate=self.calc_c)

	def load_resistor(self, component):
		extracted, value = self.component_extract(component)
		self.circ.add_edge(extracted[0], extracted[1], type='r', value=value, calculate=self.calc_r)

	# consolidate some code with load_node potentially
	def load_file(self, file_name):
		num_pressures = int(lnc.getline(file_name, 1))
		num_flows = int(lnc.getline(file_name, num_pressures+2))
		for i in range(2, num_pressures+2):
			line = lnc.getline(file_name, i).rstrip().split(';')
			self.load_node(line[0],line[1:])
		for x in range(i+2, (i+num_flows+2)):
			line = lnc.getline(file_name, x).rstrip().split(';')
			start = self.find_node(line[1])
			end = self.find_node(line[2])
			if file_name == '':
				self.circ.add_edge(start, end, type='flow', q=line[3], calculate=self.calc_q)
			else:
				time = []
				flow = []
				flows = {}  # keep here for efficiency x and y just for linear interpolation
				file = open(line[0], "r")
				for line in file:
					entry = line.split()
					time.append(float(entry[0]))
					flow.append(-float(entry[1]))
					flows[entry[0]] = -np.float64(entry[1])
				self.circ.add_edge(start, end, type='flow', q=flows, x=time, y=flow, calculate=self.calc_q)

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
				# can turn this into it's own function
				file_name = c.attrib['name']
				nodes = c.findall('Node')
				text = []
				for n in nodes:
					text.append(n.text)
				self.load_node(file_name, text)
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
				#self.circ.remove_edge(edge[0], edge[1], edge[2])
		# create the node_dict
		for wire in wires:
			node_1 = wire[0] in node_dict
			node_2 = wire[1] in node_dict
			if self.circ.node[wire[0]]['p'] is not None and self.circ.node[wire[1]]['p'] is not None: # no longer an error # fix this statement
				print('no longer an error fix this')
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
			# inductor has flow and flow edges don't have values and they have really wierd other things
			if edge[3]['type'] == 'flow':
				if edge[0] in node_dict and edge[1] in node_dict:
					self.circ.remove_edge(edge[0], edge[1], edge[2])
					self.circ.add_edge(node_dict[edge[0]], node_dict[edge[1]], type=edge[3]['type'],
									   q=edge[3]['q'], x=edge[3]['x'], y=edge[3]['y'], calculate=edge[3]['calculate'])
				elif edge[0] in node_dict:
					self.circ.remove_edge(edge[0], edge[1], edge[2])
					self.circ.add_edge(node_dict[edge[0]], edge[1], type=edge[3]['type'], q=edge[3]['q'], x=edge[3]['x'], y=edge[3]['y'],
									   calculate=edge[3]['calculate'])
				elif edge[1] in node_dict:
					self.circ.remove_edge(edge[0], edge[1], edge[2])
					self.circ.add_edge(edge[0], node_dict[edge[1]], type=edge[3]['type'], q=edge[3]['q'], x=edge[3]['x'], y=edge[3]['y'],
									   calculate=edge[3]['calculate'])
			else:
				if edge[0] in node_dict and edge[1] in node_dict:
					self.circ.remove_edge(edge[0], edge[1], edge[2])
					self.circ.add_edge(node_dict[edge[0]], node_dict[edge[1]], type=edge[3]['type'], value=edge[3]['value'], calculate=edge[3]['calculate'])
				elif edge[0] in node_dict:
					self.circ.remove_edge(edge[0], edge[1], edge[2])
					self.circ.add_edge(node_dict[edge[0]], edge[1], type=edge[3]['type'], value=edge[3]['value'], calculate=edge[3]['calculate'])
				elif edge[1] in node_dict:
					self.circ.remove_edge(edge[0], edge[1], edge[2])
					self.circ.add_edge(edge[0], node_dict[edge[1]], type=edge[3]['type'], value=edge[3]['value'], calculate=edge[3]['calculate'])
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
				# if not edge[3]['type'] == 'c': <--- why did I do that? to stop an infinite loop
				if edge[0] == node[0]:
					node[1]['out'].append(edge)
				elif edge[1] == node[0]:
					node[1]['into'].append(edge)

	# linear interpolation
	def boundary_conditions(self, time, node, special_nodes, solution_vars):
		if node in special_nodes:
			return solution_vars[special_nodes.index(node)]
		elif type(self.circ.node[node]['p']) == dict and len(self.circ.node[node]['p']) > 1:
			if time in self.circ.node[node]['p']:
				return self.circ.node[node]['p'][time]
			else:
				return np.interp([float(time)], self.circ.node[node]['x'], self.circ.node[node]['y'], period=1)[0]
		elif isinstance(self.circ.node[node]['p'], float): # this doesn't need to be altered, right?
			return self.circ.node[node]['p']
		elif 'calculated_p' in self.circ.node[node] and self.circ.node[node]['calculated_p'][0] == self.cycle:
			return self.circ.node[node]['calculated_p'][1]
		else:
			# to know pressure must be calculated
			return None

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
				initial_values.append(0) # -> how to load in initial values for flow!?
				# find initial flow to the inductor

		values = [initial_values]
		half_h = np.float64(h*0.5)
		t = [0]

		# np arrays don't seem to make a difference, but are there to protect the math so if at any point python wouldn't be able
		# to handle it, it remains protected
		for i in range(-1, n+1):
			if np.mod(i, 100) == 0:
				print('CurrentTime %d/%d ' % (i, n))

			t.append(t[0] + ((i+1)*h))
			# YOU CANT ZIP WITH TUPLES
			#t.append(t_0 + (i+1)*h)
			k1 = [h*dy for dy in dydt(str(t[i+1]), values[i], special_components)]
			wtemp = [ww + 0.5*kk1 for ww, kk1 in zip(values[i], k1)]

			time = np.float64(t[i+1] + half_h)
			self.cycle += half_h
			print(self.cycle)
			# SO FAR THIS IS WHERE IT GOES WRONG & FIGURE OUT HOW TO SAVE FLOW SEPARATELY
			# restrict acccess to it's valueu to solution vars ugh you're insane that's how you do it!
			# it only took you until 4 am!
			k2 = [h*dy for dy in dydt(str(time), wtemp, special_components)]
			wtemp = [ww + 0.5*kk2 for ww, kk2 in zip(values[i], k2)]
			k3 = [h*dy for dy in dydt(str(t[i+1]+half_h), wtemp, special_components)]
			wtemp = [ww + kk3 for ww, kk3 in zip(values[i], k3)]

			self.cycle += h
			print(self.cycle)
			k4 = [h*dy for dy in dydt(str(t[i+1]+h), wtemp, special_components)]
			values.append([ww + (1/6.)*(kk1+2.*kk2+2.*kk3+kk4) for ww, kk1, kk2, kk3, kk4 in zip(values[i], k1, k2, k3, k4)])


		return values, t



r = Circiut()
r.load_file('large_file.txt') # modify to use sys
r.open_saved('large_ode.xml')
r.prune()
r.update_nodes()
r.update_odes()
solution, time = r.rk4(r.dydt, 0, np.float64(0.001), 5000)

# Plot the solution for verification

test_sol = np.zeros(5002 + 1)
test_sol_2 = np.zeros(5002 + 1)
test_sol[0] = 90.0
test_sol_2[0] = 0.0

for i in range(0, 5001):
	test_sol[i + 1] = solution[i + 1][0]
	test_sol_2[i + 1] = solution[i + 1][1]

plt.plot(time, test_sol)
plt.show()

plt.plot(time, test_sol_2)
plt.show()

