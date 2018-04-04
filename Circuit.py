import numpy as np
from xml.etree.ElementTree import ElementTree
import linecache as lnc


class Circuit:

	def __init__(self, input_file, saved_file):
		self.circ = dict()
		self.edges = list()
		self.num_nodes = 0
		self.metric_dict = {'E': 10**18, 'P': 10**15, 'T': 10**12, 'G': 10**9, 'M': 10**6, 'k': 1000, 'h': 100,
							'da': 10, 'd': 0.1, 'c': 0.01, 'm': 0.001, 'µ': 10**(-6), 'n': 10**(-9), 'p': 10**(-12),
							'f': 10**(-18), '': 1}

		# variables to help with initialization
		self.node_dict = dict()

		# initialize circuit
		# load in starting pressure/flow values
		self.load_starting(input_file)
		# load in saved circuit
		self.open_xml(saved_file)
		# prune circuit
		self.prune()
		# fill in nodes
		self.update_nodes()

	# edges function as components (resistors, capacitors, inductors)
	class Edge:
		def __init__(self, value, type, start, end, flows, x, y):
			self.value = value
			self.type = type
			self.start = start
			self.end = end
			self.q = flows
			self.x = x
			self.y = y

	def get_nodes(self):
		return self.circ.items()

	# returns data associated with node number node_num, 
	# returns None if node doesn't exist
	def get_node(self, node_num):
		return self.circ.get(node_num, None)

	def add_node(self, num, into=[], out=[], p=None):
		if num in self.circ:
			print('error: attempting to add a node that already exists')
			return
		self.circ[num] = dict()
		self.circ[num]['into'] = into
		self.circ[num]['out'] = out
		self.circ[num]['p'] = p

	def add_edge(self, start, end, type, value, flows=None, x=None, y=None):
		self.edges.append(self.Edge(value, type, start, end, flows, x, y))

	def remove_edge(self, edge):
		self.edges.remove(edge)

	def remove_node(self, node):
		del self.circ[node]

	# returns node associated with that position, or creates a 
	# node if that position doesn't yet exist and returns it
	def find_node(self, node):
		if node not in self.node_dict:
			self.num_nodes += 1
			self.node_dict[node] = self.num_nodes
			self.add_node(self.num_nodes)
		return self.node_dict[node]

	def load_node(self, file_name, nodes): # resolve difference for linear interpolation & kinda re-implementing find name
		if not len(nodes) >= 2:
			# send out error message
			print('too many nodes/values specified in load_node')

		node = None
		if nodes[0] in self.node_dict.values():
			node = nodes[0]
		elif nodes[1] in self.node_dict.values():
			node = nodes[1]
		# if we haven't seen the node before, we add it to the graph
		elif nodes[0] not in self.node_dict and nodes[1] not in self.node_dict:
			self.num_nodes += 1
			self.node_dict[nodes[0]] = self.num_nodes
			self.node_dict[nodes[1]] = self.num_nodes
			node = self.num_nodes
			self.add_node(self.num_nodes)

		elif nodes[0] in self.node_dict:
			node = self.node_dict[nodes[0]]
		else:
			node = self.node_dict[nodes[1]]

		# adding the starting values from the file
		if not file_name:
			# the last value in nodes is the intial value
			self.circ[node]['p'] = float(nodes[-1])
		else:
			time = []
			pressure = []
			pressures = {} 
			file = open(file_name, "r")
			# load the values specified in the pressure file
			for line in file:
				entry = line.split()
				time.append(float(entry[0]))
				pressure.append(float(entry[1]))
				pressures[np.float64(entry[0])] = np.float64(entry[1])
			# save for efficient access
			self.circ[node]['p'] = pressures
			# save these values to use np.interp 
			self.circ[node]['x'] = time
			self.circ[node]['y'] = pressure

	def is_number(self, s):
		try:
			float(s)
			return True
		except ValueError:
			return False

	def load_starting(self, file_name):
		num_pressures = int(lnc.getline(file_name, 1))
		num_flows = int(lnc.getline(file_name, num_pressures+2))
		for i in range(2, num_pressures+2):
			line = lnc.getline(file_name, i).rstrip().split(';')
			self.load_node(None, line)

		for x in range(i+2, (i+num_flows+2)):
			line = lnc.getline(file_name, x).rstrip().split(';')
			start = self.find_node(line[0])
			end = self.find_node(line[1])
			if self.is_number(line[2]):
				self.add_edge(start, end, 'flow', line[2])
			else:
				time = []
				flow = []
				flows = {}  # keep here for efficiency x and y just for linear interpolation
				file = open(line[2], "r")
				for line in file:
					entry = line.split()
					time.append(float(entry[0]))
					flow.append(-float(entry[1]))
					flows[float(entry[0])] = -np.float64(entry[1])
				self.add_edge(start, end, 'flow', None, flows, time, flow)

	def component_extract(self, comp):
		value = comp.attrib['value'] * self.metric_dict[comp.attrib['metricPrefix']]
		nodes = comp.findall('Node')
		if not len(nodes) == 2:
			# send out error message
			print('error component')
		n = [self.find_node(x.text) for x in nodes]
		return n, value

	def load_inductor(self, component):
		extracted, value = self.component_extract(component)
		self.add_edge(extracted[0], extracted[1], 'i', value) # load in initial flow values later

	def load_capacitor(self, component):
		extracted, value = self.component_extract(component)
		self.add_edge(extracted[0], extracted[1], 'c', value)

	def load_resistor(self, component):
		extracted, value = self.component_extract(component)
		self.add_edge(extracted[0], extracted[1], 'r', value)

	def load_wires(self, pos_list):
		start = self.find_node(str('(' + pos_list[0] + ', ' + pos_list[1] + ')'))
		end = self.find_node(str('(' + pos_list[2] + ', ' + pos_list[3] + ')'))
		self.add_edge(start, end, 'w', None)

	def load_boundaryface(self, component):
		file_name = component.attrib['name']
		nodes = component.findall('Node')
		text = [self.find_node(x.text) for x in nodes]
		if file_name != '':
			self.load_node(file_name, text)

	def open_xml(self, file_name):
		file = open(file_name, 'r')
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
				self.load_boundaryface(c)
			else:
				# send error message
				print('error open xml file')
		# read in wires
		for wire in root.findall('wire'):
			pos = wire.find('wirePos').text.split(', ')
			self.load_wires(pos)

	def replace_values(self, use, replace, node_dict):
		node_dict[replace] = use
		for key in node_dict.keys():
			if node_dict[key] == replace:
				node_dict[key] = use
		return node_dict

	# final step
	# deleting the wires, pruning the graph
	def prune(self):
		wires = [edge for edge in self.edges if edge.type == 'w']
		# delete the wires from edges
		self.edges = [edge for edge in self.edges if edge.type != 'w']
		node_dict = {}
		# build a dictionary to consolidate the nodes
		for wire in wires:
			node_1 = wire.start in node_dict
			node_2 = wire.end in node_dict
			if node_1 and node_2:
				if self.circ[wire.start]['p'] is not None:
					use = node_dict[wire.start]
					replace = node_dict[wire.end]
				else:
					use = node_dict[wire.end]
					replace = node_dict[wire.start]
				node_dict = self.replace_values(use, replace, node_dict)
			elif node_1:
				if self.circ[wire.end]['p'] is not None:
					use = wire.end
					replace = node_dict[wire.start]
					node_dict = self.replace_values(use, replace, node_dict)

				else:
					node_dict[wire.end] = node_dict[wire.start]
			elif node_2:
				if self.circ[wire.start]['p'] is not None:
					use = node_dict[wire.start]
					replace = node_dict[wire.end]
					node_dict = self.replace_values(use, replace, node_dict)
				else:
					node_dict[wire.start] = node_dict[wire.end]
			else:
				if self.circ[wire.start]['p'] is not None:
					node_dict[wire.end] = wire.start
				else:
					node_dict[wire.start] = wire.end
		# recreate the graph in the wireless state
		new_edges = []
		new_nodes = set()
		for edge in self.edges:
			edge.start = node_dict.get(edge.start, edge.start)
			edge.end = node_dict.get(edge.end, edge.end)
			new_nodes.add(edge.start)
			new_nodes.add(edge.end)
			new_edges.append(edge)
		self.edges = new_edges
		# delete the unused nodes
		self.num_nodes = len(new_nodes)
		self.circ = {k:v for k,v in self.circ.items() if k in new_nodes}


	def update_nodes(self):
		for node in self.circ.keys():
			self.circ[node]['out'] = [edge for edge in self.edges if edge.start == node]
			self.circ[node]['into'] = [edge for edge in self.edges if edge.end == node]








