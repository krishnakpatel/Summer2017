from Circuit import Circuit
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class Calculator:

	def __init__(self, initial_pressures, xml, h, n):

		self.circuit = Circuit(initial_pressures, xml)
		self.ODEs = list()
		self.cycle = 0
		self.special_nodes = list()

		self.calculate(h, n)

	def gather_odes(self):
		# find start node
		start = None
		for num, node in self.circuit.get_nodes():
			is_start = len(node['into']) > 0
			for edge in node['into']:
				if edge.start != edge.end:
					is_start = False
			if len(node['into']) == 0 and len(node['out']) > 0 or is_start:
				start = num
				break

			
		# use bfs to find ordering of ODEs 
		queue = deque()
		visited = set()
		queue.append(start)
		while len(queue) > 0:
			node = queue.popleft()
			visited.add(node)

			# loop through the data for the outgoing edges
			for edge in self.circuit.get_node(node)['out']:
				if edge.end not in visited:
					queue.append(edge.end)

				# add any outgoing ODEs
				if edge.type == 'c' or edge.type == 'i':
					self.ODEs.append(edge)

	def boundary_conditions(self, time, node, solution_vars):
		if node in self.special_nodes:
			return solution_vars[self.special_nodes.index(node)]
		node = self.circuit.get_node(node)
		if type(node['p']) == dict:
			return node['p'].get(time, np.interp([float(time)], node['x'], node['y'], period=1)[0])
		elif node['p'] != None:
			return node['p']
		elif 'calculated' in node and node['cycle'] == self.cycle:
			return node['calculated']
		else:
			return None

# considerations for series and parallel??????

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

	def get_flows(self, type, node, time, solution_vars):
		floats = 0.
		flows = []
		for edge in self.circuit.get_node(node)[type]:
			if edge.type == 'i': # must be in solution_vars
				floats += solution_vars[self.special_nodes.index(edge)]
			elif edge.type == 'r':
				flows.append(self.calc_r(edge, time, solution_vars))
		return floats, flows

	def calc_pressure(self, node, time, solution_vars):
		# ignore capacitors
		# get flow list from resistor
		# inductor from it's flow value
		floats_into, flows_into = self.get_flows('into', node, time, solution_vars)
		floats_out, flows_out = self.get_flows('out', node, time, solution_vars)
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
		self.circuit.circ[node]['calculated'] = pressure
		self.circuit.circ[node]['cycle'] = self.cycle
		return pressure
	
	def calc_c_flow(self, edges, time, solution_vars):
		flow = 0
		for edge in edges:
			if edge.type == 'i':
				flow += solution_vars[self.special_nodes.index(edge)]
			elif edge.type == 'r':
				f = self.calc_r(edge, time, solution_vars)
				if type(f) == list:
					if f[0] is None:
						self.calc_pressure(edge.start, time, solution_vars)
					else: #f[1] is None
						self.calc_pressure(edge.end, time, solution_vars)
					f = self.calc_r(edge, time, solution_vars)
				flow += f
			elif edge.type == 'flow':
				flow += edge.q.get(time, np.interp([float(time)], edge.x, edge.y, period=1)[0])

		return flow

	
	def calc_c(self, edge, time, solution_vars):
		node = None
		if edge.start in self.special_nodes:
			node = self.circuit.get_node(edge.start)
		else:
			node = self.circuit.get_node(edge.end)
		q_in = self.calc_c_flow(node['into'], time, solution_vars)
		q_out = self.calc_c_flow(node['out'], time, solution_vars)
		return (q_in - q_out)/float(edge.value)


	def calc_r(self, edge, time, solution_vars):
		p_in = self.boundary_conditions(time, edge.start, solution_vars)
		p_out = self.boundary_conditions(time, edge.end, solution_vars)
		if p_in is None or p_out is None:
			return [p_in, p_out, float(edge.value)]
		return (p_in - p_out)/(float(edge.value))

	def calc_i(self, edge, time, solution_vars): 
		p_in = self.boundary_conditions(time, edge.start, solution_vars)
		p_out = self.boundary_conditions(time, edge.end, solution_vars)
		if p_in is None:
			p_in = self.calc_pressure(edge.start, time, solution_vars)
		elif p_out is None:
			p_out = self.calc_pressure(edge.end, time, solution_vars)
		return (p_in-p_out)/float(edge.value)

	def dydt(self, time, solution_vars):
		calculated = []
		for edge in self.ODEs:
			if edge.type == 'c':
				calculated.append(self.calc_c(edge, time, solution_vars))
			else:
				calculated.append(self.calc_i(edge, time, solution_vars))
		return calculated

	def rk4(self, h, n):
		initial_values = []
		# gather special nodes (capacitors/inductors)
		for edge in self.ODEs:
			if edge.type == 'c':
				self.special_nodes.append(edge.start)
				start = self.circuit.get_node(edge.start)['p']
				if type(start) == dict:
					if 0.0 in start:
						initial_values.append(start[0.0]) # 0 or 0.0
					else:
						initial_values.append(start[0])
				else:
					initial_values.append(start) # initial pressures loaded in
			else:
				self.special_nodes.append(edge)
				# case for inductors
				initial_values.append(0)

		values = [initial_values]
		half_h = np.float64(h*0.5)
		t = [0]

		for i in range(-1, n+1):
			if np.mod(i, 100) == 0:
				print('CurrentTime %d/%d ' % (i, n))

			t.append(t[0] + ((i+1)*h))
			k1 = [h*dy for dy in self.dydt(t[i+1], values[i])]
			wtemp = [ww + 0.5*kk1 for ww, kk1 in zip(values[i], k1)]

			time = np.float64(t[i+1] + half_h)
			self.cycle += half_h

			k2 = [h*dy for dy in self.dydt(time, wtemp)]
			wtemp = [ww + 0.5*kk2 for ww, kk2 in zip(values[i], k2)]
			k3 = [h*dy for dy in self.dydt(t[i+1]+half_h, wtemp)]
			wtemp = [ww + kk3 for ww, kk3 in zip(values[i], k3)]

			self.cycle += h
			k4 = [h*dy for dy in self.dydt(t[i+1]+h, wtemp)]
			values.append([ww + (1/6.)*(kk1+2.*kk2+2.*kk3+kk4) for ww, kk1, kk2, kk3, kk4 in zip(values[i], k1, k2, k3, k4)])


		return values, t

	def calculate(self, h, n):
		self.gather_odes()
		solution, time = self.rk4(h, n)

		test_sol = np.zeros(n+3)
		test_sol[0] = 90
		for i in range(0, n):
			test_sol[i+1] = solution[i+1][0]
		  
		plt.plot(time, test_sol)
		plt.title('Pressure vs. time in parallel circuit')
		plt.xlabel('Time (s)')
		plt.ylabel('Pressure (mmHg)')
		plt.show()



# calc = Calculator('initial_pressure.txt', 'test.xml', 0.001, 3000)
# calc = Calculator('intial_parallel_pressure.txt', 'parallel3.xml', 0.001, 2000)
calc = Calculator('heart_input.txt', 'heart_chamber.xml', 0.001, 5000)























