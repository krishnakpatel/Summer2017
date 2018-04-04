def gather_abc(self, flows, unknown):
	a = 0
	b = 0
	c = 1
	for flow in flows:
		c *= flow[2]
		a += flow[2]
		b_temp = flow[1]
		for f in flows:
			if f is not flow:
				b_temp *= f[2]
		b -= b_temp

