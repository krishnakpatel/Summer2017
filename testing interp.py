import numpy as np

file = open('pressures.txt')
x = []
y = []
for line in file:
    entry = line.split()
    x.append(float(entry[0]))
    y.append(float(entry[1]))

for a in range(0, 5):
    print(np.interp(a, x, y))