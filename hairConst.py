import matplotlib;

import matplotlib.pyplot as plt
from math import sqrt

dt=1e-3;
nbMass = 5;
m = 1e-5;
g = 9.81;
mass = [m]*nbMass;
h = 0.1

firstMassPos = (-h, 1.0);
pos = map(lambda i : (i*h, 1.0), range(0, nbMass));
vel = [(0,0)]*nbMass;
acc = [(0,0)]*nbMass;
weight = [(0, -m*g)]*nbMass;

posX = map(lambda p: p[0], pos);
posY = map(lambda p: p[1], pos);
plt.plot(posX, posY, 'ro');
plt.axis([0,2, 0,2])
plt.show();

def dist(a,b):
	return sqrt( (a[0]-b[0]) * (a[0]-b[0]) + (a[1]-b[1]) * (a[1]-b[1]));

for i in range(1,10):
	#free motion
	acc = map(lambda m, f : (f[0]/m, f[1]/m), mass, weight );
	#constrained motion
	gap = map(lambda c, p : (c[0] - (p[0]+h), c[1] - p[1] ), pos,[firstMassPos] + pos[:len(pos)-1])

	gap = map(lambda g, a : (g[0] + a[0] * dt*dt, g[1] + a[1] * dt*dt), gap, acc)
	f = map(lambda g, m : (-g[0] / (dt*dt/m), -g[1] / (dt*dt/m)), gap, mass)
	#print gap
	
	f = map(lambda ff, w : (ff[0] + w[0], ff[1] + w[1]), f, weight)
	print f

	acc = map(lambda m, f : (f[0]/m, f[1]/m), mass, f );
	vel = map(lambda v, a : (v[0] + a[0] * dt, v[1] + a[1] * dt), vel, acc );
	pos = map(lambda p, v : (p[0] + v[0] * dt, p[1] + v[1] * dt), pos, vel);
	posX = map(lambda p: p[0], pos);
	posY = map(lambda p: p[1], pos);

	plt.plot(posX, posY, 'ro');
	#plt.axis([0,2, 0,2])
	plt.draw();



print raw_input('What is your name? ')
