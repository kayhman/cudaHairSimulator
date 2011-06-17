import matplotlib;

import matplotlib.pyplot as plt
from math import sqrt

dt=1e-2;
nbMass = 5;
m = 1e-5;
g = 9.81;
mass = [m]*nbMass;
h = 0.1

firstMassPos = (0, 1.0);
pos = map(lambda i : (i*h, 1.0), range(0, nbMass));
vel = [(0,0)]*nbMass;
acc = [(0,0)]*nbMass;
weight = [(0, -m*g)]*nbMass;

posX = map(lambda p: p[0], pos);
posY = map(lambda p: p[1], pos);
plt.plot(posX, posY, 'ro');
plt.axis([-1,1, -1,1])
plt.show();

def dist(a,b):
	return sqrt( (a[0]-b[0]) * (a[0]-b[0]) + (a[1]-b[1]) * (a[1]-b[1]));

def constrainedPos(c, p, h):
	norm = dist(c,p);
	d = ( (p[0] - c[0]) / norm, (p[1] - c[1]) / norm);
	return (p[0] - h * d[0], p[1] - h * d[1]);

for i in range(1,100):
	#free motion
	print ('init vel', vel);
	acc = map(lambda m, f : (f[0]/m, f[1]/m), mass, weight );
	vel = map(lambda v, a : (v[0] + a[0] * dt, v[1] + a[1] * dt), vel, acc );
	pos = map(lambda p, v : (p[0] + v[0] * dt, p[1] + v[1] * dt), pos, vel);

	#print ('free pos', pos);
	#constrained motion
	
	posX = map(lambda p: p[0], pos);
	posY = map(lambda p: p[1], pos);

	plt.plot(posX, posY, 'bo');
	#plt.axis([0,2, 0,2])
	pos[0] = firstMassPos;
	plt.axis([-1,1, -1,1])
	plt.draw();
	for i in range(1, nbMass):
		pos[i] = constrainedPos(pos[i], pos[i-1], h)

	gap = map(lambda c, p : dist(c,p) - h, pos[1:],pos[:len(pos)-1])
	#print gap
	#print ('constraint pos', pos);

	posX = map(lambda p: p[0], pos);
	posY = map(lambda p: p[1], pos);

	plt.plot(posX, posY, 'ro');
	plt.axis([-1,1, -1,1])
	plt.draw();
	print raw_input('go one')



print raw_input('What is your name? ')
