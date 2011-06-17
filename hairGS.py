import matplotlib;

import matplotlib.pyplot as plt
from math import sqrt
import copy


dt=1e-2;
nbMass = 10;
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
axisBound = [-nbMass * h, nbMass * h,  1.0-nbMass * h, 1.0+nbMass * h]
plt.plot(posX, posY, 'ro');
plt.axis(axisBound)
plt.show();

def dist(a,b):
	return sqrt( (a[0]-b[0]) * (a[0]-b[0]) + (a[1]-b[1]) * (a[1]-b[1]));

def constrainedPos(c, p, h):
	norm = dist(c,p);
	if norm != 0.:
		d = ( (p[0] - c[0]) / norm, (p[1] - c[1]) / norm);
		return (p[0] - h * d[0], p[1] - h * d[1]);
	else:
		return c

def constrainedForce(c, p, h, m, dt):
	nP = constrainedPos(c, p, h);
	cForce = ((nP[0] - c[0]) /(dt*dt/m), (nP[1] - c[1]) / (dt*dt/m));
	return cForce;


for i in range(1,10000):
	#free motion
	f = copy.copy(weight)
	f = map(lambda f, v : (f[0] - 0.0001 * v[0], f[1] - 0.0001 * v[1]), f, vel)
	acc = map(lambda m, f : (f[0]/m, f[1]/m), mass, f );
	vel = map(lambda v, a : (v[0] + a[0] * dt, v[1] + a[1] * dt), vel, acc );
	cPos = map(lambda p, v : (p[0] + v[0] * dt, p[1] + v[1] * dt), pos, vel);

	#constrained motion

	posX = map(lambda p: p[0], cPos);
	posY = map(lambda p: p[1], cPos);

	plt.clf();
	plt.plot(posX, posY, 'bo');
	plt.axis(axisBound)
	plt.draw();

	fC =  [(0,0)]*nbMass;
	cForce = constrainedForce(cPos[0], firstMassPos, 0., m, dt)
	fC[0] = (cForce[0], cForce[1])
	cPos[0] = (cPos[0][0] + fC[0][0] * (dt*dt/m), cPos[0][1] + fC[0][1] * (dt*dt/m))

	for i in range(1, nbMass):
		cForce = constrainedForce(cPos[i], cPos[i-1], h, m, dt)
		fC[i] = (cForce[0], cForce[1])
		cPos[i] = (cPos[i][0] + fC[i][0] / m * dt*dt, cPos[i][1] + fC[i][1] / m * dt*dt)

	vel = map(lambda f, v: (f[0] / (m/dt) + v[0], f[1] / (m/dt) + v[1]), fC, vel);
	pos = map(lambda p, v : (p[0] + v[0] * dt, p[1] + v[1] * dt), pos, vel);

	#gap = map(lambda c, p : dist(c,p) - h, pos[1:],pos[:len(pos)-1])
	#print 'pos',gap
	#gap = map(lambda c, p : dist(c,p) - h, cPos[1:],cPos[:len(pos)-1])
	#print 'cpos',gap

	posX = map(lambda p: p[0], pos);
	posY = map(lambda p: p[1], pos);

	plt.plot(posX, posY, 'ro');
	plt.axis(axisBound)
	plt.draw();
	#print raw_input('go one')



print raw_input('What is your name? ')
