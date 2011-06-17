import matplotlib;

import matplotlib.pyplot as plt
from math import sqrt
from numpy import *


dt=1e-2;
nbMass = 6;
m = 1e-5;
g = 9.81;
mass = [m]*nbMass;
h = 0.1

firstMassPos = array([0, 1.0]);
pos = map(lambda i : array([i*h, 1.0]), range(0, nbMass));
vel = [array([0,0])]*nbMass;
acc = [array([0,0])]*nbMass;
weight = [array([0, -m*g])]*nbMass;
weight[0] = array([0. , 0.])

posX = map(lambda p: p[0], pos);
posY = map(lambda p: p[1], pos);
axisBound = [-nbMass * h, nbMass * h,  1.0-nbMass * h, 1.0+nbMass * h]
plt.plot(posX, posY, 'ro');
plt.axis(axisBound)
plt.show();

def dist(a,b):
	return linalg.norm(a-b);

def constrainedPos(c, p, h):
	norm = dist(c,p);
	if norm != 0.:
		d = (p - c) / norm;
		return p -h * d;
	else:
		return c

def constrainedForce(c, p, h, m, dt):
	nP = constrainedPos(c, p, h);
	cForce = (nP - c) / (dt*dt/m);
	return cForce;


for i in range(1,10000):
	acc = map(lambda m, f : f/m, mass, weight)
	gap = map(lambda v, a :  v * dt + a * dt*dt, vel, acc);

	fC = [array([0, 0])]*nbMass;
	cForce = constrainedForce(pos[0] + gap[0], firstMassPos, 0., m, dt);
	fC[0] = array(cForce);

	for i in range(1, nbMass):
		cForce = constrainedForce(array(pos[i]) + array(gap[i]), array(pos[i-1]) + array(gap[i-1]), h, m, dt);
		fC[i] = array(cForce) - array(fC[i-1]) ;

	fC  =map(lambda f, w : w+f, fC, weight);

	acc = map(lambda m, f : f/m, mass, fC )
	vel = map(lambda v, a : v + a * dt, vel, acc );
	pos = map(lambda p, v : p + v * dt, pos, vel);

	#gap = map(lambda c, p : dist(c,p) - h, pos[1:],pos[:len(pos)-1])
	#gap = map(lambda c, p : dist(c,p) - h, cPos[1:],cPos[:len(pos)-1])
	#print gap

	posX = map(lambda p: p[0], pos);
	posY = map(lambda p: p[1], pos);

	plt.clf();
	plt.plot(posX, posY, 'ro');
	plt.axis(axisBound)
	plt.draw();
	#print raw_input('go one')



print raw_input('What is your name? ')
