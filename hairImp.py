import matplotlib;

import matplotlib.pyplot as plt
from math import sqrt
import copy
from numpy import *


class mass:
	def __init__(self, mass, pos, vel):
		self.mass = mass;
		self.iMass = 1.0 / mass;
		self.pos = pos;
		self.vel = vel;

	def applyImpulse(self, imp):
		self.vel = self.vel + imp / self.mass;

	def integrate(self, dt):
		self.pos = self.pos + dt * self.vel;


class gravity:
	def __init__(self, g, axis):
		self.g = g;
		self.axis = axis

	def apply(self, mass, dt):
		I = self.axis*self.g * dt / mass.iMass;
		mass.applyImpulse(I);

class distanceConstraint:
	def __init__(self, mass1, mass2, refDist):
		self.mass1 = mass1;
		self.mass2 = mass2;
		self.refDist = refDist;

	def apply(self,dt):
		dist = self.mass2.pos - self.mass1.pos;
		if linalg.norm(dist) != 0.:
			distUnit = dist / linalg.norm(dist);

			relVel = dot((self.mass2.vel - self.mass1.vel), distUnit);
			relDist = linalg.norm(dist) - self.refDist;

			I = (relDist / dt + relVel) / (self.mass1.iMass + self.mass2.iMass) * distUnit;
			self.mass1.applyImpulse(I);
			self.mass2.applyImpulse(-I);

dt = 1e-3;

nbParticules = 50;
nbConstraints = nbParticules-1;

h = 0.1;

axisBound = [-nbParticules * h, nbParticules * h,  1.0-nbParticules * h, 1.0+nbParticules * h]

particules = []
particules.append(mass(1e5, array([0.,0]), array([0,0])));
particules[0].iMass = 0.0; #fixed m0

for i in range(1, nbParticules):
	particules.append(mass(1e-5, array([h*(i-1),0]), array([0,0])));

constraints = []
constraints.append(distanceConstraint(particules[0], particules[1], 0.));

for i in range(1, nbConstraints):
	constraints.append(distanceConstraint(particules[i], particules[i+1], h));

grav = gravity(9.81, array([0.0, -1.0]));

for t in range(0,10000):


	#grav.apply(particules[0]);
	for i in range(1,nbParticules):
		grav.apply(particules[i], dt);

	for i in range(0,nbConstraints):
		constraints[i].apply(dt);

	for i in range(0,nbParticules):
		particules[i].integrate(dt);

	posX = map(lambda p: p.pos[0], particules);
	posY = map(lambda p: p.pos[1], particules);



	plt.clf();
	plt.plot(posX, posY, 'bo');
	plt.axis(axisBound)
	plt.draw();
	plt.show();



print raw_input('What is your name? ')
