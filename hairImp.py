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

class damping:
	def __init__(self, alpha):
		self.alpha = alpha

	def apply(self, mass, dt):
		if mass.iMass != 0.:
			I = -self.alpha * mass.vel * dt;
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
	def check(self,dt):
		dist = self.mass2.pos - self.mass1.pos;
		if linalg.norm(dist) != 0.:
			distUnit = dist / linalg.norm(dist);

			relVel = dot((self.mass2.vel - self.mass1.vel), distUnit);
			relDist = linalg.norm(dist) - self.refDist;
			print 'dist', relDist / dt + relVel

class bendingConstraint:
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
			if relDist < 0. :
				I = (relDist / dt + relVel) / (self.mass1.iMass + self.mass2.iMass) * distUnit;
				self.mass1.applyImpulse(I);
				self.mass2.applyImpulse(-I);

	def check(self,dt):
		dist = self.mass2.pos - self.mass1.pos;
		if linalg.norm(dist) != 0.:
			distUnit = dist / linalg.norm(dist);

			relVel = dot((self.mass2.vel - self.mass1.vel), distUnit);
			relDist = linalg.norm(dist) - self.refDist;
			print 'bending', relDist / dt + relVel
dt = 1e-3;

nbParticules = 15
nbConstraints = nbParticules-1;
nbBendingConstraints = nbParticules-2 - 2;

h = 0.1;

axisBound = [-nbParticules * h, nbParticules * h,  -nbParticules * h, nbParticules * h]
plt.clf();
plt.axis(axisBound)
plt.show();


particules = []
particules.append(mass(1e5, array([0.,0]), array([0,0])));
particules[0].iMass = 0.0; #fixed m0

particules.append(mass(1e5, array([h,0]), array([0,0])));
particules[1].iMass = 0.0; #fixed m1

for i in range(2, nbParticules):
	particules.append(mass(1e-5, array([h*(i-2),0]), array([0,0])));

constraints = []
constraints.append(distanceConstraint(particules[0], particules[2], 0.));
constraints.append(distanceConstraint(particules[1], particules[3], 0.));

for i in range(2, nbConstraints):
	constraints.append(distanceConstraint(particules[i], particules[i+1], h));

#bending constraint
theta = 20.0 * 3.1416 / 180.0;
minL = sqrt((h + h * cos(theta)) * (h + h * cos(theta)) + (h*sin(theta)) * (h*sin(theta)) );
print 'minL', minL, 2*h

bendingConstraints = []
#bendingConstraints.append(bendingConstraint(particules[2], particules[4], minL));

for i in range(2, 2 + nbBendingConstraints):
	bendingConstraints.append(bendingConstraint(particules[i], particules[i+2], minL));


grav = gravity(9.81, array([0.0, -1.0]));
damp = damping(0.1 * 1e-5);

for t in range(0,20000):


	#grav.apply(particules[0]);
	for i in range(2,nbParticules):
		damp.apply(particules[i], dt);
		grav.apply(particules[i], dt);

	for i in range(0,50):
		for i in range(0, nbConstraints):
			constraints[i].apply(dt);

		for i in range(0,nbBendingConstraints):
			bendingConstraints[i].apply(dt);

	#for i in range(0,nbConstraints):
	#	constraints[i].check(dt);

	#for i in range(0,nbBendingConstraints):
	#	bendingConstraints[i].check(dt);

	for i in range(0,nbParticules):
		particules[i].integrate(dt);

	#print '----------------------'
	#for i in range(0,nbBendingConstraints):
	#	bendingConstraints[i].check(dt);
	#print '#######################'

	posX = map(lambda p: p.pos[0], particules);
	posY = map(lambda p: p.pos[1], particules);



	plt.clf();
	fig = plt.figure(1)
	plt.plot(posX, posY, 'bo');
	plt.axis(axisBound)
	plt.draw();
	
	name = "imp%04d.png"%t
	fig.savefig(name, dpi=100)
#plt.show();



print raw_input('What is your name? ')
