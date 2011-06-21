import matplotlib.pyplot as plt
from sympy import *;
from sympy.matrices import *

#physics params
h = 2.5;
m = 1.0;
g = 9.81;
dt = 1e-3;

axisBound = [- 2*h, 2*h,  - 2*h, + 2*h]

x1 = Symbol("x1");
y1 = Symbol("y1");

x2 = Symbol("x2");
y2 = Symbol("y2");

x3 = Symbol("x3");
y3 = Symbol("y3");



gap1 = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))  - h;
gap2 = sqrt((x2-x3)*(x2-x3) + (y2-y3)*(y2-y3))  - h;
fixed = sqrt(x1*x1 + y1*y1);

J1 = Matrix([diff(gap1, x1), diff(gap1,y1), diff(gap1, x2), diff(gap1, y2), 0, 0]).transpose();
J2 = Matrix([0, 0, diff(gap2, x2), diff(gap2,y2), diff(gap2, x3), diff(gap2, y3)]).transpose();
JFixed = Matrix([diff(fixed, x1), diff(fixed,y1), 0, 0, 0, 0]).transpose();

#print J;

#init params
x_i = Matrix([0., 0.+0.01, h*1.01, 0., h*2.01, 0.]);
v_i = Matrix([0., 0., 0., 0., 0., 0.]);
a_i = Matrix([0., 0., 0., 0., 0., 0.]);
w = Matrix([0., 0., 0., -m*g, 0., -m*g]);

plt.clf();
plt.plot([x_i[0], x_i[2], x_i[4]], [x_i[1], x_i[3], x_i[5]] , 'ro');
plt.axis(axisBound)
plt.show();

for t in range(0,5000):
	a_i = Matrix([0., 0., 0., 0., 0., 0.]);
	v_t = v_i + a_i * dt;
	x_t = x_i + v_t * dt;

	JFixed_t = JFixed.subs(x1, x_t[0]).subs(y1, x_t[1]);
	J1_t = J1.subs(x1, x_t[0]).subs(y1, x_t[1]).subs(x2, x_t[2]).subs(y2, x_t[3]);
	J2_t = J2.subs(x2, x_t[2]).subs(y2, x_t[3]).subs(x3, x_t[4]).subs(y3, x_t[5]);
	
	gJ_t = JFixed_t.col_join(J1_t).col_join(J2_t);
	delassus = dt*dt * gJ_t * gJ_t.transpose() / m;

	#print 'delassus', delassus

	z = Matrix([0, 0, 0]);

	#solve fixed constraint

	d = fixed.subs(x1, x_t[0]).subs(y1, x_t[1]);
	gii = delassus[0,0];

	#print 'initial gap', d

	q = d + dt*dt / m * (gJ_t * w)[0];
	q += (delassus[0, :] * z[0])[0,0];
	z[0] += -q / gii;

	#solve second constraint

	d = gap1.subs(x1, x_t[0]).subs(y1, x_t[1]).subs(x2, x_t[2]).subs(y2, x_t[3]);
	gii = delassus[1,1];

	#print 'initial gap1', d
	#print 'initial gii1', gii

	q = d + dt*dt / m * (gJ_t * w)[1];
	q += (delassus[1,:] * z)[0,0];
	z[1] += -q / gii;

	#solve third constraint

	d = gap2.subs(x2, x_t[2]).subs(y2, x_t[3]).subs(x3, x_t[4]).subs(y3, x_t[5]);
	gii = delassus[2,2];

	#print 'initial gap2', d
	#print 'initial gii2', gii

	q = d + dt*dt / m * (gJ_t * w)[2];
	q += (delassus[2,:] * z)[0,0];
	z[2] += -q / gii;

	force = w + gJ_t.transpose() * z;
	#print 'force', force

	a_i = force / m;

	v_i += a_i * dt;
	x_i += v_i * dt;

	plt.clf();
	fig = plt.figure(1)
	plt.plot([x_i[0], x_i[2], x_i[4]], [x_i[1], x_i[3], x_i[5]] , 'ro');
	plt.axis(axisBound)
	plt.draw();

	name = "plot%04d.png"%t
	fig.savefig(name, dpi=100)


	#print 'final gap1 ------------------>', gap1.subs(x1, x_i[0]).subs(y1, x_i[1]).subs(x2, x_i[2]).subs(y2, x_i[3]);
	#print 'final gap2 ------------------>', gap2.subs(x2, x_i[2]).subs(y2, x_i[3]).subs(x3, x_i[4]).subs(y3, x_i[5]);
	#print 'final fixed ------------------>', fixed.subs(x1, x_i[0]).subs(y1, x_i[1]);
	#print 'pos', x_i
	#print ''
	#print raw_input('What is your name? ')
 





