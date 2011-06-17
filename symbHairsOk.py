from sympy import *;
from sympy.matrices import *

#physics params
h = 2.5;
m = 1.0;
g = 9.81;
dt = 1e-3;

x1 = Symbol("x1");
y1 = Symbol("y1");

x2 = Symbol("x2");
y2 = Symbol("y2");


gap = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))  - h;
fixed = sqrt(x1*x1 + y1*y1);

J = Matrix([diff(gap, x1), diff(gap,y1), diff(gap, x2), diff(gap, y2)]).transpose();
JFixed = Matrix([diff(fixed, x1), diff(fixed,y1), 0, 0]).transpose();

print J;

#init params
x_i = Matrix([0., 0.+0.01, 0., h*1.01]);
v_i = Matrix([0., 0., 0., 0.]);
a_i = Matrix([0., 0., 0., 0.]);
w = Matrix([0., 0., 0., -m*g]);

for t in range(0,500):
	a_i = Matrix([0., 0., 0., 0.]);
	v_t = v_i + a_i * dt;
	x_t = x_i + v_t * dt;

	JFixed_t = JFixed.subs(x1, x_t[0]).subs(y1, x_t[1]);
	J_t = J.subs(x1, x_t[0]).subs(y1, x_t[1]).subs(x2, x_t[2]).subs(y2, x_t[3]);
	
	gJ_t = JFixed_t.col_join(J_t);
	delassus = dt*dt * gJ_t * gJ_t.transpose() / m;

	print 'delassus', delassus

	z = Matrix([0, 0]);

	#solve fixed constraint

	d = fixed.subs(x1, x_t[0]).subs(y1, x_t[1]);
	gii = delassus[0,0];

	print 'initial gap', d

	q = d + dt*dt / m * (gJ * w)[0,0];
	q += delassus[0:] * z[0,0];
	z[0,0] += -q / gii;

	#solve second constraint

	d = gap.subs(x1, x_t[0]).subs(y1, x_t[1]).subs(x2, x_t[2]).subs(y2, x_t[3]);
	z = 0.
	gii = dt*dt * (J_t * J_t.transpose())[0,0] / m;

	print 'initial gap', d

	q = d + dt*dt / m * (J_t * w)[0,0];
	q += gii * z;
	z += -q / gii;

	force = force + J_t.transpose() * z;
	print 'force', force

	a_i = force / m;

	v_i += a_i * dt;
	x_i += v_i * dt;

	print 'final gap ------------------>', gap.subs(x1, x_i[0]).subs(y1, x_i[1]).subs(x2, x_i[2]).subs(y2, x_i[3]);
	print 'pos', x_i
	print ''
	print raw_input('What is your name? ')
 





