# RR_LAPD_1D_static.py
# based upon
# SOL_1D_example_CG_SU_DG.py (see m6c5_phase2 repo - solves a slightly different outflow problem)
# steady-state version of solution of 1D outflow problem based on 3D Rogers-Ricci system
# this is initial simplified form, no potential, electron velocity only, isothermal, solving simply:
# (nu)' = Sn = 2*0.03
#  u u' = - tau T n'/n
# tau is proxy for electron:ion mass ratio
# this system has an analytic solution using a Lambert W function, see below (not as bad as it sounds!)

from firedrake import *

meshres = 100
mesh = IntervalMesh(meshres,-1,1)

V1 = FunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V = V1*V2
nu = Function(V)
n, u = split(nu)  # n is density, u is velocity
v1, v2 = TestFunctions(V)

# model params
nstar = 2*0.03  # density source
T = 1.0  # temperature
h=2.0/meshres
tau = 1.0

# try initializing with analytic solution or near-analytic solution
x = SpatialCoordinate(mesh)
nu.sub(0).interpolate(0.99*(nstar/sqrt(T))*(1+sqrt(1-x[0]*x[0])))
nu.sub(1).interpolate(0.99*(sqrt(T)/(x[0]))*(1-sqrt(1-x[0]*x[0])))  # possible 0(^2)/0 eval at x=0?

# last two terms are the streamline-upwind correction to suppress grid-scale oscillations
a = (n*u*grad(v1)[0]+nstar*v1)*dx \
     - n*u*v2*grad(u)[0]*dx - tau*T*grad(n)[0]*v2*dx \
#     + (0.5*h*(grad(n)[0]-nstar)*grad(v1)[0])*dx \
#     + (0.5*h*(grad(n*u)[0])*grad(v2)[0])*dx

g = Function(V)

bc01 = DirichletBC(V.sub(0), nstar/sqrt(T), 1)
bc02 = DirichletBC(V.sub(0), nstar/sqrt(T), 2)
bc11 = DirichletBC(V.sub(1), -sqrt(T), 1)
bc12 = DirichletBC(V.sub(1), sqrt(T), 2)

params = {'snes_monitor': None}

#solve(a==0, nu, bcs=[bc11, bc12])  #TRIALCODE does not converge without all four BCs
solve(a==0, nu, bcs=[bc01, bc02, bc11, bc12], solver_parameters=params)

ns, us = nu.split()
ns.rename('n')
us.rename('u')

# analytic solution
from mpmath import *  # for analytic solution using Lambert W function (which just inverts x = y e^y)

alpha = exp(-0.5/(tau*T))/(2*0.03)  # constant of integration, fixed by sonic outflow BCs and source amp
n_analytic = Function(V1)
u_analytic = Function(V2)
for i in range(0,(meshres+1)):
   if mesh.coordinates.dat.data[i] < 0.0:
      u_analytic.dat.data[i] = -sqrt(-tau*T*lambertw(-mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha*alpha*(2*0.03*2*0.03)/(tau*T))).real
   else:
      u_analytic.dat.data[i] =  sqrt(-tau*T*lambertw(-mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha*alpha*(2*0.03*2*0.03)/(tau*T))).real

n_analytic.interpolate(2*0.03*x[0]/u_analytic)

n_analytic.rename('n_analytic')
u_analytic.rename('u_analytic')

File("RR_LAPD_1D_static.pvd").write(ns, us, n_analytic, u_analytic)
