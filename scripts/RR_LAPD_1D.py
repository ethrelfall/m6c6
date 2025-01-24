# RR_LAPD_1D.py
# originally based upon:
# SOL_1D_DG_upwind_irksome.py
# which was
# attempt at time-dependent solver of 1D SOL equations, with upwind flux
# based on:
# https://www.firedrakeproject.org/demos/DG_advection.py.html
# irksome implementation follows
# https://www.firedrakeproject.org/Irksome/demos/demo_cahnhilliard.py.html
# and
# https://www.firedrakeproject.org/Irksome/demos/demo_monodomain_FHN.py.html

# time-dependent version of solution of 1D outflow problem based on 3D Rogers-Ricci system
# this is initial simplified form, solving simply:
# ndot = - (nu)' + (Sn = 2*0.03)
# udot = - u u' - tau T n'/n
# the steady state of this system has an analytic solution using a Lambert W function, see below (not as bad as it sounds!)

from firedrake import *
import math
from irksome import Dt, GaussLegendre, MeshConstant, TimeStepper
#from mpmath import *
#from mpmath import lambertw

meshres = 1000
mesh = IntervalMesh(meshres, -1, 1)
V1 = FunctionSpace(mesh, "DG", 1)
V2 = VectorFunctionSpace(mesh, "CG", 1)  # IMPORTANT velocity space needs to be continuous
V = V1*V2

# time parameters
T = 10.0
timeres = 200
t = Constant(0.0)
dt = Constant(T/timeres)

# parameters for irksome
butcher_tableau = GaussLegendre(2)

# model parameters
nstar = Constant(2.0*0.03)
Temp = 1.0
tau = 1.0  # will be ion:electron mass ratio
T0 = 1.0  # temperature

x = SpatialCoordinate(mesh)
nu = Function(V)
n, u = split(nu)
#n, u = nu.split()  # doesn't work with Irksome
v1, v2 = TestFunctions(V)

# Gaussian blob init data or flat init data
amp = 0.01  # harder to get successful convergence if this is increased ...
width = 0.1
nu.sub(0).interpolate((0.010+amp*(1/sqrt(2*math.pi*width**2))*exp(-x[0]**2/(2*width**2))))
nu.sub(1).interpolate(as_vector([0.0+1.0*x[0]]))

# source function
nstarFunc = Function(V)
nstarFunc.sub(0).interpolate(nstar + 0.0*x[0])

# TRIALCODE check init data
File("RR_LAPD_1D_init.pvd").write(nu.sub(0), nu.sub(1))
#quit()

norm = FacetNormal(mesh)
u_n = 0.5*(dot(u,norm)+abs(dot(u,norm)))

# outflow BC imposed weakly in here, penalty term in ds (note conditional not really needed here as both are outflow)

# integrated press grad term by parts (or its not stable) + seem to get away without a surface term from doing that (due to mixed DG/CG?) ...

F = -Dt(n)*v1*dx + (n*dot(u, grad(v1))+nstar*v1)*dx \
   - (v1('+') - v1('-'))*(u_n('+')*n('+') - u_n('-')*n('-'))*dS \
    -n*dot(Dt(u), v2) *dx - n*u[0]*v2[0]*grad(u[0])[0]*dx + tau*Temp*grad(v2[0])[0]*n*dx \
   - conditional(dot(u, norm) > 0, v1*dot(u, norm)*n, 0.0)*ds \
   - conditional(dot(u, norm) > 0, v2[0]*dot(u, norm)*u[0]*n, 0.0)*ds \

# params taken from Cahn-Hilliard example cited above
params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'l2',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

# Dirichlet BCs are needed for boundary velocity

bc_test1 = DirichletBC(V.sub(1),as_vector([-1.0]),1)
bc_test2 = DirichletBC(V.sub(1),as_vector([1.0]),2)

stepper = TimeStepper(F, butcher_tableau, t, dt, nu, solver_parameters=params, bcs=[bc_test1, bc_test2])

outfile = File("RR_LAPD_1D.pvd")

nu.sub(0).rename("n")
nu.sub(1).rename("u")

# analytic solution

from mpmath import *

alpha = exp(-0.5/(tau*T0))/(2*0.03)  # constant of integration, fixed by source amp and sonic outflow BCs
n_analytic = Function(V1)
u_analytic = Function(V2)
for i in range(0,(meshres+1)):
   if mesh.coordinates.dat.data[i] < 0.0:
      u_analytic.dat.data[i] = -sqrt(-tau*T0*lambertw(-mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha*alpha*(2*0.03*2*0.03)/(tau*T0))).real
   else:
      u_analytic.dat.data[i] =  sqrt(-tau*T0*lambertw(-mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha*alpha*(2*0.03*2*0.03)/(tau*T0))).real

n_analytic.interpolate(2*0.03*x[0]/u_analytic[0])  # done like this because DG space has additional data points not in 1:1 corresp with mesh points

n_analytic.rename('n_analytic')
u_analytic.rename('u_analytic')

# end analytic solution

cnt=0

while float(t) < float(T):
    if (float(t) + float(dt)) >= T:
        dt.assign(T - float(t))
    if (cnt%2)==0:
       outfile.write(nu.sub(0), nu.sub(1), n_analytic, u_analytic)
       print("done output\n")
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(float(t), float(dt))
    cnt = cnt+1

print("done.")
print("\n")

ns, us = nu.split()
ns.rename("density")
us.rename("velocity")
File("RR_LAPD_1D_final.pvd").write(ns, us)
