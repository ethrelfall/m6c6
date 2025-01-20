# RR_LAPD_1D_T.py
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

# this is first attempt at a non-isothermal 1D system
# it has u_e = u_i (unique velocity) so there is no parallel current
# there is no potential either
# see also old work on 1D systems, SOLdrake - M2.2.2 NEPTUNE report.  But note sources are different.

from firedrake import *
import math
from irksome import Dt, GaussLegendre, MeshConstant, TimeStepper

meshres = 100
mesh = IntervalMesh(meshres, -1, 1)
V1 = FunctionSpace(mesh, "DG", 1)
V2 = VectorFunctionSpace(mesh, "CG", 1)  # IMPORTANT velocity space needs to be continuous
V3 = FunctionSpace(mesh, "DG", 1)  # T
V = V1*V2*V3

# time parameters
T0 = 50.0
timeres = 5000
t = Constant(0.0)
dt = Constant(T0/timeres)
skip=50

# parameters for irksome
butcher_tableau = GaussLegendre(2)

# model parameters
nstar = Constant(2.0*0.03)
Temp = 1.0
tau = 1.0  # will be ion:electron mass ratio
nu_param = 0.03

x = SpatialCoordinate(mesh)
nuT = Function(V)
n, u, T = split(nuT)
#n, u = nu.split()  # doesn't work with Irksome
v1, v2, v3 = TestFunctions(V)

# Gaussian blob init data or flat init data
amp = 0.01  # harder to get successful convergence if this is increased ...
width = 0.1
nuT.sub(0).interpolate((0.010+amp*(1/sqrt(2*math.pi*width**2))*exp(-x[0]**2/(2*width**2))))
nuT.sub(1).interpolate(as_vector([0.0+1.0*x[0]]))
nuT.sub(2).interpolate((1.00+0.0*amp*(1/sqrt(2*math.pi*width**2))*exp(-x[0]**2/(2*width**2))))

# source function
nstarFunc = Function(V)
nstarFunc.sub(0).interpolate(nstar + 0.0*x[0])

# TRIALCODE check init data
File("RR_LAPD_1D_T_init.pvd").write(nuT.sub(0), nuT.sub(1), nuT.sub(2))
#quit()

norm = FacetNormal(mesh)
u_n = 0.5*(dot(u,norm)+abs(dot(u,norm)))

# outflow BC imposed weakly in here, penalty term in ds (note conditional not really needed here as both are outflow)

# integrated press grad term by parts (or its not stable) + seem to get away without a surface term from doing that (due to mixed DG/CG?) ...

heat_suppress = 1.0  # artificially slow down heat transport i.e. slow down tdot timescale, try e.g. 0.1

# no parallel current, v_e = v_i
# this relaxes back to reasonable-looking equilibrium

F = -Dt(n)*v1*dx + (n*dot(u, grad(v1))+nstar*v1)*dx \
   - (v1('+') - v1('-'))*(u_n('+')*n('+') - u_n('-')*n('-'))*dS \
    -n*dot(Dt(u), v2) *dx - n*u[0]*v2[0]*grad(u[0])[0]*dx + tau*grad(T*v2[0])[0]*n*dx \
   - conditional(dot(u, norm) > 0, v1*dot(u, norm)*n, 0.0)*ds \
    -dot(Dt(T), v3) *dx + heat_suppress*(nstar*v3)*dx \
    +(2.0/3.0)*heat_suppress*(T*dot(u, grad(v3)))*dx \
    -(1.0/3.0)*heat_suppress*(dot(u,grad(T))*v3)*dx \
   - (2.0/3.0)*heat_suppress*(v3('+') - v3('-'))*(u_n('+')*T('+') - u_n('-')*T('-'))*dS \
   - (2.0/3.0)*heat_suppress*conditional(dot(u, norm) > 0, v3*dot(u, norm)*T, 0.0)*ds \
   - conditional(dot(u, norm) > 0, v2[0]*dot(u, norm)*u[0]*n, 0.0)*ds \
#   - conditional(dot(u, norm) > 0, v2[0]*T*n, 0.0)*ds \

# penultimate bdy term is needed to avoid boundary artifacts
# final bdy term is experimental and is WRONG


#F = -Dt(n)*v1*dx + (n*dot(u, grad(v1))+nstar*v1)*dx \
#   - (v1('+') - v1('-'))*(u_n('+')*n('+') - u_n('-')*n('-'))*dS \
#    -n*dot(Dt(u), v2) *dx - n*u[0]*v2[0]*grad(u[0])[0]*dx + tau*T*grad(v2[0])[0]*n*dx - tau*nu_param*n*n*u[0]*v2[0]*dx \
#   - conditional(dot(u, norm) > 0, v1*dot(u, norm)*n, 0.0)*ds \
#    -n*dot(Dt(T), v3) *dx + heat_suppress*(2.0/3.0)*0.71*T*(n*dot(u, grad(v3))+n*nstar*v3)*dx \
#    - heat_suppress*(2.0/3.0)*T*v3*grad(u[0])[0]*n*dx - heat_suppress*n*u[0]*grad(T)[0]*v3*dx\
#   - heat_suppress*0.71*(2.0/3.0)*(v3('+') - v3('-'))*(u_n('+')*n('+')*T('+') - u_n('-')*n('-')*T('-'))*dS \
#   - conditional(dot(u, norm) > 0, heat_suppress*0.71*(2.0/3.0)*T*v3*dot(u, norm)*n, 0.0)*ds \


# params taken from Cahn-Hilliard example cited above
params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'l2',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

# Dirichlet BCs are needed for boundary velocity

bc_test1 = DirichletBC(V.sub(1),as_vector([-1.0]),1)
bc_test2 = DirichletBC(V.sub(1),as_vector([1.0]),2)

# removed velocity BC.  Seems to work but not clear whether outflow is sonic.
stepper = TimeStepper(F, butcher_tableau, t, dt, nuT, solver_parameters=params)
#stepper = TimeStepper(F, butcher_tableau, t, dt, nuT, solver_parameters=params, bcs=[bc_test1, bc_test2])

outfile = File("RR_LAPD_1D_T.pvd")

nuT.sub(0).rename("n")
nuT.sub(1).rename("u")
nuT.sub(2).rename("T")

cnt=0

while float(t) < float(T0):
    if (float(t) + float(dt)) >= T0:
        dt.assign(T0 - float(t))
    if (cnt%skip)==0:
       outfile.write(nuT.sub(0), nuT.sub(1), nuT.sub(2))
       print("done output\n")
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(float(t), float(dt))
    cnt = cnt+1

print("done.")
print("\n")

ns, us, Ts = nuT.split()
ns.rename("density")
us.rename("velocity")
Ts.rename("temperature")
File("RR_LAPD_1D_T_final.pvd").write(ns, us, Ts)
