# RR_LAPD_1D_T_conservative_v2.py
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

Newton_its = 10

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
nstar = 1.0  #2.0*0.03 in Rogers-Ricci BEWARE MIGHT BE RESTRICTIONS ON THIS VIZ. PHYSICAL SOLUTIONS
Temp = 1.0
tau = 1.0  # will be ion:electron mass ratio

x = SpatialCoordinate(mesh)
nuT = Function(V)
n, u, T = split(nuT)
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
#File("RR_LAPD_1D_T_conservative_v2_init.pvd").write(nuT.sub(0), nuT.sub(1), nuT.sub(2))
#quit()

norm = FacetNormal(mesh)
u_n = 0.5*(dot(u,norm)+abs(dot(u,norm)))

# outflow BC imposed weakly in here, penalty term in ds (note conditional not really needed here as both are outflow)

# integrated press grad term by parts (or its not stable) + seem to get away without a surface term from doing that (due to mixed DG/CG?) ...

# no parallel current, v_e = v_i
# this relaxes back to reasonable-looking equilibrium

C = 2.0  # coeff of conservative flux term in T eq, added 20 Jan 2025 to make Nektar++ implementation more straightforward

F = -Dt(n)*v1*dx + (n*dot(u, grad(v1))+nstar*v1)*dx \
   - (v1('+') - v1('-'))*(u_n('+')*n('+') - u_n('-')*n('-'))*dS \
    -n*dot(Dt(u), v2) *dx - n*u[0]*v2[0]*grad(u[0])[0]*dx + tau*grad(v2[0])[0]*n*T*dx \
   - conditional(dot(u, norm) > 0, v1*dot(u, norm)*n, 0.0)*ds \
    -dot(Dt(T), v3) *dx + (nstar*v3)*dx \
    +(C)*(T*dot(u, grad(v3)))*dx \
   - (C)*(v3('+') - v3('-'))*(u_n('+')*T('+') - u_n('-')*T('-'))*dS \
   - conditional(dot(u, norm) > 0, (C)*v3*dot(u, norm)*T, 0.0)*ds \
   - conditional(dot(u, norm) > 0, v2[0]*dot(u, norm)*u[0]*n, 0.0)*ds \
   - conditional(dot(u, norm) > 0, (C)*v2[0]*dot(u, norm)*u[0]*T, 0.0)*ds \

# params taken from Cahn-Hilliard example cited above
params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'l2',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

# Dirichlet BCs for boundary velocity
bc_test1 = DirichletBC(V.sub(1),as_vector([-1.0]),1)
bc_test2 = DirichletBC(V.sub(1),as_vector([1.0]),2)

# removed velocity BC.  Seems to work but not clear whether outflow is sonic.
stepper = TimeStepper(F, butcher_tableau, t, dt, nuT, solver_parameters=params)
#stepper = TimeStepper(F, butcher_tableau, t, dt, nuT, solver_parameters=params, bcs=[bc_test1, bc_test2])

outfile = File("RR_LAPD_1D_T_conservative_v2.pvd")

nuT.sub(0).rename("n")
nuT.sub(1).rename("u")
nuT.sub(2).rename("T")

# analytic solution
from mpmath import *

n_analytic = Function(V1)
u_analytic = Function(V2)
T_analytic = Function(V3)

# parameters for cubic assuming sonic (sqrt(tau T)) electron outflow
qby2 = 0.5*4.0*tau*nstar/C
pby3 = (1.0/3.0)*(pow(2.0*C,-2./3.)+4.0*nstar/(C*pow(2.0*C,-1./3.)))

for i in range(0,(meshres+1)):
   xc = mesh.coordinates.dat.data[i]
   # can't seem to get the analytic solution onto the correct solution branch
   # so try out Newton-Raphson
   uguess=0.0
   for idx in range(0,Newton_its):
      uguess = uguess - (uguess*uguess*uguess-3.0*pby3*uguess+2.0*qby2*xc)/(3.0*uguess*uguess-3.0*pby3)
   u_analytic.dat.data[i] = uguess

n_analytic.interpolate(nstar*x[0]/u_analytic[0])  # done like this because DG space has additional data points not in 1:1 corresp with mesh points
T_analytic.interpolate((1.0/C)*nstar*x[0]/u_analytic[0])  # done like this because DG space has additional data points not in 1:1 corresp with mesh points

n_analytic.rename('n_analytic')
u_analytic.rename('u_analytic')
T_analytic.rename('T_analytic')




cnt=0
while float(t) < float(T0):
    if (float(t) + float(dt)) >= T0:
        dt.assign(T0 - float(t))
    if (cnt%skip)==0:
       outfile.write(nuT.sub(0), nuT.sub(1), nuT.sub(2), n_analytic, u_analytic, T_analytic)
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
File("RR_LAPD_1D_T_conservative_v2_final.pvd").write(ns, us, Ts, n_analytic, u_analytic, T_analytic)
