# RR_LAPD_1D_T_par_current.py
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

# this is first attempt at a non-isothermal 1D system with electron and ion fluids
# i.e. there is parallel current
# there is no potential yet
# see also old work on 1D systems, SOLdrake - M2.2.2 NEPTUNE report.  But note sources are different.
# uses domain 36 units long - see de-dimensionalized eqs at
# https://github.com/ExCALIBUR-NEPTUNE/firedrake-lapd/blob/rogers-ricci_v1-debug/docs/rogers-ricci.md

# equations
# ndot = -(n u_e)' + (Sn=2*0.03)
# u_idot = -u_i u_i' - 1/n (n T_e)'
# u_edot = -u_e u_e' - tau ( T_e/n n' + 1.71 T_e' + n (u_i-u_e) nu )
# T_edot = - (2/3) 0.71 T_e/n (n(u_i-u_e))' - (2/3) T_e u_e' - u_e T_e' + (ST=2*0.03)  CARE FIRST RHS TERM SIGN CHANGED RELATIVE TO RR PAPER!!!!


from firedrake import *
import math
from irksome import Dt, GaussLegendre, MeshConstant, TimeStepper

# choose either set up from nothing or load from file (latter is selected)

meshres = 1000
#mesh = IntervalMesh(meshres, -18, 18, name="RR1D_mesh")
#V1 = FunctionSpace(mesh, "DG", 1)
#V2 = VectorFunctionSpace(mesh, "CG", 1)  # IMPORTANT velocity space needs to be continuous # electrons
#V2a = VectorFunctionSpace(mesh, "CG", 1) # ions
#V3 = FunctionSpace(mesh, "DG", 1)  # T
#V = V1*V2*V2a*V3

# load data
with CheckpointFile("RR_LAPD_1D_T_par_current_tau64.h5", 'r') as infile:
     mesh  = infile.load_mesh("RR1D_mesh")
     V1 = FunctionSpace(mesh, "DG", 1)
     V2 = VectorFunctionSpace(mesh, "CG", 1)  # IMPORTANT velocity space needs to be continuous # electrons
     V2a = VectorFunctionSpace(mesh, "CG", 1) # ions
     V3 = FunctionSpace(mesh, "DG", 1)  # T
     V = V1*V2*V2a*V3
     nuyT  = infile.load_function(mesh, "nuyT")

# time parameters
T0 = 20.0
timeres = 2000
t = Constant(0.0)
dt = Constant(T0/timeres)
skip=20

# parameters for irksome
butcher_tableau = GaussLegendre(2)

# model parameters
nstar = Constant(2.0*0.03)
Temp = 1.0
tau = 128.0  # will be ion:electron mass ratio
nu_param = 0.03

x = SpatialCoordinate(mesh)
#nuyT = Function(V, name="nuyT")
n, u, y, T = split(nuyT)
#n, u = nu.split()  # doesn't work with Irksome
v1, v2, v2a, v3 = TestFunctions(V)

# Gaussian blob init data or flat init data
#amp = 0.01  # harder to get successful convergence if this is increased ...
#width = 0.1*18
#nuyT.sub(0).interpolate((0.5+amp*(1/sqrt(2*math.pi*width**2))*exp(-x[0]**2/(2*width**2))))
#nuyT.sub(1).interpolate(as_vector([0.0+1.0*x[0]/18.0]))
#nuyT.sub(2).interpolate(as_vector([0.0+1.0*x[0]/18.0]))
#nuyT.sub(3).interpolate((0.5+0.0*amp*(1/sqrt(2*math.pi*width**2))*exp(-x[0]**2/(2*width**2))))

# source function
nstarFunc = Function(V)
nstarFunc.sub(0).interpolate(nstar + 0.0*x[0])

# TRIALCODE check init data
File("RR_LAPD_1D_T_par_current_init.pvd").write(nuyT.sub(0), nuyT.sub(1), nuyT.sub(2), nuyT.sub(3))
#quit()

norm = FacetNormal(mesh)
u_n = 0.5*(dot(u,norm)+abs(dot(u,norm)))
y_n = 0.5*(dot(y,norm)+abs(dot(y,norm)))

# outflow BC imposed weakly in here, penalty term in ds (note conditional not really needed here as both are outflow)

# integrated press grad term by parts (or its not stable) + seem to get away without a surface term from doing that (due to mixed DG/CG?) ...

heat_suppress = 1.0  # artificially slow down heat transport i.e. slow down tdot timescale, try e.g. 0.1

h = 18.0*2.0/meshres  # term \propto this is SU stabilization i.e. artificial visc for CG field

F = -Dt(n)*v1*dx + (n*dot(u, grad(v1))+nstar*v1)*dx \
   - (v1('+') - v1('-'))*(u_n('+')*n('+') - u_n('-')*n('-'))*dS \
   - conditional(dot(u, norm) > 0, v1*dot(u, norm)*n, 0.0)*ds \
    -n*dot(Dt(u), v2) *dx  - n*u[0]*v2[0]* grad(u[0])[0]*dx + tau*n*grad(T*v2[0])[0]*dx \
    +1.71*tau*n*grad(v2[0])[0]*T*dx \
    + nu_param*tau*n*n*(y[0]-u[0])*v2[0]*dx \
    -n*dot(Dt(y), v2a) *dx - n*y[0]*v2a[0]*grad(y[0])[0]*dx +     T*n*grad(v2a[0])[0]*dx \
    - (0.5*h*(grad(n*y[0])[0])*grad(v2a[0])[0])*dx \
    -dot(Dt(T), v3) *dx + heat_suppress*(nstar*v3)*dx \
    +(2.0/3.0)*heat_suppress*(T*dot(u, grad(v3)))*dx \
    -(1.0/3.0)*heat_suppress*(dot(u,grad(T))*v3)*dx \
   - (2.0/3.0)*heat_suppress*(v3('+') - v3('-'))*(u_n('+')*T('+') - u_n('-')*T('-'))*dS \
   - (2.0/3.0)*heat_suppress*conditional(dot(u, norm) > 0, v3*dot(u, norm)*T, 0.0)*ds \
    - 0.71*(2.0/3.0)*heat_suppress*(T*v3/n)*grad(n*(y[0]-u[0]))[0]*dx \

# these are integrated by parts version of last term with numerical flux:
#    -0.71*(2.0/3.0)*heat_suppress*grad(T*v3/n)[0]*n*(y[0]-u[0])*dx \
#   + 0.71*(2.0/3.0)*heat_suppress*(v3('+') - v3('-'))*(T('+')*(y_n('+')-u_n('+')) - T('-')*(y_n('-')-u_n('-')))*dS \
#   + 0.71*(2.0/3.0)*heat_suppress*conditional(dot(y-u, norm) > 0, v3*dot(y-u, norm)*T, 0.0)*ds \

# BEWARE - SIGN CHANGED IN GRAD OF PARALLEL CURRENT TERM IN TDOT EQ VS RR PAPER!

# params taken from Cahn-Hilliard example cited above
params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'l2',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

# Dirichlet BCs are needed for boundary velocity (ARE THEY?)

bc_test1 = DirichletBC(V.sub(1),as_vector([-1.0]),1)
bc_test2 = DirichletBC(V.sub(1),as_vector([1.0]),2)

# removed velocity BC.  Seems to work but not clear whether outflow is sonic.
stepper = TimeStepper(F, butcher_tableau, t, dt, nuyT, solver_parameters=params)
#stepper = TimeStepper(F, butcher_tableau, t, dt, nuyT, solver_parameters=params, bcs=[bc_test1, bc_test2])

outfile = File("RR_LAPD_1D_T_par_current.pvd")

nuyT.sub(0).rename("n")
nuyT.sub(1).rename("u")
nuyT.sub(2).rename("y")
nuyT.sub(3).rename("T")

cnt=0

while float(t) < float(T0):
    if (float(t) + float(dt)) >= T0:
        dt.assign(T0 - float(t))
    if (cnt%skip)==0:
       outfile.write(nuyT.sub(0), nuyT.sub(1), nuyT.sub(2), nuyT.sub(3))
       print("done output\n")
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(float(t), float(dt))
    cnt = cnt+1

print("done.")
print("\n")

ns, us, ys, Ts = nuyT.split()
ns.rename("density")
us.rename("electron velocity")
ys.rename("ion velocity")
Ts.rename("temperature")
File("RR_LAPD_1D_T_par_current_final.pvd").write(ns, us, ys, Ts)

# output data
with CheckpointFile("RR_LAPD_1D_T_par_current_tau128.h5", 'w') as outfile:
    outfile.save_mesh(mesh)
    outfile.save_function(nuyT)





