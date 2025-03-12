# RR_LAPD_1D_two_fluid_and_drag.py
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
# ndot = - (nu)' + (Sn = nstar)
# udot = - u u' - tau T n'/n - tau nu n u     ELECTRON VELOCITY
# vdot = - v v' - T n'/n     ION VELOCITY
# the steady state of this system has an analytic solution using a Lambert W function, see below (not as bad as it sounds!)

from firedrake import *
import math
from irksome import Dt, GaussLegendre, MeshConstant, TimeStepper

meshres = 100
mesh = IntervalMesh(meshres, -1, 1)
V1 = FunctionSpace(mesh, "DG", 1)
V2 = VectorFunctionSpace(mesh, "CG", 1)  # electron velocity IMPORTANT velocity space needs to be continuous
V3 = VectorFunctionSpace(mesh, "CG", 1)  # ion velocity
V = V1*V2*V3

# time parameters
T = 10.0
timeres = 200
t = Constant(0.0)
dt = Constant(T/timeres)
skip = 2

# parameters for irksome
butcher_tableau = GaussLegendre(1)

# model parameters
nstar = 1.0
Temp = 1.0
tau = 10.0  # will be ion:electron mass ratio
T0 = Temp  # temperature
nu = 0.1*1.6
bohm_fac = sqrt(tau*(1-nu*nstar/T0))  # electron outflow speed is \sqrt{T} times this factor TRIALCODE - MUST NOT BE GREATER THAN \sqrt{tau*Temp}


x = SpatialCoordinate(mesh)
nuv = Function(V)
n, u, v = split(nuv)
v1, v2, v3 = TestFunctions(V)

# Gaussian blob init data or flat init data
amp = 0.01  # harder to get successful convergence if this is increased ...
width = 0.1
nuv.sub(0).interpolate((0.010+amp*(1/sqrt(2*math.pi*width**2))*exp(-x[0]**2/(2*width**2))))
nuv.sub(1).interpolate(as_vector([0.0+sqrt(Temp)*x[0]*bohm_fac]))
nuv.sub(2).interpolate(as_vector([0.0+sqrt(Temp)*x[0]]))

# source function
nstarFunc = Function(V)
nstarFunc.sub(0).interpolate(nstar + 0.0*x[0])

# TRIALCODE check init data
#File("RR_LAPD_1D_two_fluid_and_drag_init.pvd").write(nuv.sub(0), nuv.sub(1), nuv.sub(2))
#quit()

norm = FacetNormal(mesh)
u_n = 0.5*(dot(u,norm)+abs(dot(u,norm)))

# outflow BC imposed weakly in here, penalty term in ds (note conditional not really needed here as both are outflow)

# integrated press grad term by parts (or its not stable) + seem to get away without a surface term from doing that (due to mixed DG/CG?) ...

F = -Dt(n)*v1*dx + (n*dot(u, grad(v1))+nstar*v1)*dx \
   - (v1('+') - v1('-'))*(u_n('+')*n('+') - u_n('-')*n('-'))*dS \
    -n*dot(Dt(u), v2) *dx - n*u[0]*v2[0]*grad(u[0])[0]*dx + tau*Temp*grad(v2[0])[0]*n*dx \
   - conditional(dot(u, norm) > 0, v1*dot(u, norm)*n, 0.0)*ds \
    -n*dot(Dt(v), v3) *dx - n*v[0]*v3[0]*grad(v[0])[0]*dx + Temp*grad(v3[0])[0]*n*dx \
   - conditional(dot(u, norm) > 0, v2[0]*dot(u, norm)*u[0]*n, 0.0)*ds \
   - conditional(dot(v, norm) > 0, v3[0]*dot(v, norm)*v[0]*n, 0.0)*ds \
   - tau*nu*n*n*dot(u,v2)*dx \


# last terms above: weak outflow BCs

# params taken from Cahn-Hilliard example cited above
params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'l2',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

# Dirichlet BCs can be used for the for boundary velocity 
bc_u1 = DirichletBC(V.sub(1),as_vector([-sqrt(Temp)*bohm_fac]),1)
bc_u2 = DirichletBC(V.sub(1),as_vector([sqrt(Temp)*bohm_fac]),2)
bc_v1 = DirichletBC(V.sub(2),as_vector([-sqrt(Temp)]),1)
bc_v2 = DirichletBC(V.sub(2),as_vector([sqrt(Temp)]),2)

# IMPORTANT NOTE FOR THIS OUTFLOW: DIRICHLET BCs NEEDED OR SONIC ELECTRON OUTFLOW SPOILS ION NUMERICS
# ALPHA IN ANALYTIC WOULD BE DIFFERENT SO THAT ANALYTIC SOLUTION WOULD HAVE DISCONTINUITY IN ION SPEED AT ORIGIN
#stepper = TimeStepper(F, butcher_tableau, t, dt, nuv, solver_parameters=params)  # weak BCs only (won't work - see above)
stepper = TimeStepper(F, butcher_tableau, t, dt, nuv, solver_parameters=params, bcs=[bc_u1, bc_u2, bc_v1, bc_v2])

outfile = File("RR_LAPD_1D_two_fluid_and_drag.pvd")

nuv.sub(0).rename("n")
nuv.sub(1).rename("u")
nuv.sub(2).rename("v")

# analytic solution

from mpmath import *

bohm_fac2 = bohm_fac  #TRIALCODE

alpha_u = exp(-tau*nu*nstar/(2.0*tau*Temp))*sqrt(Temp)*bohm_fac2*exp(-0.5*bohm_fac2*bohm_fac2*Temp*Temp/(tau*Temp))/(nstar)  # constant of integration, fixed by source amp and sonic outflow BCs
alpha_v = sqrt(Temp)*bohm_fac2*exp(-0.5)/(nstar)  # constant of integration, fixed by source amp and sonic outflow BCs

# TRIALCODE alpha_u and alpha_v have to be equal for ion velocity to be continuous (zero) at origin x=0:
print(str("alpha_u=")+str(alpha_u))
print(str("alpha_v=")+str(alpha_v))
test_val1 = (exp(-0.5*bohm_fac2*bohm_fac2*Temp*Temp/(tau*Temp))*exp(-tau*nu*nstar/(2.0*tau*Temp)))
print(str("test_val1=")+str(test_val1))
test_val2 = (exp(-0.5))
print(str("test_val2=")+str(test_val2))
test_val3 = math.log(2.0)
print(str("math.log(2)=")+str(test_val3))

n_analytic = Function(V1)
u_analytic = Function(V2)
v_analytic = Function(V3)

for i in range(0,(meshres+1)):
   xc = mesh.coordinates.dat.data[i]
   if mesh.coordinates.dat.data[i] < 0.0:
      u_analytic.dat.data[i] = -sqrt(-tau*T0*lambertw(-exp(xc*xc*tau*nu*nstar/(tau*Temp))*mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha_u*alpha_u*(nstar*nstar)/(tau*T0),0)).real
      tempval = -sqrt(-tau*T0*lambertw(-exp(xc*xc*tau*nu*nstar/(tau*Temp))*mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha_u*alpha_u*(nstar*nstar)/(tau*T0),0)).real
      #tempval = -sqrt(-tau*T0*(-exp(tau*nu*2*0.03/(tau*Temp))*mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha_u*alpha_u*(2*0.03*2*0.03)/(tau*T0)))  # SHOWS CORRECT
      #v_analytic.dat.data[i] = tempval  # TRIALCODE
      #v_analytic.dat.data[i] = -math.log(tempval/(alpha_v*2*0.03*mesh.coordinates.dat.data[i]))  #TRIALCODE
      #v_analytic.dat.data[i] = -(tempval/(alpha_v*2*0.03*mesh.coordinates.dat.data[i]))  #TRIALCODE
      v_analytic.dat.data[i] = -math.sqrt(math.fabs(2*Temp*math.log(tempval/(alpha_v*nstar*mesh.coordinates.dat.data[i]))))
      # there might be more solutions (but not physical) due to multivalued nature of the Lambert function:
      #u_analytic.dat.data[i] = -sqrt(-tau*T0*lambertw(-mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha*alpha*(2*0.03*2*0.03)/(tau*T0),-1)).real
   else:
      u_analytic.dat.data[i] =  sqrt(-tau*T0*lambertw(-exp(xc*xc*tau*nu*nstar/(tau*Temp))*mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha_u*alpha_u*(nstar*nstar)/(tau*T0),0)).real
      tempval = sqrt(-tau*T0*lambertw(-exp(xc*xc*tau*nu*nstar/(tau*Temp))*mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha_u*alpha_u*(nstar*nstar)/(tau*T0),0)).real
      #tempval = sqrt(-tau*T0*(-exp(tau*nu*2*0.03/(tau*Temp))*mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha_u*alpha_u*(2*0.03*2*0.03)/(tau*T0)))  #SHOWS CORRECT
      #v_analytic.dat.data[i] = math.log(tempval/(alpha_v*2*0.03*mesh.coordinates.dat.data[i]))  # TRIALCODE
      #v_analytic.dat.data[i] = (tempval/(alpha_v*2*0.03*mesh.coordinates.dat.data[i]))  # TRIALCODE
      v_analytic.dat.data[i] = math.sqrt(math.fabs(2*Temp*math.log(tempval/(alpha_v*nstar*mesh.coordinates.dat.data[i]))))
      # there might be more solutions (but not physical) due to multivalued nature of the Lambert function:
      #u_analytic.dat.data[i] =  sqrt(-tau*T0*lambertw(-mesh.coordinates.dat.data[i]*mesh.coordinates.dat.data[i]*alpha*alpha*(2*0.03*2*0.03)/(tau*T0),-1)).real

n_analytic.interpolate(nstar*x[0]/u_analytic[0])  # done like this because DG space has additional data points not in 1:1 corresp with mesh points


n_analytic.rename('n_analytic')
u_analytic.rename('u_analytic')
v_analytic.rename('v_analytic')

# end analytic solution

cnt=0

while float(t) < float(T):
    if (float(t) + float(dt)) >= T:
        dt.assign(T - float(t))
    if (cnt%skip)==0:
       outfile.write(nuv.sub(0), nuv.sub(1), nuv.sub(2), n_analytic, u_analytic, v_analytic)
       print("done output\n")
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(float(t), float(dt))
    cnt = cnt+1

print("done.")
print("\n")

ns, us, vs = nuv.split()
ns.rename("density")
us.rename("electron velocity")
vs.rename("ion velocity")
File("RR_LAPD_1D_two_fluid_and_drag_final.pvd").write(ns, us, vs, n_analytic, u_analytic, v_analytic)
