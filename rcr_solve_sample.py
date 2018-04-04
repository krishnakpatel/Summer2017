# This sample script solves for the pressure in an RCR circuit. It has three 
# main functions:
#  1. boundaryConditions - Returns the value of the boundary conditions on the
#       selected LPN. Since this RCR has only one inlet, we return the value
#       of the pressure at the inlet node. We assume in this example that we
#       have a sinusoidal pressure for the inlet
#  2. sample_RCR_findF - Returns the ODEs that govern the physics of the selected
#       LPN. Since we have just one ODE to solve for (the pressure on the
#       capacitor), this simply returns the value of dP/dt on that capacitor.
#  3. rk4 - Contains the Runge-Kutta 4 algorithm for solving a system of
#       ordinary differential equations. This function should not need to be
#       modified.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import glob

n_steps = 3000
dt = 0.001
heart_period = 1

BC_FILE_LIST = []

# ==============================================================================

# Returns a list of boundary condition values at the specified time. In this
# example, we only have one boundary condition, so we simply return the
# sinusoidal pressure at the inlet of the RCR
def boundaryConditions(time):

  # Get the current time in the heart cycle
  tInCycle = time%heart_period
  #print(tInCycle)
  results = []
  
  for i_bc in BC_FILE_LIST:
    bc_file = open(i_bc, 'r')
    
    # Perform linear interpolation of the boundary pressure
    bc_last = 0.0
    t_last = 0.0
    for line in bc_file:
      line_split = line.split()
      t_check = float(line_split[0])
      if(t_check < tInCycle):
        bc_last = float(line_split[1])
        t_last = t_check
      if(np.abs(t_check - tInCycle) < 1e-8):
        bc_curr = float(line_split[1])
        results.append(bc_curr)
        break
      if(t_check > tInCycle):
        t_curr = float(line_split[0])
        bc_temp = float(line_split[1])
        interpolation_scale = (tInCycle - t_last) / (t_curr - t_last)
        bc_curr = bc_last + interpolation_scale*(bc_temp - bc_last)
        results.append(bc_curr)
        break
    
    bc_file.close()
  
  assert(len(results) == len(BC_FILE_LIST))
  return results

# ==============================================================================

# Returns the ODE's that govern the LPN circuit. For this case, we have just
# one ODE that solves for the pressure on the RCR, thus our output vector
# "dydt" has just one entry. Also included are explicit expressions for the
# flow across the resistors for clarity.
def sample_RCR_findF(time, solution_vars, boundary_conditions, params):

  # Define the parameter values
  R_prox = params[0]
  C = params[1]
  R_dist = params[2]
  
  dydt = []
  
  prox_flow = (boundary_conditions[0] - solution_vars[0])/R_prox
  dist_flow = solution_vars[0] / R_dist
  pressure_dt = (1.0/C) * (prox_flow - dist_flow)
  dydt.append(pressure_dt)
  
  return dydt
  
# ==============================================================================

# Python Runge-Kutta 4 method
#    dydt is a (system of) differential eqn(s) of the form dydt = f(t,y,params)
#    t0 is the starting time, h is the time step, n is the number of time steps
#    y0 is the initial condition (vector)
#    params are other parameters needed for the dydt function
def rk4(dydt,t0,h,n,y0,params):
  hh = 0.5*h
  # solution (vector) at inital time
  w = [y0]
  # time steps
  t = [t0]
  for i in xrange(0,n):
    if np.mod(i,100) == 0:
      print 'CurrentTime %d/%d ' % (i,n)

    # keep track of time steps
    t.append(t0 + (i+1)*h)
    
    # Find the inlet boundary pressure
    bc_vector = boundaryConditions(t[i])

    k1 = [h*dy for dy in dydt(t[i+1],w[i],bc_vector,params)]
			
    wtemp = [ww + 0.5*kk1 for ww,kk1 in zip(w[i],k1)]
			
    k2 = [h*dy for dy in dydt(t[i+1]+hh,wtemp,bc_vector,params)]

    wtemp = [ww + 0.5*kk2 for ww,kk2 in zip(w[i],k2)]
			
    k3 = [h*dy for dy in dydt(t[i+1]+hh,wtemp,bc_vector,params)]

    wtemp = [ww + kk3 for ww,kk3 in zip(w[i],k3)]
			
    k4 = [h*dy for dy in dydt(t[i+1]+h,wtemp,bc_vector,params)]
    w.append([ww + (1/6.)*(kk1+2.*kk2+2.*kk3+kk4) for ww,kk1,kk2,kk3,kk4 in zip(w[i],k1,k2,k3,k4)])
		
  return w,t
  
# ==============================================================================

# Main function.
if __name__ == '__main__':

  # Check to make sure the boundary condition file is present
  if(len(sys.argv) < 2):
    print('ERROR: Please specify the boundary condition file')
    exit()
  
  for i in xrange(1, len(sys.argv)):
    BC_FILE_LIST.append(sys.argv[i])
  
  y0 = [90] # Initial pressure on capacitor
  params = [10, 0.05, 100] # R_prox, C, R_dist
  solution, time = rk4(sample_RCR_findF, 0, dt, n_steps, y0, params)
  
  # Plot the solution for verification
  test_sol = np.zeros(n_steps+1)
  test_sol[0] = y0[0]
  for i in xrange(0, n_steps):
    test_sol[i+1] = solution[i+1][0]
  
  plt.plot(time, test_sol)
  plt.title('Pressure vs. time in RCR circuit')
  plt.xlabel('Time (s)')
  plt.ylabel('Pressure (mmHg)')
  plt.show()
  
# ==============================================================================
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
