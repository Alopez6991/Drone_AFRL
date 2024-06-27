import sys
import os

sys.path.append(os.path.join('/home/austin/Nonlinear_and_Data_Driven_Estimation/', 'Drone', 'util'))

sys.path.append(os.path.join('/home/austin', 'wind-observer', 'util'))

# import numpy as np
# from numpy import matlib
import pandas as pd
import pynumdiff
# import scipy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
import figurefirst as fifi
# sys.path.append('/home/austin/wind-observer/util/figure_functions.py')
import figure_functions as ff

from casadi import *
import do_mpc

from setdict import SetDict

class Mpc_Smd:
    def __init__(self,wx,x=None,x0=None, dt=0.01, n_horizon=20, r_weight=1e-10, run=True):
    
        # Set set-point time series
        self.wx = vx.copy()
        self.vx_target = vx.copy()

        # Set initial state of non-set-point states
        # Initialize phi, phidot, theta, thetadot as arrays of zeros if they are not provided
        self.x = np.array([x]) if x is not None else np.array([0.0])


        # Set initial state
        m = 5.0     # mass
        k = 1.0     # spring constant
        b = 1.0     # damping constant
        C = 1.0     # control input constant


        self.x0 = { 'vx'       : self.wx[0], 
                    'x'       : self.x[0]}
 
        # Overwrite specified initial states
        if x0 is not None:
            SetDict().set_dict_with_overwrite(self.x0, x0)

        # Store MPC parameters
        self.dt = np.round(dt, 5)
        self.fs = 1 / self.dt
        self.n_horizon = n_horizon

        # Get total # of points & simulation time
        self.n_points = np.squeeze(self.wx).shape[0]
        self.T = (self.n_points - 1) * self.dt
        # self.tsim = np.arange(0.0, self.T + self.dt/2, self.dt)
        self.tsim = self.dt * (np.linspace(1.0, self.n_points, self.n_points) - 1)
        self.xsim = np.zeros_like(self.tsim)
        self.usim = np.zeros_like(self.tsim)
        # define the MPC model
        self.model = do_mpc.model.Model('continuous')

        # Define the state variables for the MPC model
        wx       = self.model.set_variable(var_type='_x', var_name='wx')
        # x       = self.model.set_variable(var_type='_x', var_name='x')

        # define set-point variables for the MPC model
        vx_setpoint   = self.model.set_variable(var_type='_tvp', var_name='vx_setpoint')

        # define control variables for the MPC model
        uwx  = self.model.set_variable(var_type='_u', var_name='uwx')

        # define input dynamics
        U1 = C * uwx

        # Define state equations for the MPC model
        self.model.set_rhs('wx', uwx)
        # self.model.set_rhs('x', vx)

        # Build the MPC model
        self.model.setup()
        self.mpc=do_mpc.controller.MPC(self.model)

        # Set estimator & simulator
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.simulator = do_mpc.simulator.Simulator(self.model)

        params_simulator = {
            # Note: cvode doesn't support DAE systems.
            'integration_tool': 'idas',  # cvodes, idas
            'abstol': 1e-8,
            'reltol': 1e-8,
            't_step': self.dt
        }

        self.simulator.set_param(**params_simulator)

        # Set MPC parameters
        setup_mpc = {
            'n_horizon': self.n_horizon,
            'n_robust': 0,
            'open_loop': 0,
            't_step': self.dt,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 1,
            'store_full_solution': True,

            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps',  # mumps, MA27
                            'ipopt.print_level': 0,
                            'ipopt.sb': 'yes',
                            'print_time': 0,
                            }
        }

        self.mpc.set_param(**setup_mpc)

        # Set MPC objective function
        self.set_objective(r_weight=r_weight)

        # Get template's for MPC time-varying parameters
        self.mpc_tvp_template = self.mpc.get_tvp_template()
        self.simulator_tvp_template = self.simulator.get_tvp_template()

        # Set time-varying set-point functions
        self.mpc.set_tvp_fun(self.mpc_tvp_function)
        self.simulator.set_tvp_fun(self.simulator_tvp_function)

        # Setup MPC & simulator
        self.mpc.setup()
        self.simulator.setup()

        # Set variables to store MPC simulation data
        self.x_mpc = np.array([0.0, 0.0])
        self.u_mpc = np.array([0.0, 0.0])

        if run:
            # Run MPC
            self.run_mpc()

            # # Replay
            # self.replay()

    def set_objective(self, r_weight=0.0): #r_weight=penalize control inputs
        """ Set MCP objective function.

            Inputs:
                r_weight: weight for control penalty
        """

        lterm = (self.model.x['vx'] - self.model.tvp['vx_setpoint'])**2
        #         (self.model.x['vy'] - self.model.tvp['vy_setpoint'])**2 + \
        #         (self.model.x['vz'] - self.model.tvp['vz_setpoint'])**2 + \
        #         (self.model.x['psi'] - self.model.tvp['psi_setpoint'])**2 + \
        #         (self.model.x['wx'] - self.model.tvp['wx_setpoint'])**2 + \
        #         (self.model.x['wy'] - self.model.tvp['wy_setpoint'])**2 + \
        #         (self.model.x['wz'] - self.model.tvp['wz_setpoint'])**2 

        
        # Set terminal cost same as state cost
        mterm = lterm

        # Set objective
        self.mpc.set_objective(mterm=mterm, lterm=lterm)  # objective function
        self.mpc.set_rterm(u1=r_weight)  # input penalty

    def mpc_tvp_function(self, t): # important
        """ Set the set-point function for MPC optimizer.

        Inputs:
            t: current time
        """

        # Set current step index
        k_step = int(np.round(t / self.dt))

        # Update set-point time horizon
        for n in range(self.n_horizon + 1):
            k_set = k_step + n
            if k_set >= self.n_points:  # horizon is beyond end of input data
                k_set = self.n_points - 1  # set part of horizon beyond input data to last point

            # Update each set-point over time horizon
            self.mpc_tvp_template['_tvp', n, 'vx_setpoint'] = self.vx[k_set]

        return self.mpc_tvp_template
        
    def simulator_tvp_function(self, t): # might do nothing
        """ Set the set-point function for MPC simulator.

            Inputs:
                t: current time
        """

        # Set current step index
        k_step = int(np.round(t / self.dt))
        if k_step >= self.n_points:  # point is beyond end of input data
            k_step = self.n_points - 1  # set point beyond input data to last point

        # Update current set-point
        self.simulator_tvp_template['vx_setpoint']  = self.vx[k_step]
        
        return self.simulator_tvp_template
            

    def run_mpc(self):
        #set initial state to match 1st point from set points
        x0=np.array([self.vx[0],
                    self.x[0]]).reshape(-1,1)
        
        # Initial controls are 0
        u0 = np.array([0.0]).reshape(-1,1)

        # Set initial MPC time, state, inputs
        self.mpc.t0 = np.array(0.0)
        self.mpc.x0 = x0
        self.mpc.u0 = u0
        self.mpc.set_initial_guess()

        # Set simulator MPC time, state, inputs
        self.simulator.t0 = np.array(0.0)
        self.simulator.x0 = x0
        self.simulator.set_initial_guess()

        # Initialize variables to store MPC data
        self.x_mpc = [x0]
        self.u_mpc = [u0]

        # Run simulation
        x_step = x0.copy()
        for k in range(self.n_points - 1):
            u_step = self.mpc.make_step(x_step)
            x_step = self.simulator.make_step(u_step)
            self.u_mpc.append(u_step)
            self.x_mpc.append(x_step)


        self.u_mpc = np.hstack(self.u_mpc).T
        self.x_mpc = np.hstack(self.x_mpc).T
        # make time vector based on dt and the length of the simulation
        self.t_mpc = np.arange(0, self.T + self.dt/2, self.dt)

        plt.rcParams['axes.formatter.useoffset'] = False
        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc .T[0]),label='vx',color='blue',alpha=0.5)
        plt.plot(self.t_mpc,self.vx_target,label='vx target',color='red',alpha=0.5,linestyle='dashed')
        plt.xlabel('time')
        plt.ylabel('Velocity')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[1]),label='x',color='blue',alpha=0.5)
        plt.xlabel('time')
        plt.ylabel('Position')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.u_mpc.T[0]),label='u',color='blue',alpha=0.5)
        plt.xlabel('time')
        plt.ylabel('Control Input')







        return self.x_mpc, self.u_mpc
    
      


print('Hello from MPC_drone_03.py')