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

class MpcDrone:
    def __init__(self,vx, vy,vz, psi,wx,wy,wz,phi=None, phidot=None, theta=None, thetadot=None, x0=None, dt=0.01, n_horizon=20, r_weight=1e-2, run=True):
    
        # Set set-point time series
        self.vx = vx.copy()
        self.vy = vy.copy()
        self.vz = vz.copy()
        self.psi = psi.copy()
        self.psidot = pynumdiff.finite_difference.second_order(self.psi, dt)[1]
        self.wx = wx.copy()
        self.wy = wy.copy()
        self.wz = wz.copy()

        # Set initial state of non-set-point states
        # Initialize phi, phidot, theta, thetadot as arrays of zeros if they are not provided
        self.phi = np.array([phi]) if phi is not None else np.array([0.0])
        self.phidot = np.array([phidot]) if phidot is not None else np.array([0.0])
        self.theta = np.array([theta]) if theta is not None else np.array([0.0])
        self.thetadot = np.array([thetadot]) if thetadot is not None else np.array([0.0])

        # Set initial state
        m = 3.0     # mass
        g = 9.81    # gravity
        I_x = 1800   # moment of inertia
        I_y = 1800   # moment of inertia
        I_z = 1800   # moment of inertia
        J_x = 0.01  # polar moment of inertia
        l = 0.25     # arm length
        b = 0.1 # thrust factor
        d = 0.1 # drag factor

        self.x0 = { 'vx'       : self.vx[0], 
                    'vy'       : self.vy[0],
                    'vz'       : self.vz[0],
                    'phi'      : self.phi[0],
                    'phidot'   : self.phidot[0],
                    'theta'    : self.theta[0],
                    'thetadot' : self.thetadot[0],
                    'psi'      : self.psi[0],
                    'psidot'   : self.psidot[0],
                    'wx'       : self.wx[0], 
                    'wy'       : self.wy[0], 
                    'wz'       : self.wz[0]}
 
        # Overwrite specified initial states
        if x0 is not None:
            SetDict().set_dict_with_overwrite(self.x0, x0)

        # Store MPC parameters
        self.dt = np.round(dt, 5)
        self.fs = 1 / self.dt
        self.n_horizon = n_horizon

        # Get total # of points & simulation time
        self.n_points = np.squeeze(self.vx).shape[0]
        self.T = (self.n_points - 1) * self.dt
        # self.tsim = np.arange(0.0, self.T + self.dt/2, self.dt)
        self.tsim = self.dt * (np.linspace(1.0, self.n_points, self.n_points) - 1)
        self.xsim = np.zeros_like(self.tsim)
        self.usim = np.zeros_like(self.tsim)
        # define the MPC model
        self.model = do_mpc.model.Model('continuous')

        # Define the state variables for the MPC model
        vx       = self.model.set_variable(var_type='_x', var_name='vx')
        vy       = self.model.set_variable(var_type='_x', var_name='vy')
        vz       = self.model.set_variable(var_type='_x', var_name='vz')
        phi      = self.model.set_variable(var_type='_x', var_name='phi')
        phidot   = self.model.set_variable(var_type='_x', var_name='phidot')
        theta    = self.model.set_variable(var_type='_x', var_name='theta')
        thetadot = self.model.set_variable(var_type='_x', var_name='thetadot')
        psi      = self.model.set_variable(var_type='_x', var_name='psi')
        psidot   = self.model.set_variable(var_type='_x', var_name='psidot')
        wx       = self.model.set_variable(var_type='_x', var_name='wx')
        wy       = self.model.set_variable(var_type='_x', var_name='wy')
        wz       = self.model.set_variable(var_type='_x', var_name='wz')

        # define set-point variables for the MPC model
        vx_setpoint   = self.model.set_variable(var_type='_tvp', var_name='vx_setpoint')
        vy_setpoint   = self.model.set_variable(var_type='_tvp', var_name='vy_setpoint')
        vz_setpoint   = self.model.set_variable(var_type='_tvp', var_name='vz_setpoint')
        psi_setpoint  = self.model.set_variable(var_type='_tvp', var_name='psi_setpoint')
        wx_setpoint   = self.model.set_variable(var_type='_tvp', var_name='wx_setpoint')
        wy_setpoint   = self.model.set_variable(var_type='_tvp', var_name='wy_setpoint')
        wz_setpoint   = self.model.set_variable(var_type='_tvp', var_name='wz_setpoint')

        # define control variables for the MPC model
        u1  = self.model.set_variable(var_type='_u', var_name='u1')
        u2  = self.model.set_variable(var_type='_u', var_name='u2')
        u3  = self.model.set_variable(var_type='_u', var_name='u3')
        u4  = self.model.set_variable(var_type='_u', var_name='u4')
        uwx = self.model.set_variable(var_type='_u', var_name='uwx')
        uwy = self.model.set_variable(var_type='_u', var_name='uwy')
        uwz = self.model.set_variable(var_type='_u', var_name='uwz')

        print(self.model.x.keys())

        # define input dynamics
        U1 = b * (u1**2 + u2**2 + u3**2 + u4**2)
        U2 = b * (u4**2+u1**2 - u2**2-u3**2)
        U3 = b * (u3**2+u4**2 - u1**2-u2**2)
        U4 = d * (-u1**2 + u2**2 - u3**2 + u4**2)

        # Define state equations for the MPC model
        self.model.set_rhs('vx', (np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi))*U1/m)
        self.model.set_rhs('vy', (np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi))*U1/m)
        self.model.set_rhs('vz', np.cos(phi)*np.cos(theta)*U1/m - g)
        self.model.set_rhs('phi', phidot)
        self.model.set_rhs('phidot', U2*l/I_x)
        self.model.set_rhs('theta', thetadot)
        self.model.set_rhs('thetadot', U3*l/I_y)
        self.model.set_rhs('psi', psidot)
        self.model.set_rhs('psidot', U4/I_z)
        self.model.set_rhs('wx', uwx)
        self.model.set_rhs('wy', uwy)
        self.model.set_rhs('wz', uwz)

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

    def set_objective(self, r_weight=100): #r_weight=penalize control inputs
        """ Set MCP objective function.

            Inputs:
                r_weight: weight for control penalty
        """

        # lterm = (self.model.x['vx'] - self.model.tvp['vx_setpoint'])**2 + \
        #         (self.model.x['vy'] - self.model.tvp['vy_setpoint'])**2 + \
        #         (self.model.x['vz'] - self.model.tvp['vz_setpoint'])**2 + \
        #         (self.model.x['psi'] - self.model.tvp['psi_setpoint'])**2 + \
        #         (self.model.x['wx'] - self.model.tvp['wx_setpoint'])**2 + \
        #         (self.model.x['wy'] - self.model.tvp['wy_setpoint'])**2 + \
        #         (self.model.x['wz'] - self.model.tvp['wz_setpoint'])**2 

        lterm = (self.model.x['psi'] - self.model.tvp['psi_setpoint'])**2
        
        # Set terminal cost same as state cost
        mterm = lterm

        # Set objective
        self.mpc.set_objective(mterm=mterm, lterm=lterm)  # objective function
        self.mpc.set_rterm(u1=r_weight*(1e-10),u2=r_weight*(1e-10),u3=r_weight*(1e-10),u4=r_weight*(1e-10),uwx=r_weight,uwy=r_weight,uwz=r_weight)  # input penalty

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
            self.mpc_tvp_template['_tvp', n, 'vy_setpoint'] = self.vy[k_set]
            self.mpc_tvp_template['_tvp', n, 'vz_setpoint'] = self.vz[k_set]
            self.mpc_tvp_template['_tvp', n, 'psi_setpoint'] = self.psi[k_set]
            self.mpc_tvp_template['_tvp', n, 'wx_setpoint'] = self.wx[k_set]
            self.mpc_tvp_template['_tvp', n, 'wy_setpoint'] = self.wy[k_set]
            self.mpc_tvp_template['_tvp', n, 'wz_setpoint'] = self.wz[k_set]

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
        self.simulator_tvp_template['vy_setpoint']  = self.vy[k_step]
        self.simulator_tvp_template['vz_setpoint']  = self.vz[k_step]
        self.simulator_tvp_template['psi_setpoint'] = self.psi[k_step]
        self.simulator_tvp_template['wx_setpoint']  = self.wx[k_step]
        self.simulator_tvp_template['wy_setpoint']  = self.wy[k_step]
        self.simulator_tvp_template['wz_setpoint']  = self.wz[k_step]
        
        return self.simulator_tvp_template
            

    def run_mpc(self):
        #set initial state to match 1st point from set points
        x0=np.array([self.vx[0],
                    self.vy[0],
                    self.vz[0],
                    self.phi[0],
                    self.phidot[0],
                    self.theta[0],
                    self.thetadot[0],
                    self.psi[0],
                    self.psidot[0],
                    self.wx[0],
                    self.wy[0],
                    self.wz[0]]).reshape(-1,1)
        
        # Initial controls are 0
        u0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1,1)

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
            print(x_step)
            u_step = self.mpc.make_step(x_step)
            # u_step[0:4] = 10
            x_step = self.simulator.make_step(u_step)
            self.u_mpc.append(u_step)
            self.x_mpc.append(x_step)


        self.u_mpc = np.hstack(self.u_mpc).T
        self.x_mpc = np.hstack(self.x_mpc).T

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Customizing Matplotlib:
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True

        mpc_graphics = do_mpc.graphics.Graphics(self.mpc.data)
        sim_graphics = do_mpc.graphics.Graphics(self.simulator.data)

        # We just want to create the plot and not show it right now. This "inline magic" supresses the output.
        fig, ax = plt.subplots(4, sharex=True, figsize=(16,27))
        fig.align_ylabels()

        for g in [sim_graphics, mpc_graphics]:
            #plot the states
            # g.add_line(var_type='_x', var_name='vx', axis=ax[0])
            # g.add_line(var_type='_x', var_name='vy', axis=ax[0])
            g.add_line(var_type='_x', var_name='psi', axis=ax[2], color='black',alpha=0.5)
            g.add_line(var_type='_tvp', var_name='psi_setpoint', axis=ax[2], linestyle='--', color='black')
            # g.add_line(var_type='_x', var_name='psi', axis=ax[0])
            g.add_line(var_type='_x', var_name='wx', axis=ax[0], color='red',alpha=0.5)
            g.add_line(var_type='_tvp', var_name='wx_setpoint', axis=ax[0], linestyle='--', color='red')
            g.add_line(var_type='_x', var_name='wy', axis=ax[0], color='green',alpha=0.5)
            g.add_line(var_type='_tvp', var_name='wy_setpoint', axis=ax[0], linestyle='--', color='green')
            g.add_line(var_type='_x', var_name='wz', axis=ax[0], color='blue',alpha=0.5)
            g.add_line(var_type='_tvp', var_name='wz_setpoint', axis=ax[0], linestyle='--', color='blue')

            # plot the inputs
            g.add_line(var_type='_u', var_name='u1', axis=ax[3], color='black',linestyle='--')
            # g.add_line(var_type='_u', var_name='u2', axis=ax[3], color='black',linestyle='-.')
            # g.add_line(var_type='_u', var_name='u3', axis=ax[3], color='black',linestyle=':')
            # g.add_line(var_type='_u', var_name='u4', axis=ax[3], color='black',linestyle='-')
            g.add_line(var_type='_u', var_name='uwx', axis=ax[1], color='red')
            g.add_line(var_type='_u', var_name='uwy', axis=ax[1], color='green')
            g.add_line(var_type='_u', var_name='uwz', axis=ax[1], color='blue')

            

        ax[0].set_ylabel('States wind')
        ax[1].set_ylabel('Inputs')
        ax[2].set_ylabel('States dynamics')
        ax[3].set_ylabel('Inputs dynamics')
        ax[3].set_xlabel('Time [s]')


        sim_graphics.plot_results()
        # Reset the limits on all axes in graphic to show the data.
        sim_graphics.reset_axes()
        # Show the figure:
        fig



        return self.x_mpc, self.u_mpc
    
      


print('Hello from MPC_drone.py')