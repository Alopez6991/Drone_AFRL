import sys
import os

sys.path.append(os.path.join('/home/austin/Drone_AFRL/','util'))

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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import figurefirst as fifi
# sys.path.append('/home/austin/wind-observer/util/figure_functions.py')
import figure_functions as ff

from casadi import *
import do_mpc

from setdict import SetDict

class MpcDrone:
    def __init__(self,vx,vy,vz,params,psi=None,x=None,y=None,z=None,phi=None,theta=None,phidot=None,thetadot=None,psidot=None,x0=None,U_start=None,X_start=None,dt=0.01, n_horizon=20, r_weight=0.0, run=True):
    
        # Set set-point time series - seting the target values for velocity x, velocity y, velocity z, and psi(yaw/heading)
        self.vx = vx.copy()
        # print('vx:',self.vx)
        self.vx_target = vx.copy()
        self.vy = vy.copy()
        # print('vy:',self.vy)
        self.vy_target = vy.copy()
        self.vz = vz.copy()
        # print('vz:',self.vz)
        self.vz_target = vz.copy()
        self.psi = psi.copy()
        # print('psi:',self.psi)
        self.psi_target = psi.copy()
        self.psidot=pynumdiff.finite_difference.second_order(self.psi,dt)[1]

        # Set initial state of non-set-point states
        # Initialize phi, phidot, theta, thetadot as arrays of zeros if they are not provided
        self.x = np.array([x[0]]) if x is not None else np.array([0.0])
        self.y = np.array([y[0]]) if y is not None else np.array([0.0])
        self.z = np.array([z[0]]) if z is not None else np.array([0.0])
        self.phi = np.array([phi[0]]) if phi is not None else np.array([0.0])
        self.theta = np.array([theta[0]]) if theta is not None else np.array([0.0])
        self.phidot = np.array([phidot[0]]) if phidot is not None else np.array([0.0])
        self.thetadot = np.array([thetadot[0]]) if thetadot is not None else np.array([0.0])
        # self.psidot = np.array([psidot]) if psidot is not None else np.array([0.0])
        # self.psi = np.array([psi]) if psi is not None else np.array([0.0])
        self.x_start = X_start if X_start is not None else np.array([self.x[0],self.vx[0],self.y[0],self.vy[0],self.z[0],self.vz[0],self.phi[0],self.phidot[0],self.theta[0],self.thetadot[0],self.psi[0],self.psidot[0]])
        


        # Set initial setup parameters
        self.params=params

        m  = self.params[0]    # mass
        l  = self.params[1]    # length
        Ix = self.params[2]    # moment of inertia about x
        Iy = self.params[3]    # moment of inertia about y
        Iz = self.params[4]    # moment of inertia about z
        Jr = self.params[5]    # rotor inertia
        g  = 9.81              # gravity
        b  = self.params[6]    # thrust constant
        d  = self.params[7]    # drag constant
        Dl = self.params[8]    # drag from ground speed
        Dr = self.params[9]    # drag from rotation
        self.ui=np.sqrt((m*g)/(4*b))/10 #initial input
        self.U_start = np.array([U_start]) if U_start is not None else np.array([self.ui,self.ui,self.ui,self.ui])


        self.x0 = { 'x'       : self.x[0], 
                    'vx'       : self.vx[0],
                    'y'       : self.y[0],
                    'vy'       : self.vy[0],
                    'z'       : self.z[0],
                    'vz'       : self.vz[0],
                    'phi'       : self.phi[0],
                    'phidot'       : self.phidot[0],
                    'theta'       : self.theta[0],
                    'thetadot'       : self.thetadot[0],
                    'psi'       : self.psi[0],
                    'psidot'       : self.psidot[0]}
 
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
        x = self.model.set_variable(var_type='_x', var_name='x')
        vx = self.model.set_variable(var_type='_x', var_name='vx')
        y = self.model.set_variable(var_type='_x', var_name='y')
        vy = self.model.set_variable(var_type='_x', var_name='vy')
        z = self.model.set_variable(var_type='_x', var_name='z')
        vz = self.model.set_variable(var_type='_x', var_name='vz')
        phi = self.model.set_variable(var_type='_x', var_name='phi')
        phidot = self.model.set_variable(var_type='_x', var_name='phidot')
        theta = self.model.set_variable(var_type='_x', var_name='theta')
        thetadot = self.model.set_variable(var_type='_x', var_name='thetadot')
        psi = self.model.set_variable(var_type='_x', var_name='psi')
        psidot = self.model.set_variable(var_type='_x', var_name='psidot')


        # define set-point variables for the MPC model
        vx_setpoint   = self.model.set_variable(var_type='_tvp', var_name='vx_setpoint')
        vy_setpoint   = self.model.set_variable(var_type='_tvp', var_name='vy_setpoint')
        vz_setpoint   = self.model.set_variable(var_type='_tvp', var_name='vz_setpoint')
        psi_setpoint   = self.model.set_variable(var_type='_tvp', var_name='psi_setpoint')

        # define control variables for the MPC model
        u1  = self.model.set_variable(var_type='_u', var_name='u1')
        u2  = self.model.set_variable(var_type='_u', var_name='u2')
        u3  = self.model.set_variable(var_type='_u', var_name='u3')
        u4  = self.model.set_variable(var_type='_u', var_name='u4')

        # define input dynamics
        U1 = b * (u1**2 + u2**2 + u3**2 + u4**2)
        U2 = b * (u4**2+u1**2 - u2**2-u3**2)
        U3 = b * (u3**2+u4**2 - u1**2-u2**2)
        U4 = d * (-u1**2 + u2**2 - u3**2 + u4**2)
        omega=u2+u4-u1-u3

        # Define state equations for the MPC model
        self.model.set_rhs('x', vx) 
        self.model.set_rhs('vx', (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))*U1/m - Dl*vx/m)
        self.model.set_rhs('y', vy)
        self.model.set_rhs('vy', (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))*U1/m - Dl*vy/m)
        self.model.set_rhs('z', vz)
        self.model.set_rhs('vz', (cos(phi)*cos(theta))*U1/m - Dl*vz/m - g)
        self.model.set_rhs('phi', phidot)
        self.model.set_rhs('phidot', thetadot*psidot*(Iy-Iz)/(Ix)-Jr*thetadot*omega/Ix+U2*l/Ix - phidot*Dr/Ix)
        self.model.set_rhs('theta', thetadot)
        self.model.set_rhs('thetadot', phidot*psidot*(Iz-Ix)/(Iy)+Jr*phidot*omega/Iy+U3*l/Iy - thetadot*Dr/Iy)
        self.model.set_rhs('psi', psidot)
        self.model.set_rhs('psidot', phidot*thetadot*(Ix-Iy)/(Iz)+U4/Iz - psidot*Dr/Iz)

        # self.model.set_rhs('x', vx) 
        # self.model.set_rhs('vx', (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))*U1/m )
        # self.model.set_rhs('y', vy)
        # self.model.set_rhs('vy', (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))*U1/m )
        # self.model.set_rhs('z', vz)
        # self.model.set_rhs('vz', (cos(phi)*cos(theta))*U1/m - g)
        # self.model.set_rhs('phi', phidot)
        # self.model.set_rhs('phidot', thetadot*psidot*(Iy-Iz)/(Ix)-Jr*thetadot*omega/Ix+U2*l/Ix )
        # self.model.set_rhs('theta', thetadot)
        # self.model.set_rhs('thetadot', phidot*psidot*(Iz-Ix)/(Iy)+Jr*phidot*omega/Iy+U3*l/Iy )
        # self.model.set_rhs('psi', psidot)
        # self.model.set_rhs('psidot', phidot*thetadot*(Ix-Iy)/(Iz)+U4/Iz )

        # self.model.set_rhs('x', vx) 
        # self.model.set_rhs('vx', (1*theta*cos(psi) + phi*sin(psi))*U1/m - Dl*vx/m)
        # self.model.set_rhs('y', vy)
        # self.model.set_rhs('vy', (1*theta*sin(psi) - phi*cos(psi))*U1/m - Dl*vy/m)
        # self.model.set_rhs('z', vz)
        # self.model.set_rhs('vz', (1*1)*U1/m - g - Dl*vz/m)
        # self.model.set_rhs('phi', phidot)
        # self.model.set_rhs('phidot', thetadot*psidot*(Iy-Iz)/(Ix)-Jr*thetadot*omega/Ix+U2*l/Ix - phidot*Dr/Ix)
        # self.model.set_rhs('theta', thetadot)
        # self.model.set_rhs('thetadot', phidot*psidot*(Iz-Ix)/(Iy)+Jr*phidot*omega/Iy+U3*l/Iy - thetadot*Dr/Iy)
        # self.model.set_rhs('psi', psidot)
        # self.model.set_rhs('psidot', phidot*thetadot*(Ix-Iy)/(Iz)+U4/Iz - psidot*Dr/Iz)


        # Build the MPC model
        self.model.setup()
        self.mpc=do_mpc.controller.MPC(self.model)

        # Set lowwer bounds on state variables
        ANG=45
        self.mpc.bounds['lower','_x', 'z'] = 0
        self.mpc.bounds['lower','_x', 'phi'] = -ANG*2*np.pi/180
        self.mpc.bounds['lower','_x', 'theta'] = -ANG*2*np.pi/180
        self.mpc.bounds['lower','_x', 'psi'] = -np.pi

        # Set upper bounds on state variables
        self.mpc.bounds['upper','_x', 'phi'] = ANG*2*np.pi/180
        self.mpc.bounds['upper','_x', 'theta'] = ANG*2*np.pi/180
        self.mpc.bounds['upper','_x', 'psi'] = np.pi

        # Set bounds on control inputs
        self.mpc.bounds['lower','_u', 'u1'] = 0
        self.mpc.bounds['lower','_u', 'u2'] = 0
        self.mpc.bounds['lower','_u', 'u3'] = 0
        self.mpc.bounds['lower','_u', 'u4'] = 0


        
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

        lterm = (self.model.x['vx'] - self.model.tvp['vx_setpoint'])**2 + \
                (self.model.x['vy'] - self.model.tvp['vy_setpoint'])**2 + \
                (self.model.x['vz'] - self.model.tvp['vz_setpoint'])**2 + \
                (self.model.x['psi'] - self.model.tvp['psi_setpoint'])**2
        

        
        # Set terminal cost same as state cost
        mterm = lterm

        # Set objective
        self.mpc.set_objective(mterm=mterm, lterm=lterm)  # objective function
        self.mpc.set_rterm(u1=r_weight,u2=r_weight,u3=r_weight,u4=r_weight)  # input penalty

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
        self.simulator_tvp_template['psi_setpoint']  = self.psi[k_step]
        
        return self.simulator_tvp_template
            

    def run_mpc(self):
        #set initial state to match 1st point from set points
        # x0=np.array([self.x[0],
        #             self.vx[0],
        #             self.y[0],
        #             self.vy[0],
        #             self.z[0],
        #             self.vz[0],
        #             self.phi[0],
        #             self.phidot[0],
        #             self.theta[0],
        #             self.thetadot[0],
        #             self.psi[0],
        #             self.psidot[0]]).reshape(-1,1)

        x0 = np.array(self.x_start).reshape(-1,1)
        
        # Initial controls are 0
        u0 = self.U_start.reshape(-1,1)

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
        x_step = np.full(len(x0),0.0)
        for k in range(self.n_points - 1):
            u_step = self.mpc.make_step(x_step)
            # u_step[:]=1
            x_step = self.simulator.make_step(u_step)
            self.u_mpc.append(u_step)
            self.x_mpc.append(x_step)


        self.u_mpc = np.hstack(self.u_mpc).T
        self.x_mpc = np.hstack(self.x_mpc).T
        return self.x_mpc, self.u_mpc
    
    def get_X_and_U(self):
        return (self.x_mpc, self.u_mpc)
    
    def plot_states_targets_inputs(self):

        # make time vector based on dt and the length of the simulation
        self.t_mpc = np.arange(0, self.T + self.dt/2, self.dt)
        w=3

        plt.rcParams['axes.formatter.useoffset'] = False
        plt.plot(self.t_mpc,np.array(self.x_mpc .T[0]),label='x',color='blue',alpha=0.5 ,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('x_position')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[1]),label='vx',color='blue',alpha=0.5,linewidth=w)
        plt.plot(self.t_mpc,self.vx_target,label='vx target',color='blue',alpha=1,linestyle='dashed',linewidth=w)
        plt.xlabel('time')
        plt.ylabel('x_velocity')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[2]),label='y',color='red',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('y_position')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[3]),label='vy',color='red',alpha=0.5,linewidth=w)
        plt.plot(self.t_mpc,self.vy_target,label='vy target',color='red',alpha=1,linestyle='dashed',linewidth=w)
        plt.xlabel('time')
        plt.ylabel('y_velocity')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[4]),label='z',color='green',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('z_position')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[5]),label='vz',color='green',alpha=0.5,linewidth=w)
        plt.plot(self.t_mpc,self.vz_target,label='vz target',color='green',alpha=1,linestyle='dashed',linewidth=w)
        plt.xlabel('time')
        plt.ylabel('z_velocity')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[6]),label='phi',color='purple',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('phi')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[7]),label='phidot',color='purple',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('phidot')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[8]),label='theta',color='orange',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('theta')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[9]),label='thetadot',color='orange',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('thetadot')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[10]),label='psi',color='brown',alpha=0.5,linewidth=w)
        plt.plot(self.t_mpc,self.psi_target,label='psi target',color='brown',alpha=1,linestyle='dashed',linewidth=w)
        plt.xlabel('time')
        plt.ylabel('psi')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.x_mpc.T[11]),label='psidot',color='brown',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('psidot')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.u_mpc.T[0]),label='u1',color='blue',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('u1')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.u_mpc.T[1]),label='u2',color='red',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('u2')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.u_mpc.T[2]),label='u3',color='green',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('u3')
        plt.legend()

        plt.figure()
        plt.plot(self.t_mpc,np.array(self.u_mpc.T[3]),label='u4',color='purple',alpha=0.5,linewidth=w)
        plt.xlabel('time')
        plt.ylabel('u4')
        plt.legend()

        plt.figure()
        plt.plot(np.array(self.x_mpc .T[0]),np.array(self.x_mpc .T[2]),color='black')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.figure()

        # Assuming xt, yt, zt as the coordinates for the 3D dotted line
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        x = np.array(self.x_mpc .T[0])
        y = np.array(self.x_mpc .T[2])
        z = np.array(self.x_mpc .T[4])

        # Calculate dx, dy based on yaw (assuming yaw is in radians)
        dx =np.array(self.x_mpc .T[1]) 
        dy =np.array(self.x_mpc .T[2]) 
        dz =np.array(self.x_mpc .T[3])

        # Assuming time is another array of the same length
        time = self.t_mpc


        sc = ax.scatter(x, y, z, c=time)

       
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        cbar = plt.colorbar(sc)

        # Add a label to the colorbar
        cbar.set_label('Time')

        plt.show()
    
      


print('Hello from MPC_DRONE.py')