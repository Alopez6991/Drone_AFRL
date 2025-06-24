import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import figurefirst as fifi
import figure_functions as ff
from pybounds import Simulator


class DroneModel:
    def __init__(self):
        self.params = []

        # State names
        self.state_names = ['x',  # x position in inertial frame [m]
                            'y',  # y position in inertial frame [m]
                            'z',  # elevation
                            'v_x',  # x velocity in global frame (parallel to heading) [m/s]
                            'v_y',  # y velocity in global frame (perpendicular to heading) [m/s]
                            'v_z',  # z velocity in global frame [m/s]
                            'psi',  # yaw in body-level frame (vehicle-1) [rad]
                            'w',  # wind speed in XY-plane [ms]
                            'zeta',  # wind direction in XY-plane[rad]
                            'k_x',  # x-velocity input motor calibration parameter
                            'k_y',  # y-velocity input motor calibration parameter
                            'k_psi',  # yaw angular velocity input motor calibration parameter
                            ]

        # Polar state names
        replacement_polar_states = {'v_x': 'g', 'v_y': 'beta'}
        self.state_names_polar = self.state_names.copy()
        self.state_names_polar = [replacement_polar_states.get(x, x) for x in self.state_names_polar]

        # Input names
        self.input_names = ['u_x',  # acceleration in parallel direction
                            'u_y',  # acceleration in perpendicular direction
                            'u_psi',  # yaw angular velocity [rad/s]
                            'u_z',  # change elevation
                            'u_w',  # change wind speed
                            'u_zeta',  # change wind direction
                            ]

        # Measurement names
        self.measurement_names = ['v_x_dot', 'v_y_dot',
                                  'r_x', 'r_y', 'r', 'g', 'beta',
                                  'a_x', 'a_y', 'a', 'gamma',
                                  'w_x', 'w_y']

        self.measurement_names = self.state_names + self.input_names + self.measurement_names

    def f(self, X, U):
        """ Dynamic model.
        """

        # States
        x, y, z, v_x, v_y, v_z, psi, w, zeta, k_x, k_y, k_psi = X

        # Inputs
        u_x, u_y, u_psi, u_z, u_w, u_zeta = U

        # Heading
        psi_dot = u_psi * k_psi

        # Velocity in global frame
        v_x_dot = u_x * k_x + psi_dot*v_y
        v_y_dot = u_y * k_y - psi_dot*v_x
        v_z_dot = u_z

        # Position in inertial frame
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        z_dot = v_z

        # Wind
        w_dot = u_w
        zeta_dot = u_zeta

        # Motor calibration parameters
        k_x_dot = k_x * 0.0
        k_y_dot = k_y * 0.0
        k_psi_dot = k_psi * 0.0

        # Package and return xdot
        x_dot = [x_dot, y_dot, z_dot,
                 v_x_dot, v_y_dot, v_z_dot,
                 psi_dot,
                 w_dot, zeta_dot,
                 k_x_dot, k_y_dot, k_psi_dot
                 ]

        return x_dot

    def h(self, X, U):
        """ Measurement model.
        """

        # States
        x, y, z, v_x, v_y, v_z, psi, w, zeta, k_x, k_y, k_psi = X

        # Inputs
        u_x, u_y, u_psi, u_z, u_w, u_zeta = U

        # Dynamics
        (x_dot, y_dot, z_dot, v_x_dot, v_y_dot, v_z_dot, psi_dot, w_dot, zeta_dot,
         k_x_dot, k_y_dot, k_psi_dot) = self.f(X, U)

        # Body-level velocity
        v_x_bl = v_x
        v_y_bl = v_y

        # Ground speed & course direction in body-level frame
        g = np.sqrt(v_x_bl ** 2 + v_y_bl ** 2)
        r_x = v_x_bl / z
        r_y = v_y_bl / z
        r = g / z
        beta = np.arctan2(v_y_bl, v_x_bl)

        # Apparent airflow
        a_x = v_x_bl - w * np.cos(psi - zeta)
        a_y = v_y_bl + w * np.sin(psi - zeta)
        a = np.sqrt(a_x ** 2 + a_y ** 2)
        gamma = np.arctan2(a_y, a_x)

        # Acceleration
        # v_dot = np.sqrt(v_x_dot ** 2 + v_y_dot ** 2)
        # alpha = np.arctan2(v_y, v_x)

        # Wind
        w_x = w * np.cos(zeta)
        w_y = w * np.sin(zeta)

        # Unwrap angles
        if np.array(psi).ndim > 0:
            if np.array(psi).shape[0] > 1:
                X[6] = np.unwrap(psi)
                beta = np.unwrap(beta)
                gamma = np.unwrap(gamma)

        Y = (list(X) + list(U) +
             [v_x_dot, v_y_dot, r_x, r_y, r, g, beta, a_x, a_y, a, gamma, w_x, w_y])

        return Y


class DroneSimulator(Simulator):
    def __init__(self, dt=0.1, mpc_horizon=10, r_u=1e-4, control_mode='velocity_global'):
        self.dynamics = DroneModel()
        super().__init__(self.dynamics.f, self.dynamics.h, dt=dt, mpc_horizon=mpc_horizon,
                         state_names=self.dynamics.state_names,
                         input_names=self.dynamics.input_names,
                         measurement_names=self.dynamics.measurement_names)

        # Define cost function
        self.control_mode = control_mode
        if self.control_mode == 'velocity_global':
            cost = (1.0 * (self.model.x['v_x'] - self.model.tvp['v_x_set']) ** 2 +
                    1.0 * (self.model.x['v_y'] - self.model.tvp['v_y_set']) ** 2 +
                    1.0 * (self.model.x['z'] - self.model.tvp['z_set']) ** 2 +
                    1.0 * (self.model.x['psi'] - self.model.tvp['psi_set']) ** 2 +
                    1.0 * (self.model.x['w'] - self.model.tvp['w_set']) ** 2 +
                    1.0 * (self.model.x['zeta'] - self.model.tvp['zeta_set']) ** 2)

        elif self.control_mode == 'position_global':
            cost = (1.0 * (self.model.x['x'] - self.model.tvp['x_set']) ** 2 +
                    1.0 * (self.model.x['y'] - self.model.tvp['y_set']) ** 2 +
                    1.0 * (self.model.x['z'] - self.model.tvp['z_set']) ** 2 +
                    1.0 * (self.model.x['psi'] - self.model.tvp['psi_set']) ** 2 +
                    1.0 * (self.model.x['w'] - self.model.tvp['w_set']) ** 2 +
                    1.0 * (self.model.x['zeta'] - self.model.tvp['zeta_set']) ** 2)

        elif self.control_mode == 'velocity_body_level':
            # Body-level velocity
            v_x_bl = self.model.x['v_x']
            v_y_bl = self.model.x['v_y']

            cost = (1.0 * (v_x_bl- self.model.tvp['v_x_set']) ** 2 +
                    1.0 * (v_y_bl - self.model.tvp['v_y_set']) ** 2 +
                    1.0 * (self.model.x['z'] - self.model.tvp['z_set']) ** 2 +
                    1.0 * (self.model.x['psi'] - self.model.tvp['psi_set']) ** 2 +
                    1.0 * (self.model.x['w'] - self.model.tvp['w_set']) ** 2 +
                    1.0 * (self.model.x['zeta'] - self.model.tvp['zeta_set']) ** 2)

        else:
            raise Exception('Control mode not available')

        # Set cost function
        self.mpc.set_objective(mterm=cost, lterm=cost)
        self.mpc.set_rterm(u_x=r_u, u_y=r_u, u_psi=r_u, u_z=r_u, u_w=r_u * 1e1, u_zeta=r_u * 1e1)

        # Place limit on states
        self.mpc.bounds['lower', '_x', 'z'] = 0

    def update_setpoint(self, x=None, y=None, v_x=None, v_y=None, psi=None, z=None, w=None, zeta=None,
                        k_x=None, k_y=None, k_psi=None):
        """ Set the set-point variables.
        """

        # Set time
        T = self.dt * (len(w) - 1)
        tsim = np.arange(0, T + self.dt / 2, step=self.dt)

        # Set control setpoints
        if self.control_mode == 'velocity_body_level':  # control the body-level x & y velocities
            if (v_x is None) or (v_y is None):  # must set velocities
                raise Exception('x or y velocity not set')
            else:  # x & y don't matter, set to 0
                x = 0.0 * np.ones_like(tsim)
                y = 0.0 * np.ones_like(tsim)

        elif self.control_mode == 'position_global':  # control the global position
            if (x is None) or (y is None):  # must set positions
                raise Exception('x or y position not set')
            else:  # v_x & v_y don't matter, set to 0
                pass
                v_x = 0.0 * np.ones_like(tsim)
                v_y = 0.0 * np.ones_like(tsim)

        elif self.control_mode == 'position_velocity':  # control the global position & velocity
            pass

        else:
            raise Exception('Control mode not available')

        # Set motor calibrations
        k_dict = {'k_x': k_x, 'k_y': k_y, 'k_psi': k_psi}
        for k in k_dict.keys():
            if k_dict[k] is None:  # defaults are 1
                k_dict[k] = 1.0 * np.ones_like(tsim)
            elif np.array(k_dict[k]).squeeze().size == 1:  # use same value across time
                k_dict[k] = k_dict[k] * np.ones_like(tsim)
            else:
                pass  # use given

        # Define the set-points to follow
        setpoint = {'x': x,
                    'y': y,
                    'z': z,
                    'v_x': v_x,
                    'v_y': v_y,
                    'v_z': 0.0*np.ones_like(tsim),
                    'psi': psi,
                    'w': w,
                    'zeta': zeta,
                    'k_x': k_dict['k_x'],
                    'k_y': k_dict['k_y'],
                    'k_psi': k_dict['k_psi'],
                    }

        # Update the simulator set-point
        self.update_dict(setpoint, name='setpoint')

    def plot_trajectory(self, start_index=0, nskip=0, size_radius=None, dpi=200):
        """ Plot the trajectory.
        """

        fig, ax = plt.subplots(1, 1, figsize=(3 * 1, 3 * 1), dpi=dpi)

        x = self.y['x'][start_index:]
        y = self.y['y'][start_index:]
        heading = self.y['psi'][start_index:]
        time = self.time[start_index:]

        if size_radius is None:
            size_radius = 0.08

        ff.plot_trajectory(x, y, heading,
                           color=time,
                           ax=ax,
                           size_radius=size_radius,
                           nskip=nskip)

        fifi.mpl_functions.adjust_spines(ax, [])


# Coordinate transformation function
def z_function(X):
    # Old states as sympy variables
    x, y, z, v_x, v_y, v_z, psi, psi_dot, w, zeta, k_x, k_y, k_psi = X

    # Body-level velocity
    v_x_bl = v_x
    v_y_bl = v_y

    # Expressions for new states in terms of old states
    g = (v_x_bl ** 2 + v_y_bl ** 2) ** (1 / 2)  # ground speed magnitude
    beta = sp.atan(v_y_bl / v_x_bl)  # ground speed angle

    # Define new state vector
    z = [x, y, z, g, beta, v_z, psi, psi_dot, w, zeta, k_x, k_y, k_psi]
    return sp.Matrix(z)
