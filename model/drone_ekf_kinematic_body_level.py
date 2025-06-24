import numpy as np
import scipy
import pandas as pd
from extended_kalman_filter import EKF, jacobian_numerical
import matplotlib.pyplot as plt


class DroneEKF:
    def __init__(self, dt=0.1, model='input', discrete=False,
                 measurement_names=('psi', 'gamma', 'beta')):
        """ Initialize the EKF.

        :param dt: time-step in seconds
        :param bool wind_polar: if true, use polar coordinates for wind
        :param iterable measurement_names: names of measurement variables
        """

        self.dt = dt
        self.model = model
        self.discrete = discrete

        # Set measurement names & current measurement Z
        self.measurement_names = tuple(measurement_names)
        self.p = len(self.measurement_names)  # number of measurements
        self.Z = pd.DataFrame(np.zeros((1, len(self.measurement_names))), columns=self.measurement_names)

        # Set state names in cartesian coordinates
        if self.discrete:  # discrete-time
            self.state_names = ('z',
                                'v_x', 'v_y',
                                'psi',
                                'w_x', 'w_y', 'w_x_dot', 'w_y_dot')
        else:  # continuous-time
            self.state_names = ('z',
                                'v_x', 'v_y',
                                'psi',
                                'w_x', 'w_y', 'w_x_dot', 'w_y_dot')

        self.n = len(self.state_names)  # number of states

        # Define what potential measurements are circular
        self.circular_measurements_default = {'psi': True,
                                              'gamma': True,
                                              'beta': True,
                                              'zeta': True,
                                              'g': False,
                                              'a': False,
                                              'r': False,
                                              'w': False,
                                              'z': False,
                                              'a_x': False,
                                              'a_y': False,
                                              'r_x': False,
                                              'r_y': False,
                                              'v_x': False,
                                              'v_y': False,
                                              'v_x_dot': False,
                                              'v_y_dot': False,
                                              'v_dot': False,
                                              'psi_dot': False,
                                              'u_x': False,
                                              'u_y': False
                                              }
        # For set measurement
        self.circular_measurements = subset_dict(self.circular_measurements_default, self.measurement_names)
        self.circular_measurements_index = tuple(list(self.circular_measurements.values()))  # as index

        # Set input names
        self.input_names = ('u_x', 'u_y', 'psi_dot')  # number of inputs
        self.m = len(self.input_names)

        # EKF
        self.ekf = None

    def run_ekf(self, x0, u_sim, y_sim, P_init=None, Q_init=1e-4, R_sim=None):
        """ Run the EKF.
        """

        # Default covariance diagonal values
        P_default_dict = {'z': 1e-2,
                          'v_x': 1e-2, 'v_y': 1e-2,
                          'v_x_dot': 1e-2, 'v_y_dot': 1e-2,
                          'psi': 1e-2, 'psi_dot': 1e-2,
                          'w_x': 1e-2, 'w_y': 1e-2, 'w_x_dot': 1e-2, 'w_y_dot': 1e-2}

        R_default_dict = {'psi': 1e-1, 'gamma': 1e-1, 'beta': 1e-1,
                          'u_x': 1e-1, 'u_y': 1e-1, 'psi_dot': 1e-2,
                          'a': 1e-1, 'r': 1e-1, 'g': 1e-1, 'z': 1e-1, 'zeta': 1e10, 'a_x': 1e-1, 'a_y': 1e-1,
                          'r_x': 1e-1, 'r_y': 1e-1, 'v_x': 1e-1, 'v_y': 1e-1, 'v_x_dot': 1e-1, 'v_y_dot': 1e-1}

        # Initial states & input
        if isinstance(x0, pd.DataFrame):
            x0 = x0.loc[:, self.state_names].values.squeeze().copy()
        elif isinstance(x0, dict):
            x0 = np.array(list(x0.values()))

        u0 = u_sim.values[0, :]
        w = u_sim.shape[0]  # number of time-steps

        # Initialize EKF
        self.ekf = EKF(self.f,
                       self.h,
                       x0, u0,
                       F_jacobian=self.F,
                       H_jacobian=self.H,
                       P=None, Q=None, R=None,
                       circular_measurements=self.circular_measurements_index)

        # Process noise covariance Q
        if isinstance(Q_init, float):
            Q = self.ekf.Q * Q_init
            Q_df = pd.DataFrame(Q, columns=self.state_names, index=self.state_names)

        elif isinstance(Q_init, dict):
            # Make data frame & populate with dict values
            Q_df = pd.DataFrame(self.ekf.Q, columns=self.state_names, index=self.state_names)
            for i in Q_df.index:  # each element of covariance matrix
                Q_df.loc[i, i] = Q_init[i]

        else:
            raise NotImplementedError

        # Set initial state covariance
        if isinstance(P_init, pd.DataFrame):  # data-frame given
            P_df = P_init.copy()

        elif isinstance(P_init, np.ndarray):  # matrix array given, make data-frame
            P_df = pd.DataFrame(P_init, index=self.state_names, columns=self.state_names)

        elif (P_init is None) or isinstance(P_init, dict):  # use default dict
            P_dict = P_default_dict.copy()  # defaults
            if isinstance(P_init, dict):  # dict given, make data-frame
                P_dict.update(P_init)  # update defaults

            # Make data frame & populate with dict values
            P_df = pd.DataFrame(self.ekf.P, columns=self.state_names, index=self.state_names)
            for i in P_df.index:  # each element of covariance matrix
                P_df.loc[i, i] = P_dict[i]
        else:
            raise TypeError('P_init must be a DataFrame, dict, or numpy array')

        # Update P
        self.ekf.P = P_df.values

        # Set measurement covariance at each time-step
        if R_sim is None:  # defaults
            R_sim = tuple([R_default_dict.copy() for _ in range(w)])
        elif isinstance(R_sim, list) or isinstance(R_sim, tuple):  # given at each time step
            R_sim = tuple(R_sim)
        else:  # make into iterable
            R_sim = tuple([R_sim for _ in range(w)])

        R_list = []
        for k in range(w):  # each time-step
            R = R_sim[k]
            if isinstance(R, pd.DataFrame):  # data-frame given
                R_df = R.copy()

            elif isinstance(R, np.ndarray):  # matrix array given, make data-frame
                R_df = pd.DataFrame(R, index=self.measurement_names, columns=self.measurement_names)

            elif isinstance(R, dict):  # dict given, make data-frame
                # Update defaults
                R_dict = R_default_dict.copy()
                R_dict.update(R)

                # Make data frame & populate with dict values
                R_df = pd.DataFrame(self.ekf.R, columns=self.measurement_names, index=self.measurement_names)
                for i in R_df.index:  # each element of covariance matrix
                    R_df.loc[i, i] = R_dict[i]
            else:
                raise TypeError('Elements of R must be DataFrames, dicts, or numpy arrays')

            # Update R at time-step
            R_list.append(R_df.copy())

        # Run EKF
        for k in range(1, w):  # each time-step
            # Current inputs
            u = u_sim.values[k - 1, :]

            # Predict
            self.ekf.predict(u=u, Q=Q_df.values)

            # Get measurement
            z = y_sim.values[k, :]

            # Update
            self.ekf.update(z, R=R_list[k].values)

        # State estimate
        x_est = pd.DataFrame(np.vstack(self.ekf.history['X']), columns=self.state_names)

        # Body-level velocity
        v_x_bl = x_est['v_x'].values
        v_y_bl = x_est['v_y'].values

        # Augmented with polar/cartesian states
        x_est['g'] = np.sqrt(v_x_bl ** 2 + v_y_bl ** 2)
        x_est['beta'] = np.arctan2(v_y_bl, v_x_bl)
        x_est['zeta'] = np.arctan2(x_est['w_y'].values, x_est['w_x'].values)
        x_est['w'] = np.sqrt(x_est['w_y'].values ** 2 + x_est['w_x'].values ** 2)

        # Get state covariance
        cov_list = []
        for i in range(self.n):
            cov = np.array([P[i, i] for P in self.ekf.history['P']])
            cov_list.append(cov)
        x_cov = np.vstack(cov_list).T
        x_cov = pd.DataFrame(x_cov, columns=self.state_names)

        # Get measurement covariance
        cov_list = []
        for i in range(self.p):
            cov = np.array([R[i, i] for R in self.ekf.history['R']])
            cov_list.append(cov)
        r_cov = np.vstack(cov_list).T
        r_cov = pd.DataFrame(r_cov, columns=self.measurement_names)

        # Measurements
        y_noise = pd.DataFrame(np.vstack(self.ekf.history['Z']), columns=self.measurement_names)

        return x_est, x_cov, r_cov, y_noise

    def simulate(self, x, u):
        """ Given the state x(kxn) & inputs over time u(kxm),
            Use the dynamics & measurement functions to simulate the system.
        """

        # Make x0 & u data-frames
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.atleast_2d(x), columns=self.state_names)

        if not isinstance(u, pd.DataFrame):
            u = pd.DataFrame(u, columns=self.input_names)

        # Get initial state, inputs, & compute initial measurement
        x0 = x.iloc[0, :].values.squeeze()
        u0 = u.values[0, :]
        y0 = self.h(x0, u0)

        # Initialize
        w = x.shape[0]
        t_sim = np.nan * np.zeros(w)
        x_sim = np.nan * x.copy()
        u_sim = np.nan * u.copy()
        y_sim = pd.DataFrame(np.zeros((w, self.p)), columns=self.measurement_names)

        x_sim.at[0, :] = x0
        u_sim.at[0, :] = u0
        y_sim.at[0, :] = y0

        # Run simulation
        x_k = x0.copy()
        for k in range(1, u.shape[0]):
            # Current inputs
            u_k = u.values[k-1, :]

            # New state
            x_k = self.f(x_k, u_k).copy()

            # Update the wind states with the new state because model doesn't account for their change
            for i, state_name in enumerate(self.state_names):  # wind state index
                if state_name in ['w_x', 'w_y', 'w_x_dot', 'w_y_dot']:
                    x_k[i] = x.loc[k, state_name]

            # Measurement
            y_k = self.h(x_k, u_k)

            # Store
            t_sim[k] = (k + 0) * self.dt
            u_sim.iloc[k, :] = u_k.copy()
            x_sim.iloc[k, :] = x_k.copy()
            y_sim.iloc[k, :] = y_k.copy()

        return t_sim, x_sim, u_sim, y_sim

    def F(self, X, U):
        """ Jacobian of dynamics function.
        """
        Jx = jacobian_numerical(self.f, X, U, epsilon=1e-5)
        return Jx

    def H(self, X, U):
        """ Jacobian of measurement function.
        """
        Jx = jacobian_numerical(self.h, X, U, epsilon=1e-5)
        return Jx

    def f(self, X, U):
        if self.discrete:
            X_k = self.f_discrete(X, U)
        else:
            def wrapped_dynamics(t, x):
                return self.f_continuous(x, U.copy())

            # Integrate
            sol = scipy.integrate.solve_ivp(wrapped_dynamics, [0, self.dt], X.copy(), method='RK45', t_eval=[self.dt])
            X_k = sol.y[:, -1]

        return X_k

    def f_discrete(self, X, U):
        """ Discrete-time system dynamics.
        """

        # Inputs
        u_x, u_y, psi_dot = U

        # States
        z, v_x, v_y, psi, w_x, w_y, w_x_dot, w_y_dot = X

        # Altitude
        z_k = z

        # Acceleration dynamics
        if self.model == 'input':
            pass
            v_x_dot_k = u_x + psi_dot*v_y
            v_y_dot_k = u_y - psi_dot*v_y
        else:
            raise NotImplementedError

        # Velocity dynamics
        v_x_k = v_x + v_x_dot_k * self.dt
        v_y_k = v_y + v_y_dot_k * self.dt

        # Heading dynamics
        psi_k = psi + psi_dot * self.dt

        # Wind dynamics
        w_x_k = w_x + w_x_dot * self.dt
        w_y_k = w_y + w_y_dot * self.dt

        # New state
        x_k = np.array([z_k, v_x_k, v_y_k, psi_k, w_x_k, w_y_k, w_x_dot, w_y_dot])

        # Return new state
        return x_k

    def f_continuous(self, X, U):
        """ Continuous-time system dynamics: dx/dt = f(X, U) """

        # Inputs
        u_x, u_y, psi_dot = U

        # States
        z, v_x, v_y, psi, w_x, w_y, w_x_dot, w_y_dot = X

        # Altitude dynamics
        z_dot = 0.0

        # Acceleration dynamics (depends on model)
        if self.model == 'input':
            v_x_dot = u_x + 1.0*v_y*psi_dot
            v_y_dot = u_y - 1.0*v_x*psi_dot
        elif self.model == 'constant_velocity':
            v_x_dot = 0.0
            v_y_dot = 0.0
        else:
            raise NotImplementedError

        # Wind dynamics
        w_x_dot = w_x_dot
        w_y_dot = w_y_dot

        # Wind accelerations
        w_x_ddot = 0.0
        w_y_ddot = 0.0

        # Combine into state derivative vector
        dxdt = np.array([z_dot,
                         v_x_dot,
                         v_y_dot,
                         psi_dot,
                         w_x_dot,
                         w_y_dot,
                         w_x_ddot,
                         w_y_ddot
                         ])

        return dxdt

    def h(self, X, U):
        """ Discrete-time measurement function.
        """

        # Inputs
        u_x, u_y, psi_dot = U

        # States
        if self.discrete:
            z, v_x, v_y, psi, w_x, w_y, w_x_dot, w_y_dot = X

            v_x_dot = u_x + v_y*psi_dot
            v_y_dot = u_y - v_x*psi_dot
        else:
            z, v_x, v_y, psi, w_x, w_y, w_x_dot, w_y_dot = X

            v_x_dot = u_x + v_y*psi_dot
            v_y_dot = u_y - v_x*psi_dot

        # Compute wind speed magnitude & direction
        w = np.sqrt(w_x ** 2 + w_y ** 2)
        zeta = np.arctan2(w_y, w_x)

        # Body-level velocity
        v_x_bl = v_x
        v_y_bl = v_y

        # Potential measurements
        g = np.sqrt(v_x_bl ** 2 + v_y_bl ** 2)  # ground speed
        beta = np.arctan2(v_y_bl, v_x_bl)  # ground speed angle
        a_x = v_x_bl - w * np.cos(psi - zeta)  # apparent airflow in x direction
        a_y = v_y_bl + w * np.sin(psi - zeta)  # apparent airflow in y direction
        a = np.sqrt(a_x ** 2 + a_y ** 2)  # apparent airflow magnitude
        gamma = np.arctan2(a_y, a_x)  # apparent airflow angle
        r = g / z  # optic flow magnitude
        r_x = v_x_bl / z
        r_y = v_y_bl / z
        # alpha = np.arctan2(v_y_dot, v_x_dot)  # acceleration angle
        # v_dot = np.sqrt(v_y_dot ** 2 + v_x_dot ** 2)

        # Create dict of potential measurements
        measurements = {'psi': psi,
                        'gamma': gamma,
                        'beta': beta,
                        'zeta': zeta,
                        'g': g,
                        'a': a,
                        'r': r,
                        'w': w,
                        'z': z,
                        'u_x': u_x,
                        'u_y': u_y,
                        'psi_dot': psi_dot,
                        'r_x': r_x,
                        'r_y': r_y,
                        'v_x_dot': v_x_dot,
                        'v_y_dot': v_y_dot,
                        'v_x': v_x,
                        'v_y': v_y,
                        'a_x': a_x,
                        'a_y': a_y,
                        'w_x': w_x,
                        'w_y': w_y}

        # Set current measurement based on measurement names
        measurements_df = pd.DataFrame(measurements, index=[0])
        self.Z = measurements_df.loc[:, self.measurement_names]
        Z = self.Z.values.squeeze()

        # Return measurement
        return Z


def subset_dict(d, keys):
    return {k: d[k] for k in keys if k in d}
