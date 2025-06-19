import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import figurefirst as fifi
import figure_functions as ff
from pybounds import Simulator
import scipy
import sympy as sp








class DroneParameters:
    """ Stores drone parameters.
    """

    def __init__(self):
        self.g = 9.81  # gravity [m/s^2]
        self.m = 0.086  # mass [kg]
        self.M = 2.529  # mass [kg]
        self.Mm = 4 * self.m + self.M  # total mass [kg]
        self.L = 0.2032  # length [m]
        self.R = 0.1778  # average body radius [m]
        self.I_x = .491
        # self.I_x = 2 * (self.M * self.R ** 2) / 5 + 2 * self.m * self.L ** 2  # [kg*m^2] moment of inertia about x
        self.I_y = .387
        # self.I_y = 2 * (self.M * self.R ** 2) / 5 + 2 * self.m * self.L ** 2  # [kg*m^2] moment of inertia about y
        self.I_z = .667
        # self.I_z = 2 * (self.M * self.R ** 2) / 5 + 4 * self.m * self.L ** 2  # [kg*m^2] moment of inertia about y
        # self.b = 1.8311  # thrust coefficient
        self.b = 1.34
        self.d = 1  # drag constant
        self.C = 0.1  # drag coefficient from ground speed plus air speed

    def get_params(self):
        return {
            'g': self.g,
            'm': self.m,
            'M': self.M,
            'Mm': self.Mm,
            'L': self.L,
            'R': self.R,
            'I_x': self.I_x,
            'I_y': self.I_y,
            'I_z': self.I_z,
            'b': self.b,
            'd': self.d,
            'C': self.C
                            }


class DroneModel:
    """ Stores drone model
    """

    def __init__(self):
        # Parameters
        self.params = DroneParameters()

        # State names
        # self.state_names = ['x',  # x position in inertial frame [m]
        #                     'y',  # y position in inertial frame [m]
        #                     'z',  # z position in inertial frame [m]
        #                     'v_x',  # x velocity in body-level frame (parallel to heading) [m/s]
        #                     'v_y',  # y velocity in body-level frame (perpendicular to heading) [m/s]
        #                     'v_z',  # z velocity in body-level frame [m/s]
        #                     'phi',  # roll in body frame [rad],
        #                     'theta',  # pitch in vehicle-2 frame [rad],
        #                     'psi',  # yaw in body-level frame (vehicle-1) [rad]
        #                     'omega_x',  # roll rate in body-frame [rad/s]
        #                     'omega_y',  # pitch rate in body-frame [rad/s]
        #                     'omega_z',  # yaw rate in body-frame [rad/s]
        #                     'wx',  # wind speed in XY-plane [ms]
        #                     'wy',  # wind direction in XY-plane[rad]
        #                     # 'w',  # wind speed in XY-plane [ms]
        #                     # 'zeta',  # wind direction in XY-plane[rad]

        #                     # 'm',  # mass [kg]
        #                     # 'I_x',  # mass moment of inertia about body x-axis [kg*m^2]
        #                     # 'I_y',  # mass moment of inertia about body y-axis [kg*m^2]
        #                     # 'I_z',  # mass moment of inertia about body z-axis [kg*m^2]
        #                     # 'C',  # translational drag damping constant [N/m/s]
        #                     ]

        self.state_names = ['x',  # x position in inertial frame [m]
                            'y',  # y position in inertial frame [m]
                            'z',  # z position in inertial frame [m]
                            'v_x',  # x velocity in body-level frame (parallel to heading) [m/s]
                            'v_y',  # y velocity in body-level frame (perpendicular to heading) [m/s]
                            'v_z',  # z velocity in body-level frame [m/s]
                            'phi',  # roll in body frame [rad],
                            'theta',  # pitch in vehicle-2 frame [rad],
                            'psi',  # yaw in body-level frame (vehicle-1) [rad]
                            'omega_x',  # roll rate in body-frame [rad/s]
                            'omega_y',  # pitch rate in body-frame [rad/s]
                            'omega_z',  # yaw rate in body-frame [rad/s]
                            'wx',  # wind speed in XY-plane [ms]
                            'wy',  # wind direction in XY-plane[rad]
                            # 'w',  # wind speed in XY-plane [ms]
                            # 'zeta',  # wind direction in XY-plane[rad]

                            # 'm',  # mass [kg]
                            # 'I_x',  # mass moment of inertia about body x-axis [kg*m^2]
                            # 'I_y',  # mass moment of inertia about body y-axis [kg*m^2]
                            # 'I_z',  # mass moment of inertia about body z-axis [kg*m^2]
                            # 'C',  # translational drag damping constant [N/m/s]
                            ]

        # Input names
        self.input_names = ['PWM1',  # PWM [-]
                            'PWM2',  # PWM [-]
                            'PWM3',  # PWM [-]
                            'PWM4',  # PWM [-]
                            ]

        # self.input_names = ['PWM1',  # PWM [-]
        #                     'PWM2',  # PWM [-]
        #                     'PWM3',  # PWM [-]
        #                     'PWM4',  # PWM [-]
        #                     'phi',  # roll in body frame [rad]
        #                     'theta',  # pitch in vehicle-2 frame [rad]
        #                     ]

        # Measurement names
        # self.measurement_names = ['g', 'beta', 'a', 'gamma', 'q', 'alpha', 'r']
        self.measurement_names = ['Ax','Ay','Az','Axt','Ayt','rx','ry','Wax','Way']
        self.measurement_names = self.state_names + self.measurement_names

    def f(self, X, U):
        """ Dynamic model.
        """
        m = self.params.Mm
        Ix = self.params.I_x
        Iy = self.params.I_y
        Iz = self.params.I_z
        C = self.params.C
        g = self.params.g

        # States
        x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, wx, wy = np.ravel(X)

        # Inputs
        PWM1,PWM2,PWM3,PWM4 = np.ravel(U)

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)
        
        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

        # Drag dynamics
        w = np.sqrt(wx ** 2 + wy ** 2)
        zeta = np.arctan2(wy, wx)
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_z = v_z

        # Rotation
        phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_y * np.sin(phi) * (1 / np.cos(theta))

        omega_x_dot = (1 / Ix) * u_phi + omega_z * omega_y * (Iy - Iz) / Ix
        omega_y_dot = (1 / Iy) * (u_theta) + omega_z * omega_x * (Iz - Ix) / Iy
        omega_z_dot = (1 / Iz) * (u_psi) + omega_x * omega_y * (Ix - Iy) / Iz

        # Position in inertial frame (ENU, unchanged)
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        z_dot = v_z

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        v_z_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_z - m * g)

        # Wind remains unchanged
        wx_dot = 0 * wx
        wy_dot = 0 * wy


        # Package and return xdot
        x_dot = [x_dot, y_dot, z_dot,
                 v_x_dot, v_y_dot, v_z_dot,
                 phi_dot, theta_dot, psi_dot,
                 omega_x_dot, omega_y_dot, omega_z_dot,
                 wx_dot, wy_dot
                 ]

        return x_dot
    
    def f2(self, X, U):
        """ Dynamic model.
        """
        m = self.params.Mm
        Ix = self.params.I_x
        Iy = self.params.I_y
        Iz = self.params.I_z
        C = self.params.C
        g = self.params.g

        # States
        x, y, z, v_x, v_y, v_z, psi, omega_x, omega_y, omega_z, wx, wy = np.ravel(X)

        # Inputs
        PWM1,PWM2,PWM3,PWM4, phi, theta = np.ravel(U)

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)
        
        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

        # Drag dynamics
        w = np.sqrt(wx ** 2 + wy ** 2)
        zeta = np.arctan2(wy, wx)
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_z = v_z

        u_x = u_thrust * np.cos(phi) * np.sin(theta) - C * a_x
        u_y = -u_thrust * np.sin(phi) - C * a_y
        u_z = u_thrust * np.cos(phi) * np.cos(theta) - C * v_z

        # Rotation
        # phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        # theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_y * np.sin(phi) * (1 / np.cos(theta))

        omega_x_dot = (1 / Ix) * u_phi + omega_z * omega_y * (Iy - Iz) / Ix
        omega_y_dot = (1 / Iy) * (u_theta) + omega_z * omega_x * (Iz - Ix) / Iy
        omega_z_dot = (1 / Iz) * (u_psi) + omega_x * omega_y * (Ix - Iy) / Iz

        # Position in inertial frame (ENU, unchanged)
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        z_dot = v_z

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (u_y) - v_x * psi_dot
        v_z_dot = (1 / m) * (u_z  - m * g)

        # Wind remains unchanged
        wx_dot = 0 * wx
        wy_dot = 0 * wy


        # Package and return xdot
        x_dot = [x_dot, y_dot, z_dot,
                 v_x_dot, v_y_dot, v_z_dot,
                 psi_dot,
                 omega_x_dot, omega_y_dot, omega_z_dot,
                 wx_dot, wy_dot
                 ]

        return x_dot
    
    def f_c(self, X, U, DT):
        """ Continuous-time dynamic model.
        """
        # m = self.params.Mm
        # Ix = self.params.Ix
        # Iy = self.params.Iy
        # Iz = self.params.Iz
        # C = self.params.C
        g = self.params.g

        # States
        x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta, m, Ix, Iy, Iz, C = np.ravel(X)

        # Inputs
        PWM1,PWM2,PWM3,PWM4 = np.ravel(U)

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)
        
        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

        # Drag dynamics
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_z = v_z

        # Rotation
        phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_y * np.sin(phi) * (1 / np.cos(theta))

        omega_x_dot = (1 / Ix) * u_phi + omega_z * omega_y * (Iy - Iz) / Ix
        omega_y_dot = (1 / Iy) * (u_theta) + omega_z * omega_x * (Iz - Ix) / Iy
        omega_z_dot = (1 / Iz) * (u_psi) + omega_x * omega_y * (Ix - Iy) / Iz

        # Position in inertial frame (ENU, unchanged)
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        z_dot = v_z

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        v_z_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_z - m * g)

        # Wind remains unchanged
        w_dot = 0 * w
        zeta_dot = 0 * zeta

        # Parameters remain constant
        m_dot = 0 * m
        I_x_dot = 0 * Ix
        I_y_dot = 0 * Iy
        I_z_dot = 0 * Iz
        C_dot = 0 * C
        new_X = scipy.integrate.odeint(lambda X, t: [x_dot, y_dot, z_dot,v_x_dot, v_y_dot, v_z_dot, phi_dot, theta_dot, psi_dot, omega_x_dot, omega_y_dot, omega_z_dot, w_dot, zeta_dot, m_dot, I_x_dot, I_y_dot, I_z_dot, C_dot], np.ravel(X), [0, DT])[-1]
        new_X= np.atleast_2d(new_X).T
        return new_X
    
    def f_c_car(self, X, U, DT):
        """ Continuous-time dynamic model.
        """
        # m = self.params.Mm
        # Ix = self.params.I_x
        # Iy = self.params.I_y
        # Iz = self.params.I_z
        # C = self.params.C
        # g = self.params.g

        # States
        # x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, wx, wy = np.ravel(X)

        # Inputs
        # PWM1,PWM2,PWM3,PWM4 = np.ravel(U)

        # # for x shaped drone
        # Lx = 2 * self.params.L / np.sqrt(2)
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        # u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        # u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)
        
        
        # # Drag dynamics
        # w = np.sqrt(wx ** 2 + wy ** 2)
        # zeta = np.arctan2(wy, wx)
        # a_x = v_x - w * np.cos(psi - zeta)
        # a_y = v_y + w * np.sin(psi - zeta)
        # a_z = v_z

        # # Rotation
        # phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        # theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        # psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_y * np.sin(phi) * (1 / np.cos(theta))

        # omega_x_dot = (1 / Ix) * u_phi + omega_z * omega_y * (Iy - Iz) / Ix
        # omega_y_dot = (1 / Iy) * (u_theta) + omega_z * omega_x * (Iz - Ix) / Iy
        # omega_z_dot = (1 / Iz) * (u_psi) + omega_x * omega_y * (Ix - Iy) / Iz

        # # Position in inertial frame (ENU, unchanged)
        # x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        # y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        # z_dot = v_z

        # # Velocity in body-level frame (adjusted for FLU)
        # v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        # v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        # v_z_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_z - m * g)

        # # Wind remains unchanged
        # wx_dot = 0 * wx
        # wy_dot = 0 * wy
        # x_dot = self.f(X, U)
        # new_X = scipy.integrate.odeint(lambda X, t: x_dot, np.ravel(X), [0, DT])[-1]

        # # new_X = scipy.integrate.odeint(lambda X, t: [x_dot, y_dot, z_dot,v_x_dot, v_y_dot, v_z_dot, phi_dot, theta_dot, psi_dot, omega_x_dot, omega_y_dot, omega_z_dot, wx_dot, wy_dot], np.ravel(X), [0, DT])[-1]
        # new_X= np.atleast_2d(new_X).T
        def wrapped_dynamics(t, x):
            return self.f(x, U)

        solution = scipy.integrate.solve_ivp(wrapped_dynamics,
                                              [0, DT],
                                                np.ravel(X),
                                                  method='RK45',
                                                    t_eval=[DT])

        new_X = np.atleast_2d(solution.y[:, -1]).T
        return new_X
    
    def f_c_car2(self, X, U, DT):
        """ Continuous-time dynamic model.
        """

        def wrapped_dynamics(t, x):
            return self.f2(x, U)

        solution = scipy.integrate.solve_ivp(wrapped_dynamics,
                                              [0, DT],
                                                np.ravel(X),
                                                  method='RK45',
                                                    t_eval=[DT])

        new_X = np.atleast_2d(solution.y[:, -1]).T
        return new_X
    
    def f_kin_car(self, X, U, DT):
        """ Continuous-time dynamic model.
        """
        # m = self.params.Mm
        # Ix = self.params.I_x
        # Iy = self.params.I_y
        # Iz = self.params.I_z
        # C = self.params.C
        # g = self.params.g

        # States
        x, y, z, v_x, v_y, v_z, phi, theta, psi, wx, wy = np.ravel(X)

        # Inputs
        u_thrust, u_phi, u_theta, u_psi = np.ravel(U)

        # Rotation
        phi_dot = u_phi
        theta_dot = u_theta
        psi_dot = u_psi


        # Position in inertial frame (ENU, unchanged)
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        z_dot = v_z

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = u_thrust * np.cos(phi) * np.sin(theta)
        v_y_dot = -u_thrust * np.sin(phi) 
        v_z_dot = u_thrust * np.cos(phi) * np.cos(theta)

        # Wind remains unchanged
        wx_new = 0*wx
        wy_new = 0*wy

        new_X = scipy.integrate.odeint(lambda X, t: [x_dot, y_dot, z_dot,v_x_dot, v_y_dot, v_z_dot, phi_dot, theta_dot, psi_dot, wx, wy], np.ravel(X), [0, DT])[-1]
        new_X= np.atleast_2d(new_X).T
        return new_X
    
    def f_c_small(self, X, U, DT):
        """ Continuous-time dynamic model.
        """
        m = self.params.Mm
        Ix = self.params.I_x
        Iy = self.params.I_y
        Iz = self.params.I_z
        C = self.params.C
        g = self.params.g

        # States
        x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta = np.ravel(X)

        # Inputs
        PWM1,PWM2,PWM3,PWM4 = np.ravel(U)

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)
        
        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

            # Drag dynamics
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_z = v_z

        # Rotation
        phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_y * np.sin(phi) * (1 / np.cos(theta))

        omega_x_dot = (1 / Ix) * u_phi + omega_z * omega_y * (Iy - Iz) / Ix
        omega_y_dot = (1 / Iy) * (u_theta) + omega_z * omega_x * (Iz - Ix) / Iy
        omega_z_dot = (1 / Iz) * (u_psi) + omega_x * omega_y * (Ix - Iy) / Iz

        # Position in inertial frame (ENU, unchanged)
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        z_dot = v_z

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        v_z_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_z - m * g)*0.0  

        # Wind remains unchanged
        w_dot = 0 * w
        zeta_dot = 0 * zeta

        # Parameters remain constant
        m_dot = 0 * m
        I_x_dot = 0 * Ix
        I_y_dot = 0 * Iy
        I_z_dot = 0 * Iz
        C_dot = 0 * C
        new_X = scipy.integrate.odeint(lambda X, t: [x_dot, y_dot, z_dot,v_x_dot, v_y_dot, v_z_dot, phi_dot, theta_dot, psi_dot, omega_x_dot, omega_y_dot, omega_z_dot, w_dot, zeta_dot], np.ravel(X), [0, DT])[-1]
        new_X= np.atleast_2d(new_X).T
        return new_X
    
    def f_c_mu(self, X, U, DT):
        """ Continuous-time dynamic model.
        """
        m = self.params.Mm
        Ix = self.params.I_x
        Iy = self.params.I_y
        Iz = self.params.I_z
        C = self.params.C
        g = self.params.g

        # States
        # x, y, mu, v_x, v_y, v_mu, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta, m, Ix, Iy, Iz, C = np.ravel(X)
        x, y, mu, v_x, v_y, v_mu, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta = np.ravel(X)

        # Inputs
        PWM1,PWM2,PWM3,PWM4 = np.ravel(U)

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)
        
        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

            # Drag dynamics
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_mu = v_mu

        # Rotation
        phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_y * np.sin(phi) * (1 / np.cos(theta))

        omega_x_dot = (1 / Ix) * u_phi + omega_z * omega_y * (Iy - Iz) / Ix
        omega_y_dot = (1 / Iy) * (u_theta) + omega_z * omega_x * (Iz - Ix) / Iy
        omega_z_dot = (1 / Iz) * (u_psi) + omega_x * omega_y * (Ix - Iy) / Iz

        # Position in inertial frame (ENU, unchanged)
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        mu_dot = v_mu

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        v_mu_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_mu - m * g)*0.0  

        # Wind remains unchanged
        w_dot = 0 * w
        zeta_dot = 0 * zeta

        # Parameters remain constant
        # m_dot = 0 * m
        # I_x_dot = 0 * Ix
        # I_y_dot = 0 * Iy
        # I_z_dot = 0 * Iz
        # C_dot = 0 * C
        new_X = scipy.integrate.odeint(lambda X, t: [x_dot, y_dot, mu_dot,v_x_dot, v_y_dot, v_mu_dot, phi_dot, theta_dot, psi_dot, omega_x_dot, omega_y_dot, omega_z_dot, w_dot, zeta_dot], np.ravel(X), [0, DT])[-1]
        new_X= np.atleast_2d(new_X).T
        return new_X

    def h(self, X, U):
        """ Measurement model.
        """
        m = self.params.Mm
        Ix = self.params.I_x
        Iy = self.params.I_y
        Iz = self.params.I_z
        C = self.params.C
        g = self.params.g

        # States
        x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, wx, wy  = X

        # Inputs
        PWM1,PWM2,PWM3,PWM4 = U

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)

        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

        # Rotation
        phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_z * np.sin(phi) * (1 / np.cos(theta))

        # Ground speed & course direction in body-level frame
        G = np.sqrt(v_x ** 2 + v_y ** 2)
        r = G / z
        rx = v_x / z
        ry = v_y / z
        beta = np.arctan2(v_y, v_x)

        # Airspeed & apparent airflow angle in body-level frame
        w = np.sqrt(wx ** 2 + wy ** 2)
        zeta = np.arctan2(wy, wx)
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_z = v_z
        a = np.sqrt(a_x ** 2 + a_y ** 2)
        gamma = np.arctan2(a_y, a_x)

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        v_z_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_z - m * g)

        v_x_dot_test = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot*0.0
        v_y_dot_test = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot*0.0

        q = np.sqrt(v_x_dot ** 2 + v_y_dot ** 2)
        alpha = np.arctan2(v_y_dot, v_x_dot)

        # Unwrap angles
        if np.array(phi).ndim > 0:
            if np.array(phi).shape[0] > 1:
                phi = np.unwrap(phi)
                theta = np.unwrap(theta)
                psi = np.unwrap(psi)
                beta = np.unwrap(beta)
                alpha = np.unwrap(alpha)
        # self.measurement_names = ['Ax','Ay','Az', 'rx','ry','Wax','Way']
        # Y = np.hstack((X, beta))
        Y = [x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, wx, wy,
             v_x_dot, v_y_dot, v_z_dot, v_x_dot_test, v_y_dot_test, rx, ry, a_x, a_y ]

        return Y
    
    def h2(self, X, U):
        """ Measurement model.
        """
        m = self.params.Mm
        Ix = self.params.I_x
        Iy = self.params.I_y
        Iz = self.params.I_z
        C = self.params.C
        g = self.params.g

        # States
        x, y, z, v_x, v_y, v_z, psi, omega_x, omega_y, omega_z, wx, wy  = X

        # Inputs
        PWM1,PWM2,PWM3,PWM4, phi, theta = U

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        u_x = u_thrust * np.cos(phi) * np.sin(theta)
        u_y = -u_thrust * np.sin(phi)
        u_z = u_thrust * np.cos(phi) * np.cos(theta)
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)

        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

        # Rotation
        # phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        # theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_z * np.sin(phi) * (1 / np.cos(theta))

        # Ground speed & course direction in body-level frame
        G = np.sqrt(v_x ** 2 + v_y ** 2)
        r = G / z
        rx = v_x / z
        ry = v_y / z
        beta = np.arctan2(v_y, v_x)

        # Airspeed & apparent airflow angle in body-level frame
        w = np.sqrt(wx ** 2 + wy ** 2)
        zeta = np.arctan2(wy, wx)
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_z = v_z
        a = np.sqrt(a_x ** 2 + a_y ** 2)
        gamma = np.arctan2(a_y, a_x)

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_x - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (u_y - C * a_y) - v_x * psi_dot
        v_z_dot = (1 / m) * (u_z - C * v_z - m * g)

        v_x_dot_test = (1 / m) * (u_x - C * a_x) + v_y * psi_dot*0.0
        v_y_dot_test = (1 / m) * (u_y - C * a_y) - v_x * psi_dot*0.0

        q = np.sqrt(v_x_dot ** 2 + v_y_dot ** 2)
        alpha = np.arctan2(v_y_dot, v_x_dot)

        # Unwrap angles
        if np.array(phi).ndim > 0:
            if np.array(phi).shape[0] > 1:
                phi = np.unwrap(phi)
                theta = np.unwrap(theta)
                psi = np.unwrap(psi)
                beta = np.unwrap(beta)
                alpha = np.unwrap(alpha)
        # self.measurement_names = ['Ax','Ay','Az', 'rx','ry','Wax','Way']
        # Y = np.hstack((X, beta))
        Y = [x, y, z, v_x, v_y, v_z, psi, omega_x, omega_y, omega_z, wx, wy,
             v_x_dot, v_y_dot, v_z_dot, v_x_dot_test, v_y_dot_test, rx, ry, a_x, a_y ]

        return Y
    
    def h_c(self, X, U):
        """ Continuous-time measurement model.
        """
        # m = self.params.Mm
        # Ix = self.params.Ix
        # Iy = self.params.Iy
        # Iz = self.params.Iz
        # C = self.params.C
        g = self.params.g

        # States
        x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta, m, Ix, Iy, Iz, C = np.ravel(X)

        # Inputs
        PWM1,PWM2,PWM3,PWM4 = np.ravel(U)

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)

        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

        # Rotation
        phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_z * np.sin(phi) * (1 / np.cos(theta))

        # Ground speed & course direction in body-level frame
        G = np.sqrt(v_x ** 2 + v_y ** 2)
        r = G / z
        beta = np.arctan2(v_y, v_x)

        # Airspeed & apparent airflow angle in body-level frame
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_z = v_z
        a = np.sqrt(a_x ** 2 + a_y ** 2)
        gamma = np.arctan2(a_y, a_x)

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        v_z_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_z - m * g)

        q = np.sqrt(v_x_dot ** 2 + v_y_dot ** 2)
        alpha = np.arctan2(v_y_dot, v_x_dot)

        # Unwrap angles
        if np.array(phi).ndim > 0:
            if np.array(phi).shape[0] > 1:
                phi = np.unwrap(phi)
                theta = np.unwrap(theta)
                psi = np.unwrap(psi)
                beta = np.unwrap(beta)
                alpha = np.unwrap(alpha)

        # mocap measurements
        Px = x
        Py = y
        Pz = z
        P_cluster = np.array([Px, Py, Pz])
        Vx = v_x
        Vy = v_y
        Vz = v_z
        V_cluster = np.array([Vx, Vy, Vz])
        Phi = phi
        Theta = theta
        Psi = psi
        Attitude_cluster = np.array([Phi, Theta, Psi])
        Mocap_cluster = np.array([Px, Py, Pz, Vx, Vy, Vz, Phi, Theta, Psi])
        # IMU measurements
        Omega_x = omega_x
        Omega_y = omega_y
        Omega_z = omega_z
        Omega_cluster = np.array([Omega_x, Omega_y, Omega_z])
        Ax = v_x_dot
        Ay = v_y_dot
        Az = v_z_dot
        A_cluster = np.array([Ax, Ay, Az])
        IMU_cluster = np.array([Phi, Theta, Psi, Omega_x, Omega_y, Omega_z, Ax, Ay, Az])
        # optical flow measurements
        OF_x = v_x*(1/z)
        OF_y = v_y*(1/z)
        OF_XY = np.array([OF_x, OF_y])
        OF_z = v_z*(1/z)
        OF_cluster = np.array([OF_x, OF_y, OF_z])
        OF_cluster_XY = np.array([OF_x, OF_y])
        # wind measurements
        Awx = a_x
        Awy = a_y
        Awz = a_z
        Aa = a
        Agamma = gamma
        Wind_XY = np.array([Awx, Awy])
        Wind_cluster = np.array([Awx, Awy, Awz, Aa, Agamma])
        Wind_cluster_XY = np.array([Awx, Awy,Aa, Agamma])

        y_all=np.atleast_2d(np.hstack((P_cluster, V_cluster, Attitude_cluster, Omega_cluster, A_cluster, OF_cluster_XY, Wind_XY))).T
        y_real = np.atleast_2d(np.hstack((IMU_cluster, OF_cluster_XY, Wind_XY))).T
        Y = y_real
        # Y= y_all
        return Y

    def h_kin_car(self, X, U):
        """ Continuous-time measurement model.
        """

        # States
        x, y, z, v_x, v_y, v_z, phi, theta, psi, wx, wy= np.ravel(X)

        # Inputs
        u_thrust, u_phi, u_theta, u_psi = np.ravel(U)

        # Rotation
        phi_dot = u_phi
        theta_dot = u_theta
        psi_dot = u_psi

        # Ground speed & course direction in body-level frame
        G = np.sqrt(v_x ** 2 + v_y ** 2)
        r = G / z
        beta = np.arctan2(v_y, v_x)

        # Airspeed & apparent airflow angle in body-level frame
        # Drag dynamics
        w = np.sqrt(wx ** 2 + wy ** 2)
        zeta = np.arctan2(wy, wx)
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_z = v_z
        a = np.sqrt(a_x ** 2 + a_y ** 2)
        gamma = np.arctan2(a_y, a_x)

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        v_z_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_z - m * g)

        q = np.sqrt(v_x_dot ** 2 + v_y_dot ** 2)
        alpha = np.arctan2(v_y_dot, v_x_dot)

        # Unwrap angles
        if np.array(phi).ndim > 0:
            if np.array(phi).shape[0] > 1:
                phi = np.unwrap(phi)
                theta = np.unwrap(theta)
                psi = np.unwrap(psi)
                beta = np.unwrap(beta)
                alpha = np.unwrap(alpha)

        # mocap measurements
        Px = x
        Py = y
        Pz = z
        P_cluster = np.array([Px, Py, Pz])
        Vx = v_x
        Vy = v_y
        Vz = v_z
        V_cluster = np.array([Vx, Vy, Vz])
        Phi = phi
        Theta = theta
        Psi = psi
        Attitude_cluster = np.array([Phi, Theta, Psi])
        Mocap_cluster = np.array([Px, Py, Pz, Vx, Vy, Vz, Phi, Theta, Psi])
        # IMU measurements
        Omega_x = omega_x
        Omega_y = omega_y
        Omega_z = omega_z
        Omega_cluster = np.array([Omega_x, Omega_y, Omega_z])
        Ax = v_x_dot
        Ay = v_y_dot
        Az = v_z_dot
        A_cluster = np.array([Ax, Ay, Az])
        IMU_cluster = np.array([Phi, Theta, Psi, Omega_x, Omega_y, Omega_z, Ax, Ay, Az])
        # optical flow measurements
        OF_x = v_x*(1/z)
        OF_y = v_y*(1/z)
        OF_XY = np.array([OF_x, OF_y])
        OF_z = v_z*(1/z)
        OF_cluster = np.array([OF_x, OF_y, OF_z])
        OF_cluster_XY = np.array([OF_x, OF_y])
        # wind measurements
        Awx = a_x
        Awy = a_y
        Awz = a_z
        Aa = a
        Agamma = gamma
        Wind_XY = np.array([Awx, Awy])
        Wind_cluster = np.array([Awx, Awy, Awz, Aa, Agamma])
        Wind_cluster_XY = np.array([Awx, Awy,Aa, Agamma])

        Y_all=np.atleast_2d(np.hstack((P_cluster, V_cluster, Attitude_cluster, Omega_cluster, A_cluster, OF_cluster_XY, Wind_XY))).T
        Y_real = np.atleast_2d(np.hstack((IMU_cluster, OF_cluster_XY, Wind_XY))).T
        Y_no_wind = np.atleast_2d(np.hstack((IMU_cluster, OF_cluster_XY))).T
        Y_no_OF = np.atleast_2d(np.hstack((IMU_cluster, Wind_XY))).T
        Y_IMU_only = np.atleast_2d(np.hstack((IMU_cluster))).T
        Y = Y_IMU_only
        # Y= y_all
        return Y
    

    def h_c_car(self, X, U, Y_SWEEP):
        """ Continuous-time measurement model.
        """
        # m = self.params.Mm
        # Ix = self.params.I_x
        # Iy = self.params.I_y
        # Iz = self.params.I_z
        # C = self.params.C
        # g = self.params.g

        # # States
        # x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, wx, wy  = X

        # # Inputs
        # PWM1,PWM2,PWM3,PWM4 = U

        # # for x shaped drone
        # Lx = 2 * self.params.L / np.sqrt(2)
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        # u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        # u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)

        # # for + shaped drone
        # # Lx = self.params.L
        # # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

        # # Rotation
        # phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        # theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        # psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_z * np.sin(phi) * (1 / np.cos(theta))

        # # Ground speed & course direction in body-level frame
        # G = np.sqrt(v_x ** 2 + v_y ** 2)
        # r = G / z
        # rx = v_x / z
        # ry = v_y / z
        # beta = np.arctan2(v_y, v_x)

        # # Airspeed & apparent airflow angle in body-level frame
        # w = np.sqrt(wx ** 2 + wy ** 2)
        # zeta = np.arctan2(wy, wx)
        # a_x = v_x - w * np.cos(psi - zeta)
        # a_y = v_y + w * np.sin(psi - zeta)
        # a_z = v_z
        # a = np.sqrt(a_x ** 2 + a_y ** 2)
        # gamma = np.arctan2(a_y, a_x)

        # # Velocity in body-level frame (adjusted for FLU)
        # v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        # v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        # v_z_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_z - m * g)

        # v_x_dot_test = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot*0.0
        # v_y_dot_test = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot*0.0

        # q = np.sqrt(v_x_dot ** 2 + v_y_dot ** 2)
        # alpha = np.arctan2(v_y_dot, v_x_dot)

        # # Unwrap angles
        # if np.array(phi).ndim > 0:
        #     if np.array(phi).shape[0] > 1:
        #         phi = np.unwrap(phi)
        #         theta = np.unwrap(theta)
        #         psi = np.unwrap(psi)
        #         beta = np.unwrap(beta)
        #         alpha = np.unwrap(alpha)

        Y = self.h(X,U)
        # Y = [x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, wx, wy,
        #      v_x_dot, v_y_dot, v_z_dot, v_x_dot_test, v_y_dot_test, rx, ry, a_x, a_y ]
        # mocap measurements
        Px = Y[0]
        Py = Y[1]
        Pz = Y[2]
        P_cluster = np.array([Px, Py, Pz])
        Vx = Y[3]
        Vy = Y[4]
        Vz = Y[5]
        V_cluster = np.array([Vx, Vy, Vz])
        Phi = Y[6]
        Theta = Y[7]
        Psi = Y[8]
        Attitude_cluster = np.array([Phi, Theta, Psi])
        Mocap_cluster = np.array([Px, Py, Pz, Vx, Vy, Vz, Phi, Theta, Psi])
        # IMU measurements
        Omega_x = Y[9]
        Omega_y = Y[10]
        Omega_z = Y[11]
        Omega_cluster = np.array([Omega_x, Omega_y, Omega_z])
        Ax = Y[17]
        Ay = Y[18]
        Az = Y[16]
        A_cluster = np.array([Ax, Ay, Az])
        IMU_cluster = np.array([Phi, Theta, Psi, Omega_x, Omega_y, Omega_z, Ax, Ay, Az])
        # optical flow measurements
        OF_x = Y[19]
        OF_y = Y[20]
        OF_XY = np.array([OF_x, OF_y])
        OF_z = Vz
        OF_cluster = np.array([OF_x, OF_y, OF_z])
        OF_cluster_XY = np.array([OF_x, OF_y])
        # wind measurements
        Awx = Y[21]
        Awy = Y[22]
        Awz = Vz
        # Aa = a
        # Agamma = gamma
        Wind_XY = np.array([Awx, Awy])
        # Wind_cluster = np.array([Awx, Awy, Awz, Aa, Agamma])
        # Wind_cluster_XY = np.array([Awx, Awy,Aa, Agamma])

        # y_all=np.atleast_2d(np.hstack((P_cluster, V_cluster, Attitude_cluster, Omega_cluster, A_cluster, OF_cluster, Wind_cluster))).T
        # y_real = np.atleast_2d(np.hstack((IMU_cluster, OF_cluster_XY, Wind_XY))).T
        if Y_SWEEP == 'IMU':
            Y = np.asarray(IMU_cluster).reshape(-1, 1)
        elif Y_SWEEP == 'IMU + OPTIC_FLOW':
            Y = np.concatenate((IMU_cluster, OF_cluster_XY)).reshape(-1, 1)
        elif Y_SWEEP == 'IMU + WIND':
            Y = np.concatenate((IMU_cluster, Wind_XY)).reshape(-1, 1)
        elif Y_SWEEP == 'IMU + WIND + OPTIC_FLOW':
            Y = np.concatenate((IMU_cluster, OF_cluster_XY, Wind_XY)).reshape(-1, 1)
        elif Y_SWEEP == 'IMU + VEL + WIND':
            Y = np.concatenate((IMU_cluster, V_cluster, Wind_XY)).reshape(-1, 1)
        return Y

    def h_c_mu(self, X, U):
        """ Continuous-time measurement model.
        """
        m = self.params.Mm
        Ix = self.params.I_x
        Iy = self.params.I_y
        Iz = self.params.I_z
        C = self.params.C
        g = self.params.g

        # States
        # x, y, mu, v_x, v_y, v_mu, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta, m, Ix, Iy, Iz, C = np.ravel(X)
        x, y, mu, v_x, v_y, v_mu, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta = np.ravel(X)
        # Inputs
        PWM1,PWM2,PWM3,PWM4 = np.ravel(U)

        # for x shaped drone
        Lx = 2 * self.params.L / np.sqrt(2)
        u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        u_phi = self.params.b * Lx * (-PWM1 + PWM2 + PWM3 - PWM4)
        u_theta = self.params.b * Lx * (-PWM1 + PWM2 - PWM3 + PWM4)
        u_psi = self.params.d * (-PWM1 - PWM2 + PWM3 + PWM4)

        # for + shaped drone
        # Lx = self.params.L
        # u_thrust = self.params.b * (PWM1 + PWM2 + PWM3 + PWM4)
        # u_phi = self.params.b * Lx * (-PWM2 + PWM4)
        # u_theta = self.params.b * Lx * (PWM1 - PWM3)
        # u_psi = self.params.d * (-PWM1 + PWM2 - PWM3 + PWM4)

        # Rotation
        phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_z * np.sin(phi) * (1 / np.cos(theta))

        # Ground speed & course direction in body-level frame
        G = np.sqrt(v_x ** 2 + v_y ** 2)
        r = G *mu
        beta = np.arctan2(v_y, v_x)

        # Airspeed & apparent airflow angle in body-level frame
        a_x = v_x - w * np.cos(psi - zeta)
        a_y = v_y + w * np.sin(psi - zeta)
        a_mu = v_mu
        a = np.sqrt(a_x ** 2 + a_y ** 2)
        gamma = np.arctan2(a_y, a_x)

        # Velocity in body-level frame (adjusted for FLU)
        v_x_dot = (1 / m) * (u_thrust * np.cos(phi) * np.sin(theta) - C * a_x) + v_y * psi_dot
        v_y_dot = (1 / m) * (-u_thrust * np.sin(phi) - C * a_y) - v_x * psi_dot
        v_mu_dot = (1 / m) * (u_thrust * np.cos(phi) * np.cos(theta) - C * v_mu - m * g)*0.0

        q = np.sqrt(v_x_dot ** 2 + v_y_dot ** 2)
        alpha = np.arctan2(v_y_dot, v_x_dot)

        # Unwrap angles
        if np.array(phi).ndim > 0:
            if np.array(phi).shape[0] > 1:
                phi = np.unwrap(phi)
                theta = np.unwrap(theta)
                psi = np.unwrap(psi)
                beta = np.unwrap(beta)
                alpha = np.unwrap(alpha)

        # mocap measurements
        Px = x
        Py = y
        Pz = 1/mu
        P_cluster = np.array([Px, Py, Pz])
        Vx = v_x
        Vy = v_y
        Vz = v_mu
        V_cluster = np.array([Vx, Vy, Vz])
        Phi = phi
        Theta = theta
        Psi = psi
        Attitude_cluster = np.array([Phi, Theta, Psi])
        Mocap_cluster = np.array([Px, Py, Pz, Vx, Vy, Vz, Phi, Theta, Psi])
        # IMU measurements
        Omega_x = omega_x
        Omega_y = omega_y
        Omega_z = omega_z
        Omega_cluster = np.array([Omega_x, Omega_y, Omega_z])
        Ax = v_x_dot
        Ay = v_y_dot
        Az = v_mu_dot
        A_cluster = np.array([Ax, Ay, Az])
        IMU_cluster = np.array([Phi, Theta, Psi, Omega_x, Omega_y, Omega_z, Ax, Ay, Az])
        # optical flow measurements
        OF_x = v_x*mu
        OF_y = v_y*mu
        OF_XY = np.array([OF_x, OF_y])
        OF_z = v_mu*mu
        OF_cluster = np.array([OF_x, OF_y, OF_z])
        OF_cluster_XY = np.array([OF_x, OF_y])
        # wind measurements
        Awx = a_x
        Awy = a_y
        Awz = a_mu
        Aa = a
        Agamma = gamma
        Wind_XY = np.array([Awx, Awy])
        Wind_cluster = np.array([Awx, Awy, Awz, Aa, Agamma])
        Wind_cluster_XY = np.array([Awx, Awy,Aa, Agamma])

        y_all=np.atleast_2d(np.hstack((P_cluster, V_cluster, Attitude_cluster, Omega_cluster, A_cluster, OF_cluster, Wind_cluster))).T
        y_real = np.atleast_2d(np.hstack((IMU_cluster, OF_cluster_XY, Wind_XY))).T
        Y = y_real
        return Y

    def z_function(self, X):
        x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, wx, wy  = X
        g = (v_x ** 2 + v_y ** 2) ** (1 / 2)  # ground speed magnitude
        beta = sp.atan(v_y / v_x)  # ground speed angle
        W = (wx ** 2 + wy ** 2) ** (1 / 2)  # wind speed magnitude
        zeta = sp.atan(wy / wx)  # wind speed angle
        z = [x, y, z, g, beta, v_z, phi, theta, psi, omega_x, omega_y, omega_z, W, zeta]
        return sp.Matrix(z)

    def simulate(self, x, u,DT):
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
        # y0 = self.h2(x0, u0)

        # Initialize
        t_sim = [0.0]
        x_sim = [x0]
        u_sim = [u0]
        y_sim = [y0]

        # Run simulation
        for k in range(0, u.shape[0] - 1):
            # Current inputs
            u_k = u.values[k, :]

            # New state
            x_k = self.f_c_car(x_sim[k], u_k, DT)
            # x_k = self.f_c_car2(x_sim[k], u_k, DT)

            # New measurement
            y_k = self.h(x_k, u_k)
            # y_k = self.h2(x_k, u_k)

            # Store
            t_sim.append((k + 1) * DT)
            x_sim.append(x_k)
            u_sim.append(u_k)
            y_sim.append(y_k)
        # print(np.shape(x_sim))
        x_sim = pd.DataFrame(np.vstack([np.array(x).reshape(-1) for x in x_sim]), columns=self.state_names)
        u_sim = pd.DataFrame(np.vstack([np.array(u).reshape(-1) for u in u_sim]), columns=self.input_names)
        y_sim = pd.DataFrame(np.vstack([np.array(y).reshape(-1) for y in y_sim]), columns=self.measurement_names)
        t_sim = np.hstack(t_sim)

        return t_sim, x_sim, u_sim, y_sim

class DroneSimulator(Simulator):
    def __init__(self,
                 dt=0.1,
                 mpc_horizon=10,
                 r_u=1e-2,
                 control_mode='velocity_body_level',
                 params: DroneParameters = None):
        self.dynamics = DroneModel()
        super().__init__(self.dynamics.f, self.dynamics.h, dt=dt, mpc_horizon=mpc_horizon,
                         state_names=self.dynamics.state_names,
                         input_names=self.dynamics.input_names,
                         measurement_names=self.dynamics.measurement_names)
        # super().__init__(self.dynamics.f2, self.dynamics.h2, dt=dt, mpc_horizon=mpc_horizon,
        #                  state_names=self.dynamics.state_names,
        #                  input_names=self.dynamics.input_names,
        #                  measurement_names=self.dynamics.measurement_names)

        # Set parameters
        self.params = params or DroneParameters()
        print('Drone parameters:', self.params)
        # Place limit on controls
        self.mpc.bounds['lower', '_u', 'PWM1'] = 0
        self.mpc.bounds['lower', '_u', 'PWM2'] = 0
        self.mpc.bounds['lower', '_u', 'PWM3'] = 0
        self.mpc.bounds['lower', '_u', 'PWM4'] = 0
        ##########################################
        # self.mpc.bounds['lower', '_u', 'phi'] = -np.pi / 4
        # self.mpc.bounds['lower', '_u', 'theta'] = -np.pi / 4
        # self.mpc.bounds['upper', '_u', 'phi'] = np.pi / 4
        # self.mpc.bounds['upper', '_u', 'theta'] = np.pi / 4
        ##########################################

        # Place limit on states
        self.mpc.bounds['lower', '_x', 'z'] = 0

        self.mpc.bounds['upper', '_x', 'phi'] = np.pi / 4
        self.mpc.bounds['upper', '_x', 'theta'] = np.pi / 4

        self.mpc.bounds['lower', '_x', 'phi'] = -np.pi / 4
        self.mpc.bounds['lower', '_x', 'theta'] = -np.pi / 4

        # Define cost function
        self.control_mode = control_mode
        if self.control_mode == 'velocity_body_level':
            cost = (1.0 * (self.model.x['v_x'] - self.model.tvp['v_x_set']) ** 2 +
                    1.0 * (self.model.x['v_y'] - self.model.tvp['v_y_set']) ** 2 +
                    1.0 * (self.model.x['z'] - self.model.tvp['z_set']) ** 2 +
                    1.0 * (self.model.x['psi'] - self.model.tvp['psi_set']) ** 2)

        elif self.control_mode == 'position_global':
            cost = (1.0 * (self.model.x['x'] - self.model.tvp['x_set']) ** 2 +
                    1.0 * (self.model.x['y'] - self.model.tvp['y_set']) ** 2 +
                    1.0 * (self.model.x['z'] - self.model.tvp['z_set']) ** 2 +
                    1.0 * (self.model.x['psi'] - self.model.tvp['psi_set']) ** 2)
        else:
            raise Exception('Control mode not available')

        # Set cost function
        self.mpc.set_objective(mterm=cost, lterm=cost)

        # Set input penalty: make this small for accurate state following
        self.mpc.set_rterm(PWM1=r_u, PWM2=r_u, PWM3=r_u, PWM4=r_u)

    def update_setpoint(self, x=None, y=None, v_x=None, v_y=None, psi=None, z=None, wx=None, wy=None):
        """ Set the set-point variables.
        """

        # Set time
        T = self.dt * (len(wx) - 1)
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
                v_x = 0.0 * np.ones_like(tsim)
                v_y = 0.0 * np.ones_like(tsim)

        else:
            raise Exception('Control mode not available')

        # Define the set-points to follow
        setpoint = {'x': x,
                    'z': z,
                    'y': y,
                    'v_x': v_x,
                    'v_y': v_y,
                    'v_z': 0.0 * np.ones_like(tsim),
                    'phi': 0.0 * np.ones_like(tsim),
                    'theta': 0.0 * np.ones_like(tsim),
                    'psi': psi,
                    'omega_x': 0.0 * np.ones_like(tsim),
                    'omega_y': 0.0 * np.ones_like(tsim),
                    'omega_z': 0.0 * np.ones_like(tsim),
                    'wx': wx,
                    'wy': wy,
                    # 'w': w,
                    # 'zeta': zeta,

                    # 'm': self.params.Mm * np.ones_like(tsim),
                    # 'I_x': self.params.I_x * np.ones_like(tsim),
                    # 'I_y': self.params.I_y * np.ones_like(tsim),
                    # 'I_z': self.params.I_z * np.ones_like(tsim),
                    # 'C': self.params.C * np.ones_like(tsim),
                    }

        # Update the simulator set-point
        self.update_dict(setpoint, name='setpoint')

    def plot_trajectory(self, start_index=0, dpi=200, size_radius=None):
        """ Plot the trajectory.
        """

        fig, ax = plt.subplots(1, 1, figsize=(3 * 1, 3 * 1), dpi=dpi)

        x = self.y['x'][start_index:]
        y = self.y['y'][start_index:]
        heading = self.y['psi'][start_index:]
        time = self.time[start_index:]

        if size_radius is None:
            size_radius = 0.06 * np.max(np.array([range_of_vals(x), range_of_vals(y)]))

        ff.plot_trajectory(x, y, heading,
                           color=time,
                           ax=ax,
                           size_radius=size_radius,
                           nskip=0)

        fifi.mpl_functions.adjust_spines(ax, [])


def range_of_vals(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)