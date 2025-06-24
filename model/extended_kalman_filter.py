import numpy as np
import sys
sys.path.append('../util')
import utils



class EKF:
    def __init__(self, f, h, x0, u0,
                 F_jacobian=None, H_jacobian=None,
                 P=None, Q=None, R=None,
                 circular_measurements=None):
        """
        Initialize Extended Kalman Filter (EKF).

        :param callable f: state transition function, f(x, u)
        :param callable h: measurement function, h(x, u)
        :param callable F_jacobian: function returning the Jacobian of f w.r.t x, F_jacobian(x, u)
        :param callable H_jacobian: function returning the Jacobian of h w.r.t x, H_jacobian(x, u)
        :param np.ndarray x0: initial guess of x
        :param np.ndarray u0: initial inputs
        :param np.ndarray | None Q: optional default process noise covariance matrix
        :param np.ndarray | None R: optional default measurement noise covariance matrix
        :param np.ndarray | None P: optional default process noise covariance matrix
        :param tuple | list | np.ndarray | None circular_measurements: optional iterable of bools to indicate
        what measurements are circular variables
        """

        # Store state transition & measurement functions
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian

        # Store initial state & input vectors
        self.x0 = x0
        self.u0 = u0

        # Run f & h, make sure they work & get sizes
        self.x0 = self.f(self.x0, self.u0)
        self.z0 = self.h(self.x0, self.u0)

        self.n = self.x0.squeeze().shape[0]  # number of states
        self.p = self.z0.squeeze().shape[0]  # number of measurements

        # Set what variables are circular
        if circular_measurements is None:  # default is to assume no variables are circular
            self.circular_measurements = tuple(np.zeros(self.n))
        else:
            self.circular_measurements = tuple(circular_measurements)

        # Set noise covariances
        if P is None:
            self.P = np.eye(self.n)
        else:
            self.P = P

        if Q is None:
            self.Q = np.eye(self.n)
        else:
            self.Q = Q

        if R is None:
            self.R = np.eye(self.p)
        else:
            self.R = R

        # Store state & covariance history
        self.history = {'X': [self.x0],
                        'U': [self.u0],
                        'Z': [self.z0],
                        'P': [self.P],
                        'R': [self.R],
                        'Q': [self.Q]
                        }

        # Current state, inputs, & measurements
        self.x = self.x0.copy()
        self.u = self.u0.copy()
        self.z = self.z0.copy()

        # Current timestep
        self.k = 0

    def predict(self, u, Q=None):
        """
        EKF prediction step.

        :param u: input vector
        :param Q: optional process noise covariance matrix for this step
        """

        # Update controls
        self.u = u.copy()

        # Set process noise covariance
        if Q is not None:
            self.Q = Q

        # Predict next state
        self.x = self.f(self.x, self.u)

        # Use jacobian of state transition function to predict state estimate covariance
        F = self.F_jacobian(self.x, self.u)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, R=None):
        """
        EKF update step with measurement z.

        :param z: measurement vector
        :param R: optional measurement noise covariance matrix for this step
        """

        # Current measurement
        self.z = np.array(z).copy()

        # Set measurement noise covariance
        if R is not None:
            self.R = R

        # Predicted measurements from state estimate
        z_pred = self.h(self.x, self.u)

        # Compute innovation (measurement residual)
        # y = self.z - z_pred
        y = np.zeros(self.p)
        for j in range(self.p):
            if self.circular_measurements[j]:  # circular measurement
                # y[j] = angle_difference(z[j], z_pred[j])
                # y[j] = angle_difference(self.z[j], z_pred[j])
                y[j] = util.wrapToPi(np.array(self.z[j] - z_pred[j]))
            else:  # non-circular measurement
                y[j] = self.z[j] - z_pred[j]

        # Use jacobian of measurement function to compute the innovation/residual covariance
        H = self.H_jacobian(self.x, self.u)
        S = H @ self.P @ H.T + self.R

        # Near-optimal Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update state covariance estimate
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
        # self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

        # Update history
        self.history['X'].append(self.x.copy())
        self.history['U'].append(self.u.copy())
        self.history['Z'].append(self.z.copy())
        self.history['P'].append(self.P.copy())
        self.history['Q'].append(self.Q.copy())
        self.history['R'].append(self.R.copy())

        # If it's the 1st time-step, set the initial values
        if self.k == 0:
            self.history['Z'][0] = self.z.copy()
            self.history['P'][0] = self.P.copy()
            self.history['Q'][0] = self.Q.copy()
            self.history['R'][0] = self.R.copy()

        # Update time-step
        self.k += 1


def jacobian_numerical(f, x, u, epsilon=1e-6):
    """
    Approximate the Jacobian of a function f at the point x using finite differences.

    :param callable f: function for which to compute the Jacobian
    :param np.ndarray x: point at which to evaluate the Jacobian
    :param float epsilon: perturbation value for finite differences
    :return np.ndarray: Jacobian matrix
    """

    n = len(x)  # number of function inputs
    m = len(f(x, u))  # number of function outputs

    # Jacobian
    jacobian = np.zeros((m, n))

    for i in range(n):
        # Perturb x[i] with epsilon in the positive direction
        x_plus = x.copy()
        x_plus[i] = x_plus[i] + epsilon

        # Perturb x[i] with epsilon in the negative direction
        x_minus = x.copy()
        x_minus[i] = x_minus[i] - epsilon

        # Evaluate the function at the perturbed point
        jacobian[:, i] = (f(x_plus, u) - f(x_minus, u)) / (2.0 * epsilon)

    return jacobian


def rk4_discretize(f, x, u, dt):
    """
    Discretizes the continuous-time dynamics using the Runge-Kutta 4th order method (RK4).

    :param f: Function that defines the system dynamics (dx/dt = f(x, u))
              f should accept the current state `x` and input `u` and return the state derivatives.
    :param x: Current state (numpy array), representing the state at time t
    :param u: Control input (numpy array), control applied at time t
    :param dt: Time step (float), the discretization time step

    :return: Discretized state at time t+dt (numpy array)
    """

    # Step 1: Compute k1, the first estimate of the state change (function evaluation at time t)
    k1 = f(x, u)  # k1 is the rate of change at the current state

    # Step 2: Compute k2, estimate of state change at time t + dt/2, based on k1
    # Perturb x by half the step size (dt/2) in the direction of k1
    k2 = f(x + 0.5 * dt * k1, u)  # k2 is the rate of change at t + dt/2

    # Step 3: Compute k3, another estimate of state change at time t + dt/2, based on k2
    # Perturb x by half the step size (dt/2) in the direction of k2
    k3 = f(x + 0.5 * dt * k2, u)  # k3 is the rate of change at t + dt/2 (but using k2)

    # Step 4: Compute k4, estimate of state change at time t + dt, based on k3
    # Perturb x by the full time step (dt) in the direction of k3
    k4 = f(x + dt * k3, u)  # k4 is the rate of change at t + dt

    # Step 5: Compute the weighted sum of the estimates (k1, k2, k3, k4) to update x
    # The final estimate is a weighted average of all k's
    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next


def angle_difference(target, current):
    """
    Computes the signed minimal angular difference (in radians)
    from current to target, in the range [-π, π).
    """
    diff = (target - current + np.pi) % (2 * np.pi) - np.pi
    return diff
