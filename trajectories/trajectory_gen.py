# trajectory.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

class TrajectoryGenerator:
    """
    Generates a simple 2D trajectory: time series of velocities, headings, and altitude.
    """

    def __init__(self):
        print("TrajectoryGenerator initialized.")
        self.wx = 1.0  # default wx
        self.wy = 2.0  # default wy
        print(f"Default wx={self.wx}, wy={self.wy}")
        self.dt = 0.1
        

    def all_motifs(self, T = 11.0, t0 = 1.0, g0 = 1.0):
        """
        Generates all motifs for the trajectory.
        """

        self.fs = 1.0 / self.dt
        self.T = T
        self.t0 = t0
        self.g0 = g0
        # time vector
        self.tsim = np.arange(0, T + self.dt/2, step=self.dt)
        tsim       = self.tsim.copy()
        v_x        = self.g0 * np.ones_like(tsim)
        v_y        = np.zeros_like(tsim)
        psi        = -np.pi/4 * np.ones_like(tsim)
        psi_global = psi.copy()

        # -- body‐frame velocity changes --
        idx = int((0.5 + self.t0) * self.fs)
        v_x[idx:] -= 3*self.g0/4
        idx = int((3.0 + self.t0) * self.fs)
        v_x[idx:] += 3*self.g0/4

        # -- body‐frame heading changes --
        idx = int((4.0 + self.t0) * self.fs)
        psi[idx:] += np.pi/2
        idx = int((6.9 + self.t0) * self.fs)
        psi[idx:] += np.pi/1.3

        # add a little sinusoidal wiggle at t = 8 + t0
        idx = int((8.0 + self.t0) * self.fs)
        tsim_temp = tsim[idx:] - tsim[idx]
        psi_temp  = (
            - (np.pi/3) * np.sin(2*np.pi*0.6*tsim_temp + np.pi/2)
            + 0*(np.pi/6)*np.cos(2*np.pi*1.0*tsim_temp + np.pi/3)
        )
        psi_temp -= psi_temp[0]
        psi[idx:] += psi_temp
        psi[idx:] += np.pi/2  # final bump at same index

        # -- global velocities & heading shifts --
        x_dot =  v_x*np.cos(psi_global) + v_y*np.sin(psi_global)
        y_dot =  v_x*np.sin(psi_global) + v_y*np.cos(psi_global)

        idx = int((5.5 + self.t0) * self.fs)
        psi_global[idx:] -= np.pi/2
        idx = int((6.9 + self.t0) * self.fs)
        psi_global[idx:] += np.pi/2

        # recalc body‐frame from global if you like
        v_x =  x_dot*np.cos(psi_global) + y_dot*np.sin(psi_global)
        v_y =  y_dot*np.cos(psi_global) - x_dot*np.sin(psi_global)

        z = 2 * np.ones_like(v_x)

        # get x0_sim for states x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, wx, wy
        X0_sim = np.array([0, 0, z[0], v_x[0], v_y[0], 0, 0, 0, psi[0], 0, 0, 0, self.wx, self.wy])
        # Check if the absolute values of the last two elements are both < 1e-12
        if abs(X0_sim[-1]) < 1e-12 and abs(X0_sim[-2]) < 1e-12:
            print('hello')
            X0_sim[-2] = 1e-12
            X0_sim[-1] = 1e-12

        return tsim, self.dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0_sim
    def sine_wave_motif(self,
                        T=11.0,
                        t0=1.0,
                        g0=1.0,
                        amplitude=1.5,
                        frequency=0.25):
        """
        Sine-wave heading motif.
        Same inputs as all_motifs, plus:
          amplitude (rad): peak heading angle
          frequency (Hz):  oscillation rate of the sine wave
        """
        # 1) store basics & build time vector
        self.dt = self.dt
        self.fs = 1.0 / self.dt
        self.T = T
        self.t0 = t0
        self.g0 = g0
        self.tsim = np.arange(0, T + self.dt/2, step=self.dt)
        tsim = self.tsim.copy()

        # 2) body-frame speeds
        v_x = self.g0 * np.ones_like(tsim)
        v_y = np.zeros_like(tsim)

        # 3) sinusoidal heading about t0
        phase = 2 * np.pi * frequency * (tsim - t0)
        psi = amplitude * np.sin(phase)
        psi_global = psi.copy()

        # 4) global-frame velocity
        x_dot = v_x * np.cos(psi_global) + v_y * np.sin(psi_global)
        y_dot = v_x * np.sin(psi_global) + v_y * np.cos(psi_global)

        # 5) recalc body-frame in case you need it
        v_x = x_dot * np.cos(psi_global) + y_dot * np.sin(psi_global)
        v_y = y_dot * np.cos(psi_global) - x_dot * np.sin(psi_global)

        # 6) altitude & initial state
        z = 2 * np.ones_like(v_x)
        X0_sim = np.array([
            0, 0, z[0],       # x,y,z
            v_x[0], v_y[0],   # vx, vy
            0, 0, 0,          # vz, phi, theta
            psi[0],           # psi
            0, 0, 0,          # omega_x, omega_y, omega_z
            self.wx, self.wy  # wx, wy
        ])

        # avoid exact zero at tail of X0_sim
        if abs(X0_sim[-1]) < 1e-12 and abs(X0_sim[-2]) < 1e-12:
            X0_sim[-2] = 1e-12
            X0_sim[-1] = 1e-12

        # 7) return same signature as all_motifs
        return tsim, self.dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0_sim

    def sine_wave_motif_constant_heading(self,
                            T=11.0,
                            t0=1.0,
                            g0=1.0,
                            amplitude=1.5,
                            frequency=0.25,
                            psi0=0.0):
        """
        Sine-wave spatial trajectory, but with constant heading.
        The body-frame velocities are time-varying to achieve the same path.
        """
        # 1) store basics & build time vector
        self.dt = self.dt
        self.fs = 1.0 / self.dt
        self.T = T
        self.t0 = t0
        self.g0 = g0
        self.tsim = np.arange(0, T + self.dt/2, step=self.dt)
        tsim = self.tsim.copy()

        # 2) Compute the desired global trajectory as before
        v_x_body_orig = self.g0 * np.ones_like(tsim)
        v_y_body_orig = np.zeros_like(tsim)
        phase = 2 * np.pi * frequency * (tsim - t0)
        psi_traj = amplitude * np.sin(phase)  # original oscillating heading

        # 3) Compute global velocities from the original trajectory
        x_dot = v_x_body_orig * np.cos(psi_traj) + v_y_body_orig * np.sin(psi_traj)
        y_dot = v_x_body_orig * np.sin(psi_traj) + v_y_body_orig * np.cos(psi_traj)

        # 4) Set constant heading
        psi = psi0 * np.ones_like(tsim)
        psi_global = psi.copy()

        # 5) Compute required body-frame velocities to follow the same global trajectory
        # [v_x; v_y] = R(-psi) * [x_dot; y_dot]
        v_x = x_dot * np.cos(psi0) + y_dot * np.sin(psi0)
        v_y = -x_dot * np.sin(psi0) + y_dot * np.cos(psi0)

        # 6) altitude & initial state
        z = 2 * np.ones_like(v_x)
        X0_sim = np.array([
            0, 0, z[0],       # x,y,z
            v_x[0], v_y[0],   # vx, vy
            0, 0, 0,          # vz, phi, theta
            psi0,             # psi (constant)
            0, 0, 0,          # omega_x, omega_y, omega_z
            self.wx, self.wy  # wx, wy
        ])

        # avoid exact zero at tail of X0_sim
        if abs(X0_sim[-1]) < 1e-12 and abs(X0_sim[-2]) < 1e-12:
            X0_sim[-2] = 1e-12
            X0_sim[-1] = 1e-12

        return tsim, self.dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0_sim

    def constant_accel_motif(self,
                               T=11.0,
                               accel=0.5,
                               v_min=1.0,
                               v_max=2.0,
                               psi0=0.0):
        """
        Constant acceleration cyclic motif: ramp speed between v_min and v_max
        at fixed accel, repeating as needed for total duration.

        Args:
            dt (float):   time step [s]
            T (float):    total simulation time [s]
            accel (float): constant acceleration magnitude [m/s^2]
            v_min (float): minimum speed [m/s]
            v_max (float): maximum speed [m/s]
            psi0 (float): constant heading [rad]

        Returns:
            tsim, self.dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0_sim
        """
        # store parameters
        self.dt = self.dt
        self.fs = 1.0 / self.dt
        self.T = T

        # time vector
        self.tsim = np.arange(0, T + self.dt/2, step=self.dt)
        tsim = self.tsim.copy()

        # compute period for up/down ramp
        up_time = (v_max - v_min) / accel
        period = 2 * up_time

        # phase within each period
        t_mod = np.mod(tsim, period)

        # velocity profile: up then down
        v_x = np.where(
            t_mod <= up_time,
            v_min + accel * t_mod,
            v_max - accel * (t_mod - up_time)
        )
        v_y = np.zeros_like(tsim)

        # constant heading
        psi = psi0 * np.ones_like(tsim)
        psi_global = psi.copy()

        # global frame velocities
        x_dot = v_x * np.cos(psi_global) + v_y * np.sin(psi_global)
        y_dot = v_x * np.sin(psi_global) + v_y * np.cos(psi_global)

        # recalc body-frame (optional)
        v_x = x_dot * np.cos(psi_global) + y_dot * np.sin(psi_global)
        v_y = y_dot * np.cos(psi_global) - x_dot * np.sin(psi_global)

        # constant altitude
        z = 2 * np.ones_like(v_x)

        # initial state vector
        X0_sim = np.array([
            0, 0, z[0],        # x, y, z
            v_x[0], v_y[0],     # vx, vy
            0, 0,               # vz, phi
            0,                  # theta
            psi0,               # psi
            0, 0, 0,            # omega_x, omega_y, omega_z
            self.wx, self.wy     # wx, wy
        ])

        # avoid exact zeros at end
        if abs(X0_sim[-2]) < 1e-12 and abs(X0_sim[-1]) < 1e-12:
            X0_sim[-2] = 1e-12
            X0_sim[-1] = 1e-12

        return tsim, self.dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0_sim

    def constant_velocity_motif(self,
                                T=11.0,
                                v=1.0,
                                psi0=0.0):
        """
        Constant velocity motif: move at constant speed v along heading psi0.
        """
        self.dt = self.dt
        self.fs = 1.0 / self.dt
        self.T = T

        self.tsim = np.arange(0, T + self.dt/2, step=self.dt)
        tsim = self.tsim.copy()

        # body-frame velocities
        v_x = v * np.ones_like(tsim)
        v_y = np.zeros_like(tsim)

        # heading
        psi = psi0 * np.ones_like(tsim)
        psi_global = psi.copy()

        # global velocities
        x_dot = v_x * np.cos(psi_global) + v_y * np.sin(psi_global)
        y_dot = v_x * np.sin(psi_global) + v_y * np.cos(psi_global)

        # recalc body-frame (optional)
        v_x = x_dot * np.cos(psi_global) + y_dot * np.sin(psi_global)
        v_y = y_dot * np.cos(psi_global) - x_dot * np.sin(psi_global)

        # altitude
        z = 2 * np.ones_like(v_x)

        # initial state
        X0_sim = np.array([
            0, 0, z[0],       # x, y, z
            v_x[0], v_y[0],   # vx, vy
            0, 0,             # vz, phi
            0,                # theta
            psi0,             # psi
            0, 0, 0,          # omega_x, omega_y, omega_z
            self.wx, self.wy  # wx, wy
        ])
        if abs(X0_sim[-2]) < 1e-12 and abs(X0_sim[-1]) < 1e-12:
            X0_sim[-2] = 1e-12
            X0_sim[-1] = 1e-12

        return tsim, self.dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0_sim

    def constant_velocity_yaw_motif(self,
                                    T=11.0,
                                    v=1.0,
                                    psi0=0.0,
                                    yaw_rate=0.75*2):
        """
        Constant global velocity with constant yaw rate:
        - moves at constant global speed `v` in direction `psi0`,
        - yaw (body/global orientation) spins at `yaw_rate` [rad/s].
        """
        # store parameters
        self.dt = self.dt
        self.fs = 1.0 / self.dt
        self.T = T

        # time vector
        self.tsim = np.arange(0, T + self.dt/2, step=self.dt)
        tsim = self.tsim.copy()

        # desired constant global velocity components
        x_dot = v * np.cos(psi0)
        y_dot = v * np.sin(psi0)

        # global yaw over time
        psi_global = psi0 + yaw_rate * tsim
        psi = psi_global.copy()

        # body-frame velocities: rotate global vel into body frame
        # [v_x; v_y] = R(-psi_global) * [x_dot; y_dot]
        v_x =  x_dot * np.cos(psi_global) + y_dot * np.sin(psi_global)
        v_y = -x_dot * np.sin(psi_global) + y_dot * np.cos(psi_global)

        # altitude
        z = 2 * np.ones_like(v_x)

        # initial state vector
        X0_sim = np.array([
            0,         # x0
            0,         # y0
            z[0],      # z0
            v_x[0],    # vx0
            v_y[0],    # vy0
            0,         # vz0
            0, 0,      # phi, theta
            psi0,      # psi0
            0, 0, 0,   # omega_x, omega_y, omega_z
            self.wx, self.wy  # wx, wy
        ])
        # avoid exact zeros at tail
        if abs(X0_sim[-2]) < 1e-12 and abs(X0_sim[-1]) < 1e-12:
            X0_sim[-2] = 1e-12
            X0_sim[-1] = 1e-12

        # return same signature
        return tsim, self.dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0_sim


    def motifs(self, motif='all', **kwargs):
        """
        Dispatch and remember.
        """
        mapping = {
            'all':  self.all_motifs,
            'sine': self.sine_wave_motif,
            'sine_constant_heading': self.sine_wave_motif_constant_heading,
            'constant_accel': self.constant_accel_motif,
            'constant_velocity': self.constant_velocity_motif,
            'constant_velocity_yaw': self.constant_velocity_yaw_motif
        }
        if motif not in mapping:
            raise ValueError(f"Unknown motif '{motif}'")
        # store for next plot
        self._last_motif = motif
        self._last_kwargs = kwargs
        return mapping[motif](**kwargs)

    def plot_trajectory(self,
                    motif=None,
                    marker_step=2,
                    scale=2.0,
                    **motif_kwargs):
        """
        Plots using either the passed-in motif/motif_kwargs,
        or the last-used ones if you leave them out.
        """
        import matplotlib as mpl

        # decide which motif + kwargs to use
        motif_to_call = motif or self._last_motif
        kwargs_to_call = motif_kwargs or self._last_kwargs

        # generate data
        tsim, dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0 = (
            self.motifs(motif_to_call, **kwargs_to_call)
        )

        # compute global positions
        v_xg = v_x*np.cos(psi) - v_y*np.sin(psi)
        v_yg = v_x*np.sin(psi) + v_y*np.cos(psi)
        x0, y0 = X0[0], X0[1]
        x = x0 + cumtrapz(v_xg, tsim, initial=0)
        y = y0 + cumtrapz(v_yg, tsim, initial=0)

        # plot path
        plt.figure(figsize=(10,5))
        plt.plot(x, y, '-k', label=f"motif='{motif_to_call}'")
        plt.axis('equal'); plt.grid()
        plt.xlabel('X (m)'); plt.ylabel('Y (m)')
        plt.title(f"Trajectory: {motif_to_call}")

        # Set up colormap
        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=tsim[0], vmax=tsim[-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # overlay triangles with color by time
        tri = np.array([[.1,0],[-.05,-.05],[-.05,.05]]) * scale
        for i in range(0, len(tsim), marker_step):
            R = np.array([[np.cos(psi[i]), -np.sin(psi[i])],
                        [np.sin(psi[i]),  np.cos(psi[i])]])
            tri_r = tri @ R.T
            color = cmap(norm(tsim[i]))
            plt.fill(x[i]+tri_r[:,0],
                    y[i]+tri_r[:,1],
                    color=color, alpha=0.8, edgecolor=None)

        plt.legend()
        plt.colorbar(sm, label='Time [s]')
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        # plt.xlim(x_min - 1 , x_max + 1)
        # plt.ylim(y_min - 1 , y_max + 1)
        plt.show()

