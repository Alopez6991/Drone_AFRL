import numpy as np

class TrajectoryGenerator:
    def __init__(self, target_height, dt, heading=False, Wx=0.0, Wy=0.0, Wz=0.0):
        """
        Initialize the trajectory generator with the target height for take-off, time step, 
        heading calculation, and wind velocities.
        
        Parameters:
        - target_height: float, the target height for take-off.
        - dt: float, the time step for trajectory generation.
        - heading: bool, whether to calculate the drone's heading (yaw) or keep it at 0.
        - Wx, Wy, Wz: floats, the constant wind velocities in the x, y, and z directions.
        """
        self.target_height = target_height
        self.dt = dt
        self.heading = heading
        self.Wx = Wx
        self.Wy = Wy
        self.Wz = Wz
    
    def generate_trajectory(self, mode, **kwargs):
        """
        Generate a trajectory based on the specified mode: constant, accelerating, or circular.
        All trajectories start with a take-off using the exponential function.

        Parameters:
        - mode: str, specifies the type of trajectory ('constant', 'acc', or 'circ').
        - **kwargs: additional parameters specific to the trajectory type.

        Returns:
        - trajectory: Generated trajectory (dict with vx, vy, vz, z, wx, wy, wz, psi, t).
        """

        # Take-off phase using exponential trajectory
        t_takeoff, z_takeoff, vz_takeoff = self.exponential_trajectory(kwargs.get('k', 1))
        vx_takeoff, vy_takeoff, vz_takeoff, z_takeoff, wx_takeoff, wy_takeoff, wz_takeoff, psi_takeoff, t_takeoff = self.get_take_off_targets(t_takeoff, z_takeoff, vz_takeoff)

        # After take-off, switch to the chosen trajectory mode
        if mode == 'constant':
            t_values, x_values, y_values = self.linear_motion_constant_velocity(
                num_points=kwargs['num_points'], velocity=kwargs['velocity']
            )
        elif mode == 'acc':
            t_values, x_values, y_values = self.accelerating_motion(
                num_points=kwargs['num_points'], acceleration=kwargs['acceleration']
            )
        elif mode == 'circ':
            # Circular trajectory with multiple arcs
            x_values, y_values = self.circular_flight_velocity(
                T=kwargs['T'], start_x=kwargs['start_x'], start_y=kwargs['start_y'], 
                r=kwargs['r'], arc_length=kwargs['arc_length'], num_circles=kwargs['num_circles']
            )
            # Generate time array based on the circular trajectory
            total_time = kwargs['T'] * kwargs['num_circles']
            t_values = np.arange(0, total_time, self.dt)[:len(x_values)]  # Adjust time array length to match
        else:
            raise ValueError("Invalid mode. Choose from 'constant', 'acc', or 'circ'.")

        # Check if lengths of time arrays match the trajectory
        num_takeoff_points = len(t_takeoff)
        num_main_points = len(t_values)

        # Adjust the time array lengths
        t_values_adjusted = t_values[:min(num_main_points, len(x_values))]
        x_values_adjusted = x_values[:min(num_main_points, len(t_values))]
        y_values_adjusted = y_values[:min(num_main_points, len(t_values))]

        # Combine take-off and trajectory into a full trajectory
        t_total = np.concatenate([t_takeoff, t_values_adjusted + t_takeoff[-1]])
        z_total = np.concatenate([z_takeoff, np.full_like(t_values_adjusted, self.target_height)])  # Hold z constant
        vz_total = np.concatenate([vz_takeoff, np.full_like(t_values_adjusted, 0.0)])  # No vertical motion after take-off

        # Wind values
        wx_total = np.concatenate([wx_takeoff, np.full_like(t_values_adjusted, self.Wx)])
        wy_total = np.concatenate([wy_takeoff, np.full_like(t_values_adjusted, self.Wy)])
        wz_total = np.concatenate([wz_takeoff, np.full_like(t_values_adjusted, self.Wz)])

        # Yaw (heading) adjustment to match other lengths
        if self.heading:
            vx_total = np.concatenate([vx_takeoff, x_values_adjusted])
            vy_total = np.concatenate([vy_takeoff, y_values_adjusted])
            psi_total = np.arctan2(vy_total, vx_total)  # Ensure lengths match by using combined vx and vy arrays
        else:
            psi_total = np.concatenate([psi_takeoff, np.zeros_like(t_values_adjusted)])

        return {
            'vx': np.concatenate([vx_takeoff, x_values_adjusted]),
            'vy': np.concatenate([vy_takeoff, y_values_adjusted]),
            'vz': vz_total,
            'z': z_total,
            'wx': wx_total,
            'wy': wy_total,
            'wz': wz_total,
            'psi': psi_total,
            't': t_total
        }

    def exponential_trajectory(self, k=1):
        """
        Generate a smooth trajectory using an exponential function that reaches the target height.
        """
        z_values = []
        v_values = []
        t_values = []
        t = 0
        z = 0

        while z < self.target_height * 0.99:  # Stop when close to the target height
            z = self.target_height * (1 - np.exp(-k * t))
            v = k * self.target_height * np.exp(-k * t)  # Velocity is the derivative of the height function
            z_values.append(z)
            v_values.append(v)
            t_values.append(t)
            t += self.dt

        return np.array(t_values), np.array(z_values), np.array(v_values)

    def get_take_off_targets(self, t_values, z_values, v_values):
        """
        Get the take-off targets based on the exponential trajectory.
        """
        vx = np.full_like(t_values, 0.0)
        vy = np.full_like(t_values, 0.0)
        vz = v_values
        wx = np.full_like(t_values, self.Wx)
        wy = np.full_like(t_values, self.Wy)
        wz = np.full_like(t_values, self.Wz)
        z = z_values
        psi = np.full_like(t_values, 0.0)
        t = t_values

        return vx, vy, vz, z, wx, wy, wz, psi, t

    def linear_motion_constant_velocity(self, num_points, velocity):
        """
        Generate linear motion with constant velocity that alternates between (1, 10), (2, -10), etc.
        """
        x_total, y_total, t_total = [], [], []
        points = [(i, 10 if i % 2 == 1 else -10) for i in range(1, num_points + 1)]
        x_current, y_current, t_current = 0, 0, 0

        for point in points:
            x_target, y_target = point
            dx, dy = x_target - x_current, y_target - y_current
            distance = np.sqrt(dx**2 + dy**2)
            Vx, Vy = velocity * (dx / distance), velocity * (dy / distance)
            T = distance / velocity
            t_segment = np.arange(0, T, self.dt)
            x_segment = x_current + Vx * t_segment
            y_segment = y_current + Vy * t_segment
            t_total = np.concatenate([t_total, t_segment + t_current])
            x_total = np.concatenate([x_total, x_segment])
            y_total = np.concatenate([y_total, y_segment])
            x_current, y_current, t_current = x_target, y_target, t_current + T

        return t_total, x_total, y_total

    def accelerating_motion(self, num_points, acceleration):
        """
        Generate linear motion with constant acceleration that alternates between (1, 10), (2, -10), etc.
        """
        x_total, y_total, t_total = [], [], []
        points = [(i, 10 if i % 2 == 1 else -10) for i in range(1, num_points + 1)]
        x_current, y_current, t_current = 0, 0, 0

        for point in points:
            x_target, y_target = point
            dx, dy = x_target - x_current, y_target - y_current
            distance = np.sqrt(dx**2 + dy**2)
            Vx_initial, Vy_initial = 0, 0
            T = np.sqrt(2 * distance / acceleration)
            t_segment = np.arange(0, T, self.dt)
            x_segment = x_current + 0.5 * acceleration * (t_segment**2) * (dx / distance)
            y_segment = y_current + 0.5 * acceleration * (t_segment**2) * (dy / distance)
            t_total = np.concatenate([t_total, t_segment + t_current])
            x_total = np.concatenate([x_total, x_segment])
            y_total = np.concatenate([y_total, y_segment])
            x_current, y_current, t_current = x_target, y_target, t_current + T

        return t_total, x_total, y_total

    def circular_flight_velocity(self, T, start_x, start_y, r=1, arc_length=np.pi, num_circles=1):
        """
        Generates velocity arrays for Vx and Vy to fly in an arc of a circle.
        """
        full_Vx, full_Vy = [], []
        t = np.arange(0, T, self.dt)
        omega = arc_length / T

        for i in range(num_circles):
            if i % 2 == 0:
                Vx_start = -r * omega * np.sin(omega * t) + start_x
                Vy_start = r * omega * np.cos(omega * t) + start_y
                full_Vx.extend(Vx_start)
                full_Vy.extend(Vy_start)
            else:
                Vx_return = -r * omega * np.sin(omega * t) + start_x
                Vy_return = -r * omega * np.cos(omega * t) + start_y
                full_Vx.extend(Vx_return)
                full_Vy.extend(Vy_return)

        return np.array(full_Vx), np.array(full_Vy)

# Example usage:

# Create the TrajectoryGenerator class
traj_gen = TrajectoryGenerator(target_height=2, dt=0.1, heading=True, Wx=1.0, Wy=1.0, Wz=1.0)

# Constant velocity trajectory after take-off with alternating points and constant velocity
trajectory_constant = traj_gen.generate_trajectory('constant', num_points=5, velocity=5)
# print("Constant velocity motion trajectory:", trajectory_constant)

# Accelerating trajectory after take-off with alternating points and constant acceleration
trajectory_acc = traj_gen.generate_trajectory('acc', num_points=5, acceleration=10)
# print("Accelerating motion trajectory:", trajectory_acc)

# Circular trajectory after take-off with 5 arcs (start, return, start, return, start)
trajectory_circ = traj_gen.generate_trajectory('circ', T=10, start_x=0, start_y=0, r=5, arc_length=np.pi, num_circles=5)
# print("Circular motion trajectory with 5 arcs:", trajectory_circ)
# 