import numpy as np

class TrajectoryGenerator:
    def __init__(self, target_height, dt, params, Wx=0.0, Wy=0.0, Wz=0.0, heading=False):
        """
        Initialize the trajectory generator with the target height for take-off, time step, 
        heading calculation, and wind velocities.
        
        Parameters:
        - target_height: float, the target height for take-off.
        - dt: float, the time step for trajectory generation.
        - params: additional parameters for the trajectory.
        - Wx, Wy, Wz: wind velocities in x, y, and z directions.
        - heading: bool, whether to calculate the drone's heading (yaw) or keep it at 0.
        """
        self.target_height = target_height
        self.dt = dt
        self.heading = heading  # This flag will control heading across all methods
        self.Wx = Wx
        self.Wy = Wy
        self.Wz = Wz
        self.params = params

        # Storage for trajectory values
        self.Z_val = []
        self.Vx_val = []
        self.Vy_val = []
        self.Vz_val = []
        self.Wx_val = []
        self.Wy_val = []
        self.Wz_val = []
        self.T_val = []
        self.PSI_val = []

        # Generate the take-off trajectory during initialization
        self.take_off()

    def take_off(self, k=1):
        """
        Generate the initial takeoff trajectory using an exponential function.
        """
        target_height = self.target_height
        dt = self.dt

        # Initialize time and height values
        z_values = []
        v_values = []
        t_values = []

        # Initial conditions
        t = 0
        z = 0

        # Generate the takeoff trajectory
        while z < target_height * 0.99:
            z = target_height * (1 - np.exp(-k * t))
            v = k * target_height * np.exp(-k * t)

            z_values.append(z)
            v_values.append(v)
            t_values.append(t)

            t += dt

        # Store the takeoff values in the instance variables
        self.Z_val = np.array(z_values)
        self.Vx_val = np.zeros_like(self.Z_val)
        self.Vy_val = np.zeros_like(self.Z_val)
        self.Vz_val = np.array(v_values)
        self.Wx_val = np.full_like(self.Z_val, self.Wx)
        self.Wy_val = np.full_like(self.Z_val, self.Wy)
        self.Wz_val = np.full_like(self.Z_val, self.Wz)
        self.T_val = np.array(t_values)

        if self.heading:
            self.PSI_val = np.arctan2(self.Wy_val, self.Wx_val)
        else:
            self.PSI_val = np.zeros_like(self.Z_val)

    def get_constant_vel(self, V_mag, T, N, Step_V=False):
        """
        Generate a constant velocity profile for the drone after takeoff and concatenate
        it with the takeoff trajectory.
        
        Parameters:
        - V_mag: The magnitude of the velocity.
        - T: Time interval between velocity changes.
        - N: Number of velocity changes.
        - Step_V: If True, velocity changes in steps; otherwise, just the direction changes.
        """
        Vx_values = []
        Vy_values = []
        Vz_values = []
        t_values = []
        psi_values = []
        Wx_values = []
        Wy_values = []
        Wz_values = []
        z_values = []

        # Get the takeoff end time and last z value from the takeoff trajectory
        take_off_end_time = self.T_val[-1] if len(self.T_val) > 0 else 0
        last_z = self.Z_val[-1] if len(self.Z_val) > 0 else 0  # Get the last z from takeoff
        t_out = take_off_end_time
        time_array = np.arange(0, T, self.dt)

        V_mag_step = V_mag

        for i in range(1, N + 1):
            for t in time_array:
                if Step_V:
                    if i % 2 == 0:
                        Vx = V_mag_step * np.sin(np.pi / 8)
                        Vy = V_mag_step * np.cos(np.pi / 8)
                    else:
                        Vx = V_mag_step * np.sin(np.pi - np.pi / 8)
                        Vy = V_mag_step * np.cos(np.pi - np.pi / 8)
                else:
                    if i % 2 == 0:
                        Vx = V_mag * np.sin(np.pi / 8)
                        Vy = V_mag * np.cos(np.pi / 8)
                    else:
                        Vx = V_mag * np.sin(np.pi - np.pi / 8)
                        Vy = V_mag * np.cos(np.pi - np.pi / 8)

                Vx_values.append(Vx)
                Vy_values.append(Vy)
                Vz_values.append(0.0)
                t_values.append(t_out)
                t_out += self.dt

            V_mag_step += 0.2

        # Calculate heading if needed
        if self.heading:
            psi_values = np.arctan2(Vy_values, Vx_values)
        else:
            psi_values = np.zeros_like(t_values)

        # Create constant wind values
        Wx_values = np.full_like(t_values, self.Wx)
        Wy_values = np.full_like(t_values, self.Wy)
        Wz_values = np.full_like(t_values, self.Wz)
        z_values = np.full_like(t_values, last_z)  # Use last z from takeoff

        # Concatenate new values with the takeoff trajectory
        self.Vx_val = np.concatenate((self.Vx_val, Vx_values))
        self.Vy_val = np.concatenate((self.Vy_val, Vy_values))
        self.Vz_val = np.concatenate((self.Vz_val, Vz_values))
        self.Wx_val = np.concatenate((self.Wx_val, Wx_values))
        self.Wy_val = np.concatenate((self.Wy_val, Wy_values))
        self.Wz_val = np.concatenate((self.Wz_val, Wz_values))
        self.Z_val = np.concatenate((self.Z_val, z_values))
        self.T_val = np.concatenate((self.T_val, t_values))
        self.PSI_val = np.concatenate((self.PSI_val, psi_values))


    def get_Acc(self, Acc, T, N, Step_V=False):
        """
        Generate an acceleration profile for the drone after takeoff and concatenate it 
        with the takeoff trajectory.
        
        Parameters:
        - Acc: The magnitude of the acceleration.
        - T: Time interval between velocity changes.
        - N: Number of velocity changes.
        - Step_V: If True, velocity changes in steps; otherwise, just the direction changes.
        """
        Vx_values = []
        Vy_values = []
        Vz_values = []
        t_values = []
        psi_values = []
        Wx_values = []
        Wy_values = []
        Wz_values = []
        z_values = []

        # Get the takeoff end time and last z value from the takeoff trajectory
        take_off_end_time = self.T_val[-1] if len(self.T_val) > 0 else 0
        last_z = self.Z_val[-1] if len(self.Z_val) > 0 else 0  # Get the last z from takeoff
        t_out = take_off_end_time
        time_array = np.arange(0, T, self.dt)

        Acc_step = Acc

        for i in range(1, N + 1):
            for t in time_array:
                if t <= time_array[-1] / 2:
                    # Accelerating phase
                    if Step_V:
                        if i % 2 == 0:
                            Vx = Acc_step * t * np.sin(np.pi / 8)
                            Vy = Acc_step * t * np.cos(np.pi / 8)
                        else:
                            Vx = Acc_step * t * np.sin(np.pi - np.pi / 8)
                            Vy = Acc_step * t * np.cos(np.pi - np.pi / 8)
                    else:
                        if i % 2 == 0:
                            Vx = Acc * t * np.sin(np.pi / 8)
                            Vy = Acc * t * np.cos(np.pi / 8)
                        else:
                            Vx = Acc * t * np.sin(np.pi - np.pi / 8)
                            Vy = Acc * t * np.cos(np.pi - np.pi / 8)

                    Vz = 0.0

                    # Append values
                    Vx_values.append(Vx)
                    Vy_values.append(Vy)
                    Vz_values.append(Vz)
                    t_values.append(t_out)
                    t_out += self.dt

                    if t == time_array[-2] / 2:
                        Vx_mid = Vx
                        Vy_mid = Vy

                else:
                    # Decelerating phase
                    t_decel = t - time_array[-1] / 2

                    if Step_V:
                        if i % 2 == 0:
                            Vx = Vx_mid - Acc_step * t_decel * np.sin(np.pi / 8)
                            Vy = Vy_mid - Acc_step * t_decel * np.cos(np.pi / 8)
                        else:
                            Vx = Vx_mid - Acc_step * t_decel * np.sin(np.pi - np.pi / 8)
                            Vy = Vy_mid - Acc_step * t_decel * np.cos(np.pi - np.pi / 8)
                    else:
                        if i % 2 == 0:
                            Vx = Vx_mid - Acc * t_decel * np.sin(np.pi / 8)
                            Vy = Vy_mid - Acc * t_decel * np.cos(np.pi / 8)
                        else:
                            Vx = Vx_mid - Acc * t_decel * np.sin(np.pi - np.pi / 8)
                            Vy = Vy_mid - Acc * t_decel * np.cos(np.pi - np.pi / 8)

                    Vz = 0.0

                    # Append values
                    Vx_values.append(Vx)
                    Vy_values.append(Vy)
                    Vz_values.append(Vz)
                    t_values.append(t_out)
                    t_out += self.dt

            Acc_step += 0.2

        # Calculate heading if needed
        if self.heading:
            psi_values = np.arctan2(Vy_values, Vx_values)
        else:
            psi_values = np.zeros_like(t_values)

        Wx_values = np.full_like(t_values, self.Wx)
        Wy_values = np.full_like(t_values, self.Wy)
        Wz_values = np.full_like(t_values, self.Wz)
        z_values = np.full_like(t_values, last_z)  # Use last z from takeoff

        # Concatenate new values with the takeoff trajectory
        self.Vx_val = np.concatenate((self.Vx_val, Vx_values))
        self.Vy_val = np.concatenate((self.Vy_val, Vy_values))
        self.Vz_val = np.concatenate((self.Vz_val, Vz_values))
        self.Wx_val = np.concatenate((self.Wx_val, Wx_values))
        self.Wy_val = np.concatenate((self.Wy_val, Wy_values))
        self.Wz_val = np.concatenate((self.Wz_val, Wz_values))
        self.Z_val = np.concatenate((self.Z_val, z_values))
        self.T_val = np.concatenate((self.T_val, t_values))
        self.PSI_val = np.concatenate((self.PSI_val, psi_values))


        # Circular flight velocity generator (start arc)
    def circular_flight_velocity_start(self, T, dt, r=1, arc_length=np.pi):
        """
        Generates velocity arrays for Vx and Vy to fly in an arc of a circle.
        
        Parameters:
        T (float): Total time to complete the defined arc.
        dt (float): Time step.
        r (float): Radius of the circle. Default is 1.
        arc_length (float): The total radians to travel. Default is pi (half circle).
        
        Returns:
        tuple: (Vx, Vy) velocity arrays.
        """
        # Time array
        t = np.arange(0, T, dt)
        
        # Angular velocity
        omega = arc_length / T
        
        # Velocity equations
        Vx = -r * omega * np.sin(omega * t)
        Vy = r * omega * np.cos(omega * t)
        
        return Vx, Vy

    # Circular flight velocity generator (return arc)
    def circular_flight_velocity_return(self, T, dt, r=1, arc_length=np.pi):
        """
        Generates velocity arrays for Vx and Vy to fly back in an arc of a circle.
        
        Parameters:
        T (float): Total time to complete the defined arc.
        dt (float): Time step.
        r (float): Radius of the circle. Default is 1.
        arc_length (float): The total radians to travel. Default is pi (half circle).
        
        Returns:
        tuple: (Vx, Vy) velocity arrays.
        """
        # Time array
        t = np.arange(0, T, dt)
        
        # Angular velocity
        omega = arc_length / T
        
        # Velocity equations
        Vx = -r * omega * np.sin(omega * t)
        Vy = -r * omega * np.cos(omega * t)
        
        return Vx, Vy

    def get_Turns(self, r, T, N, R_step=False, arc_length=np.pi):
        """
        Generate a turning trajectory for the drone after takeoff and concatenate it 
        with the takeoff trajectory.
        
        Parameters:
        - r: Radius of the turn.
        - T: Time interval between direction changes.
        - N: Number of turns.
        - R_step: If True, the radius increases with each turn. Default is False.
        - arc_length: The arc angle (in radians) for the turn. Default is pi (half-circle).
        """
        Vx_values = []
        Vy_values = []
        Vz_values = []
        psi_values = []
        t_values = []

        # Get the takeoff end time and last z value from the takeoff trajectory
        take_off_end_time = self.T_val[-1] if len(self.T_val) > 0 else 0
        last_z = self.Z_val[-1] if len(self.Z_val) > 0 else 0  # Get the last z from takeoff
        t_out = take_off_end_time  # Start time for turns is the end of takeoff time
        
        r_step = r

        for i in range(1, N + 1):
            if R_step:
                if i % 2 == 0:
                    Vx, Vy = self.circular_flight_velocity_start(T, self.dt, r_step, arc_length)
                else:
                    Vx, Vy = self.circular_flight_velocity_return(T, self.dt, r_step, arc_length)
                # Append or concatenate the values
                Vx_values.extend(Vx)
                Vy_values.extend(Vy)
                Vz_values.extend([0.0] * len(Vx))  # Assuming Vz is constant, fill with zeros
            else:
                if i % 2 == 0:
                    Vx, Vy = self.circular_flight_velocity_start(T, self.dt, r, arc_length)
                else:
                    Vx, Vy = self.circular_flight_velocity_return(T, self.dt, r, arc_length)
                    # Append or concatenate the values
                Vx_values.extend(Vx)
                Vy_values.extend(Vy)
                Vz_values.extend([0.0] * len(Vx))  # Assuming Vz is constant, fill with zeros

            # Generate a time array for this segment and add it to t_values
            t_segment = np.arange(t_out, t_out + len(Vx) * self.dt, self.dt)
            if len(t_segment) > len(Vx):  # Trim if t_segment is longer due to rounding
                t_segment = t_segment[:len(Vx)]
            t_values.extend(t_segment)
            t_out += len(Vx) * self.dt  # Update t_out to continue for the next turn

            # Increment radius if needed
            if R_step:
                r_step += 0.2

        # Calculate heading if needed
        if self.heading:
            psi_values = np.arctan2(Vy_values, Vx_values)
        else:
            psi_values = np.zeros_like(Vx_values)

        Wx_values = np.full_like(t_values, self.Wx)
        Wy_values = np.full_like(t_values, self.Wy)
        Wz_values = np.full_like(t_values, self.Wz)
        z_values = np.full_like(t_values, last_z)

        # Concatenate new values with the takeoff trajectory
        self.Vx_val = np.concatenate((self.Vx_val, Vx_values))
        self.Vy_val = np.concatenate((self.Vy_val, Vy_values))
        self.Vz_val = np.concatenate((self.Vz_val, Vz_values))
        self.Wx_val = np.concatenate((self.Wx_val, Wx_values))
        self.Wy_val = np.concatenate((self.Wy_val, Wy_values))
        self.Wz_val = np.concatenate((self.Wz_val, Wz_values))
        self.Z_val = np.concatenate((self.Z_val, z_values))
        self.T_val = np.concatenate((self.T_val, t_values))
        self.PSI_val = np.concatenate((self.PSI_val, psi_values))


    
    def get_full_trajectory(self):
        """
        Get the full trajectory, including takeoff and any subsequent velocity profile.
        """
        return {
            "Vx": self.Vx_val,
            "Vy": self.Vy_val,
            "Vz": self.Vz_val,
            "Wx": self.Wx_val,
            "Wy": self.Wy_val,
            "Wz": self.Wz_val,
            "Z": self.Z_val,
            "T": self.T_val,
            "PSI": self.PSI_val
        }
