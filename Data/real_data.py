import sympy as sp
import numpy as np
import pandas as pd
import math

class RealData:
    def __init__(self, data_source):
        """
        Initialize the RealData class with a CSV file path or a pandas DataFrame.

        :param data_source: str (path to CSV) or pd.DataFrame
        """
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.data = data_source
        else:
            raise ValueError("data_source must be a file path (str) or a pandas DataFrame.")

    def get_mocap_data(self):
        """
        Placeholder method to extract motion capture (mocap) data.
        """
        pass

    def get_imu_data(self):
        """
        Placeholder method to extract IMU data.
        """
        pass

    def get_optic_flow_data(self):
        """
        Placeholder method to extract optic flow data.
        """
        pass

    def get_wind_sensor_data(self):
        """
        Placeholder method to extract wind sensor data.
        """
        pass

    def rotate_data(self):
        """
        Placeholder method to rotate data into the correct coordinate frame.
        """
        pass

    def interpolate_data(self, time1, data1, time2):
        """
        input: time1 (interp from), data1 (interp from), time2 (interp to)
        output: interpolated data2
        """
        return np.interp(time2, time1, data1)
    
    def time_array(self, data):
        "input: data frame with columns secs and nsecs"
        "output: array of time in seconds"
        time = []
        for i in range(len(data)):
            time.append(data['secs'][i] + data['nsecs'][i] * 1e-9)

        return time
    
    def quaternion_to_euler(self, x, y, z, w):
        """
        input: x, y, z, w
        output: R, P, Y
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        R = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        P = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Y = math.degrees(math.atan2(t3, t4))

        return R, P, Y
    
    def high_pass_filter(self, data, dt, cut_off_freq):
        """
        input: data, dt, cut_off_freq
        output: data_filtered
        """
        # Create the high pass filter
        b, a = signal.butter(1, cut_off_freq, 'high', fs=1/dt)
        
        # Apply the high pass filter
        data_filtered = signal.filtfilt(b, a, data)
        
        return data_filtered

    def low_pass_filter(self, data, dt, cut_off_freq):
        """
        input: data, dt, cut_off_freq
        output: data_filtered
        """
        # Create the high pass filter
        b, a = signal.butter(1, cut_off_freq, 'low', fs=1/dt)
        
        # Apply the high pass filter
        data_filtered = signal.filtfilt(b, a, data)
        
        return data_filtered
    
    def rotation_matrix_from_euler(self,roll, pitch, yaw):
        """
        Creates a rotation matrix from roll, pitch, yaw angles (in radians).
        """
        # Compute rotation matrices for roll, pitch, and yaw
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combine the rotation matrices (Z * Y * X)
        R = R_yaw @ R_pitch @ R_roll
        return R
    
    def omega_dot_to_euler_dot(omega_x, omega_y, omega_z, phi, theta):
        """
        input: omega_x, omega_y, omega_z, phi, theta
        output: phi_dot, theta_dot, psi_dot
        """
        phi_dot = omega_x + omega_z * np.tan(theta) * np.cos(phi) + omega_y * np.tan(theta) * np.sin(phi)
        theta_dot = omega_y * np.cos(phi) - omega_z * np.sin(phi)
        psi_dot = omega_z * np.cos(phi) * (1 / np.cos(theta)) + omega_z * np.sin(phi) * (1 / np.cos(theta))
        return phi_dot, theta_dot, psi_dot