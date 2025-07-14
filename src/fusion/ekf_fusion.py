"""
Extended Kalman Filter (EKF) Sensor Fusion Module for UAV Localization

This module implements an Extended Kalman Filter for fusing GPS, IMU, and
Stereo Visual Odometry measurements for robust UAV pose estimation.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.linalg import inv

class EKFSensorFusion:
    """
    Extended Kalman Filter for UAV sensor fusion.
    
    State vector definition:
    [x, y, z, vx, vy, vz, roll, pitch, yaw, accel_bias_x, accel_bias_y, accel_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
    
    This EKF fuses GPS position measurements, IMU inertial measurements,
    and stereo visual odometry pose estimates.
    """
    
    def __init__(self, initial_state: np.ndarray = None, initial_covariance: np.ndarray = None):
        """
        Initialize Extended Kalman Filter for sensor fusion.
        
        Args:
            initial_state: Initial state vector (15x1) - if None, uses zeros
            initial_covariance: Initial covariance matrix (15x15) - if None, uses default
        """
        # State vector dimension: [pos(3), vel(3), att(3), accel_bias(3), gyro_bias(3)]
        self.state_dim = 15
        
        # Initialize state vector
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            self.state = np.zeros(self.state_dim)
        
        # Initialize covariance matrix
        if initial_covariance is not None:
            self.P = initial_covariance.copy()
        else:
            self.P = np.eye(self.state_dim) * 0.1
            # Higher uncertainty for biases
            self.P[9:, 9:] = np.eye(6) * 1.0
        
        # Process noise covariance
        self.Q = self._initialize_process_noise()
        
        # Gravity vector
        self.gravity = np.array([0, 0, 9.81])
        
        # Last update timestamp
        self.last_time = None
    
    def _initialize_process_noise(self) -> np.ndarray:
        """
        Initialize process noise covariance matrix Q.
        
        Returns:
            15x15 process noise covariance matrix
        """
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Position noise (integrated from velocity)
        Q[0:3, 0:3] = np.eye(3) * 0.01
        
        # Velocity noise (integrated from acceleration)
        Q[3:6, 3:6] = np.eye(3) * 0.1
        
        # Attitude noise
        Q[6:9, 6:9] = np.eye(3) * 0.01
        
        # Accelerometer bias noise
        Q[9:12, 9:12] = np.eye(3) * 0.001
        
        # Gyroscope bias noise
        Q[12:15, 12:15] = np.eye(3) * 0.0001
        
        return Q
    
    def ekf_predict(self, imu_data: Dict, dt: float) -> None:
        """
        EKF Prediction step
        Predict the next state based on previous state and IMU measurements
        
        Args:
            imu_data: Processed IMU data dictionary from IMU processor
            dt: Time step in seconds
        """
        # Extract current state components
        pos = self.state[0:3]
        vel = self.state[3:6]
        att = self.state[6:9]  # [roll, pitch, yaw]
        accel_bias = self.state[9:12]
        gyro_bias = self.state[12:15]
        
        # Get IMU measurements
        accel_meas = imu_data['accel_filtered']
        gyro_meas = imu_data['gyro_filtered']
        
        # Bias-corrected measurements
        accel_corrected = accel_meas - accel_bias
        gyro_corrected = gyro_meas - gyro_bias
        
        # Predict attitude using gyroscope
        att_new = att + gyro_corrected * dt
        
        # Normalize angles
        att_new = self._normalize_angles(att_new)
        
        # Get rotation matrix for current attitude
        R_body_to_world = self._euler_to_rotation_matrix(att_new)
        
        # Transform acceleration to world frame
        accel_world = R_body_to_world @ accel_corrected
        
        # Remove gravity
        accel_world -= self.gravity
        
        # Predict velocity and position
        vel_new = vel + accel_world * dt
        pos_new = pos + vel * dt + 0.5 * accel_world * dt**2
        
        # Update state vector
        self.state[0:3] = pos_new
        self.state[3:6] = vel_new
        self.state[6:9] = att_new
        # Biases remain constant (random walk model)
        
        # Compute Jacobian of the state transition function
        F = self._compute_state_jacobian(accel_corrected, gyro_corrected, att, dt)
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q * dt**2
    
    def ekf_update_gps(self, gps_measurement: np.ndarray, gps_covariance: np.ndarray) -> None:
        """
        EKF Update step for GPS measurements
        Update state estimates based on GPS position measurements
        
        Args:
            gps_measurement: GPS position measurement [x, y, z]
            gps_covariance: GPS measurement covariance (3x3)
        """
        # Measurement function: h(x) = [x, y, z] (position only)
        h_x = self.state[0:3]
        
        # Measurement Jacobian (linear for GPS)
        H = np.zeros((3, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # GPS measures position directly
        
        # Innovation
        y = gps_measurement - h_x
        
        # Innovation covariance
        S = H @ self.P @ H.T + gps_covariance
        
        # Kalman gain
        K = self.P @ H.T @ inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
        # Normalize angles after update
        self.state[6:9] = self._normalize_angles(self.state[6:9])
    
    def ekf_update_visual_odometry(self, vo_measurement: np.ndarray, vo_covariance: np.ndarray) -> None:
        """
        EKF Update step for Visual Odometry measurements
        Update state estimates based on stereo visual odometry pose measurements
        
        Args:
            vo_measurement: VO pose measurement [x, y, z, roll, pitch, yaw]
            vo_covariance: VO measurement covariance (6x6)
        """
        # Measurement function: h(x) = [x, y, z, roll, pitch, yaw]
        h_x = self.state[0:3].tolist() + self.state[6:9].tolist()
        h_x = np.array(h_x)
        
        # Measurement Jacobian (linear for VO)
        H = np.zeros((6, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:6, 6:9] = np.eye(3)  # Attitude
        
        # Innovation
        y = vo_measurement - h_x
        
        # Handle angle wrapping in innovation
        y[3:6] = self._normalize_angles(y[3:6])
        
        # Innovation covariance
        S = H @ self.P @ H.T + vo_covariance
        
        # Kalman gain
        K = self.P @ H.T @ inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
        # Normalize angles after update
        self.state[6:9] = self._normalize_angles(self.state[6:9])
    
    def _compute_state_jacobian(self, accel: np.ndarray, gyro: np.ndarray, 
                               att: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute Jacobian of state transition function.
        
        Args:
            accel: Accelerometer measurement
            gyro: Gyroscope measurement
            att: Current attitude [roll, pitch, yaw]
            dt: Time step
            
        Returns:
            15x15 Jacobian matrix
        """
        F = np.eye(self.state_dim)
        
        # Position derivatives
        F[0:3, 3:6] = np.eye(3) * dt  # d_pos/d_vel
        
        # Velocity derivatives w.r.t. attitude
        roll, pitch, yaw = att
        
        # Simplified Jacobian for attitude effect on acceleration transformation
        # This is a linearization of the rotation matrix
        F[3, 6] = dt * (accel[1] * np.cos(roll) * np.sin(pitch) + 
                       accel[2] * np.sin(roll) * np.sin(pitch))
        F[4, 7] = dt * (accel[0] * np.cos(pitch) - 
                       accel[2] * np.sin(pitch))
        F[5, 8] = dt * (-accel[0] * np.sin(yaw) + accel[1] * np.cos(yaw))
        
        # Velocity derivatives w.r.t. accelerometer bias
        R = self._euler_to_rotation_matrix(att)
        F[3:6, 9:12] = -R * dt
        
        # Attitude derivatives
        F[6:9, 6:9] = np.eye(3)  # Attitude evolves as att_new = att + gyro * dt
        F[6:9, 12:15] = -np.eye(3) * dt  # d_att/d_gyro_bias
        
        return F
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.
        
        Args:
            euler: Euler angles [roll, pitch, yaw]
            
        Returns:
            3x3 rotation matrix
        """
        roll, pitch, yaw = euler
        
        # Rotation matrices for each axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (Z-Y-X order)
        R = R_z @ R_y @ R_x
        
        return R
    
    def _normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Normalize angles to [-pi, pi] range.
        
        Args:
            angles: Array of angles in radians
            
        Returns:
            Normalized angles
        """
        return np.mod(angles + np.pi, 2 * np.pi) - np.pi
    
    def get_state_estimate(self) -> Dict:
        """
        Get current state estimate with uncertainty.
        
        Returns:
            Dictionary containing state estimates and covariances
        """
        return {
            'position': self.state[0:3].copy(),
            'velocity': self.state[3:6].copy(),
            'attitude': self.state[6:9].copy(),
            'accel_bias': self.state[9:12].copy(),
            'gyro_bias': self.state[12:15].copy(),
            'position_cov': self.P[0:3, 0:3].copy(),
            'velocity_cov': self.P[3:6, 3:6].copy(),
            'attitude_cov': self.P[6:9, 6:9].copy(),
            'full_state': self.state.copy(),
            'full_covariance': self.P.copy()
        }
    
    def set_initial_state(self, position: np.ndarray, velocity: np.ndarray = None, 
                         attitude: np.ndarray = None) -> None:
        """
        Set initial state for the EKF.
        
        Args:
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz] (default: zeros)
            attitude: Initial attitude [roll, pitch, yaw] (default: zeros)
        """
        self.state[0:3] = position
        
        if velocity is not None:
            self.state[3:6] = velocity
        
        if attitude is not None:
            self.state[6:9] = attitude
