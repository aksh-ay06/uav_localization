"""
Improved Extended Kalman Filter (EKF) with Better Tuning for UAV Localization

This version addresses the high positioning errors by:
1. Better process noise tuning
2. Adaptive measurement noise
3. Improved state initialization
4. Enhanced numerical stability
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.linalg import inv, cholesky, LinAlgError
import warnings

class ImprovedEKFSensorFusion:
    """
    Improved Extended Kalman Filter for UAV sensor fusion with better error performance.
    
    State vector definition:
    [x, y, z, vx, vy, vz, roll, pitch, yaw, accel_bias_x, accel_bias_y, accel_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
    
    Key improvements:
    - Reduced process noise for better stability
    - Adaptive measurement noise based on sensor quality
    - Better initial state estimation
    - Numerical stability improvements
    """
    
    def __init__(self, initial_state: np.ndarray = None, initial_covariance: np.ndarray = None):
        """
        Initialize Improved Extended Kalman Filter for sensor fusion.
        
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
        
        # Initialize covariance matrix with much lower uncertainty
        if initial_covariance is not None:
            self.P = initial_covariance.copy()
        else:
            self.P = self._initialize_covariance()
        
        # Much smaller process noise for better stability
        self.Q = self._initialize_process_noise()
        
        # Gravity vector
        self.gravity = np.array([0, 0, 9.81])
        
        # Last update timestamp
        self.last_time = None
        
        # Innovation monitoring for adaptive filtering
        self.innovation_history = []
        self.max_innovation_history = 50
        
        # Numerical stability parameters
        self.min_eigenvalue = 1e-8
        self.max_eigenvalue = 1e8
    
    def _initialize_covariance(self) -> np.ndarray:
        """
        Initialize covariance matrix with realistic uncertainty values.
        
        Returns:
            15x15 initial covariance matrix
        """
        P = np.zeros((self.state_dim, self.state_dim))
        
        # Position uncertainty: ±10m initially
        P[0:3, 0:3] = np.eye(3) * 100.0  # 10m std
        
        # Velocity uncertainty: ±5m/s initially
        P[3:6, 3:6] = np.eye(3) * 25.0   # 5m/s std
        
        # Attitude uncertainty: ±30 degrees initially
        P[6:9, 6:9] = np.eye(3) * (np.pi/6)**2  # 30 deg std
        
        # Accelerometer bias uncertainty: ±0.5m/s²
        P[9:12, 9:12] = np.eye(3) * 0.25  # 0.5m/s² std
        
        # Gyroscope bias uncertainty: ±0.1rad/s
        P[12:15, 12:15] = np.eye(3) * 0.01  # 0.1rad/s std
        
        return P
    
    def _initialize_process_noise(self) -> np.ndarray:
        """
        Initialize process noise covariance matrix Q with much smaller values.
        
        Returns:
            15x15 process noise covariance matrix
        """
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Position noise (very small - integrated from velocity)
        Q[0:3, 0:3] = np.eye(3) * 0.001  # Reduced from 0.01
        
        # Velocity noise (integrated from acceleration)
        Q[3:6, 3:6] = np.eye(3) * 0.01   # Reduced from 0.1
        
        # Attitude noise (very small for stable attitude)
        Q[6:9, 6:9] = np.eye(3) * 0.001  # Reduced from 0.01
        
        # Accelerometer bias noise (very slow drift)
        Q[9:12, 9:12] = np.eye(3) * 0.0001  # Reduced from 0.001
        
        # Gyroscope bias noise (very slow drift)
        Q[12:15, 12:15] = np.eye(3) * 0.00001  # Reduced from 0.0001
        
        return Q
    
    def _ensure_numerical_stability(self) -> None:
        """
        Ensure numerical stability of the covariance matrix.
        """
        # Check for NaN or infinite values
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            warnings.warn("NaN or infinite values in covariance matrix. Reinitializing.")
            self.P = self._initialize_covariance()
            return
        
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
        
        # Eigenvalue decomposition for positive definiteness
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self.P)
            
            # Clip eigenvalues to ensure positive definiteness
            eigenvals = np.clip(eigenvals, self.min_eigenvalue, self.max_eigenvalue)
            
            # Reconstruct matrix
            self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
        except LinAlgError:
            warnings.warn("Eigenvalue decomposition failed. Reinitializing covariance.")
            self.P = self._initialize_covariance()
    
    def ekf_predict(self, imu_data: Dict, dt: float) -> None:
        """
        Improved EKF Prediction step with better numerical stability.
        
        Args:
            imu_data: Processed IMU data dictionary from IMU processor
            dt: Time step in seconds
        """
        # Clamp time step to reasonable bounds
        dt = np.clip(dt, 0.001, 0.1)  # Between 1ms and 100ms
        
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
        
        # Predict attitude using gyroscope with better integration
        att_new = att + gyro_corrected * dt
        
        # Normalize angles to [-pi, pi]
        att_new = self._normalize_angles(att_new)
        
        # Get rotation matrix for current attitude
        R_body_to_world = self._euler_to_rotation_matrix(att_new)
        
        # Transform acceleration to world frame
        accel_world = R_body_to_world @ accel_corrected
        
        # Remove gravity (more robust gravity handling)
        accel_world -= self.gravity
        
        # Apply velocity damping to prevent drift
        velocity_damping = 0.999  # Small damping factor
        vel_new = vel * velocity_damping + accel_world * dt
        
        # Predict position with improved integration
        pos_new = pos + vel * dt + 0.5 * accel_world * dt**2
        
        # Update state vector
        self.state[0:3] = pos_new
        self.state[3:6] = vel_new
        self.state[6:9] = att_new
        # Biases remain constant (random walk model)
        
        # Compute Jacobian of the state transition function
        F = self._compute_state_jacobian(accel_corrected, gyro_corrected, att, dt)
        
        # Predict covariance with adaptive process noise
        Q_adaptive = self.Q * dt  # Scale with time step
        
        # Apply process noise based on acceleration magnitude (adaptive)
        accel_magnitude = np.linalg.norm(accel_world)
        if accel_magnitude > 2.0:  # High acceleration
            Q_adaptive *= 2.0  # Increase process noise
        
        self.P = F @ self.P @ F.T + Q_adaptive
        
        # Ensure numerical stability
        self._ensure_numerical_stability()
    
    def ekf_update_gps(self, gps_measurement: np.ndarray, gps_covariance: np.ndarray) -> None:
        """
        Improved EKF Update step for GPS measurements with outlier detection.
        
        Args:
            gps_measurement: GPS position measurement [x, y, z]
            gps_covariance: GPS measurement covariance (3x3)
        """
        # Measurement function: h(x) = [x, y, z] (position only)
        h_x = self.state[0:3]
        
        # Innovation (residual)
        y = gps_measurement - h_x
        
        # Outlier detection based on Mahalanobis distance
        innovation_magnitude = np.linalg.norm(y)
        if innovation_magnitude > 50.0:  # 50m threshold
            warnings.warn(f"GPS outlier detected: {innovation_magnitude:.2f}m. Skipping update.")
            return
        
        # Measurement Jacobian (linear for GPS)
        H = np.zeros((3, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # GPS measures position directly
        
        # Adaptive GPS covariance based on innovation
        gps_covariance_adaptive = gps_covariance.copy()
        
        # Store innovation for adaptive filtering
        self.innovation_history.append(innovation_magnitude)
        if len(self.innovation_history) > self.max_innovation_history:
            self.innovation_history.pop(0)
        
        # Increase GPS noise if consistently high innovations
        if len(self.innovation_history) >= 10:
            avg_innovation = np.mean(self.innovation_history[-10:])
            if avg_innovation > 10.0:  # High innovation
                gps_covariance_adaptive *= 2.0
        
        # Innovation covariance
        S = H @ self.P @ H.T + gps_covariance_adaptive
        
        # Ensure S is positive definite
        try:
            S_inv = inv(S)
        except LinAlgError:
            warnings.warn("Singular innovation covariance matrix. Skipping GPS update.")
            return
        
        # Kalman gain
        K = self.P @ H.T @ S_inv
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I = np.eye(self.state_dim)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ gps_covariance_adaptive @ K.T
        
        # Normalize angles after update
        self.state[6:9] = self._normalize_angles(self.state[6:9])
        
        # Ensure numerical stability
        self._ensure_numerical_stability()
    
    def ekf_update_visual_odometry(self, vo_measurement: np.ndarray, vo_covariance: np.ndarray) -> None:
        """
        Improved EKF Update step for Visual Odometry measurements.
        
        Args:
            vo_measurement: VO pose measurement [x, y, z, roll, pitch, yaw]
            vo_covariance: VO measurement covariance (6x6)
        """
        # Measurement function: h(x) = [x, y, z, roll, pitch, yaw]
        h_x = np.concatenate([self.state[0:3], self.state[6:9]])
        
        # Innovation
        y = vo_measurement - h_x
        
        # Handle angle wrapping in innovation
        y[3:6] = self._normalize_angles(y[3:6])
        
        # Outlier detection
        innovation_magnitude = np.linalg.norm(y[:3])  # Check position component
        if innovation_magnitude > 30.0:  # 30m threshold for VO
            warnings.warn(f"VO outlier detected: {innovation_magnitude:.2f}m. Skipping update.")
            return
        
        # Measurement Jacobian (linear for VO)
        H = np.zeros((6, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:6, 6:9] = np.eye(3)  # Attitude
        
        # Innovation covariance
        S = H @ self.P @ H.T + vo_covariance
        
        # Ensure S is positive definite
        try:
            S_inv = inv(S)
        except LinAlgError:
            warnings.warn("Singular innovation covariance matrix. Skipping VO update.")
            return
        
        # Kalman gain
        K = self.P @ H.T @ S_inv
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance (Joseph form)
        I = np.eye(self.state_dim)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ vo_covariance @ K.T
        
        # Normalize angles after update
        self.state[6:9] = self._normalize_angles(self.state[6:9])
        
        # Ensure numerical stability
        self._ensure_numerical_stability()
    
    def _compute_state_jacobian(self, accel: np.ndarray, gyro: np.ndarray, 
                               att: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute improved Jacobian of state transition function.
        
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
        
        # Velocity derivatives w.r.t. attitude (simplified linearization)
        roll, pitch, yaw = att
        
        # More accurate Jacobian for attitude effect on acceleration transformation
        R = self._euler_to_rotation_matrix(att)
        
        # Compute derivatives of rotation matrix w.r.t. Euler angles
        dR_droll = self._compute_rotation_derivative(att, 0)
        dR_dpitch = self._compute_rotation_derivative(att, 1)
        dR_dyaw = self._compute_rotation_derivative(att, 2)
        
        # Velocity derivatives w.r.t. attitude
        F[3:6, 6] = (dR_droll @ accel) * dt  # d_vel/d_roll
        F[3:6, 7] = (dR_dpitch @ accel) * dt  # d_vel/d_pitch
        F[3:6, 8] = (dR_dyaw @ accel) * dt   # d_vel/d_yaw
        
        # Velocity derivatives w.r.t. accelerometer bias
        F[3:6, 9:12] = -R * dt
        
        # Attitude derivatives
        F[6:9, 6:9] = np.eye(3)  # Attitude evolves as att_new = att + gyro * dt
        F[6:9, 12:15] = -np.eye(3) * dt  # d_att/d_gyro_bias
        
        return F
    
    def _compute_rotation_derivative(self, euler: np.ndarray, axis: int) -> np.ndarray:
        """
        Compute derivative of rotation matrix w.r.t. specific Euler angle.
        
        Args:
            euler: Euler angles [roll, pitch, yaw]
            axis: Axis index (0=roll, 1=pitch, 2=yaw)
            
        Returns:
            3x3 derivative matrix
        """
        roll, pitch, yaw = euler
        
        if axis == 0:  # Roll derivative
            return np.array([
                [0, 0, 0],
                [0, -np.sin(roll), -np.cos(roll)],
                [0, np.cos(roll), -np.sin(roll)]
            ])
        elif axis == 1:  # Pitch derivative
            return np.array([
                [-np.sin(pitch), 0, np.cos(pitch)],
                [0, 0, 0],
                [-np.cos(pitch), 0, -np.sin(pitch)]
            ])
        else:  # Yaw derivative
            return np.array([
                [-np.sin(yaw), -np.cos(yaw), 0],
                [np.cos(yaw), -np.sin(yaw), 0],
                [0, 0, 0]
            ])
    
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
    
    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the filter performance.
        
        Returns:
            Dictionary containing diagnostic information
        """
        # Compute condition number of covariance matrix
        try:
            cond_num = np.linalg.cond(self.P)
        except:
            cond_num = float('inf')
        
        return {
            'condition_number': cond_num,
            'avg_innovation': np.mean(self.innovation_history) if self.innovation_history else 0.0,
            'max_innovation': np.max(self.innovation_history) if self.innovation_history else 0.0,
            'position_uncertainty': np.sqrt(np.trace(self.P[0:3, 0:3])),
            'velocity_uncertainty': np.sqrt(np.trace(self.P[3:6, 3:6])),
            'attitude_uncertainty': np.sqrt(np.trace(self.P[6:9, 6:9]))
        }
