"""
IMU Processor Module for UAV Localization System

This module handles IMU data processing including bias correction,
noise filtering, and integration for pose estimation.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R

class IMUProcessor:
    """
    IMU data processor for handling accelerometer and gyroscope data.
    
    This class processes raw IMU data by applying bias correction,
    noise filtering, and integration to estimate pose increments.
    """
    
    def __init__(self, accel_bias: np.ndarray = None, gyro_bias: np.ndarray = None):
        """
        Initialize IMU processor with bias corrections.
        
        Args:
            accel_bias: Accelerometer bias as 3D numpy array (default: zeros)
            gyro_bias: Gyroscope bias as 3D numpy array (default: zeros)
        """
        self.accel_bias = accel_bias if accel_bias is not None else np.zeros(3)
        self.gyro_bias = gyro_bias if gyro_bias is not None else np.zeros(3)
        
        # Initialize state variables
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.orientation = np.eye(3)  # Rotation matrix
        
        # Noise parameters
        self.accel_noise_std = 0.1  # m/s^2
        self.gyro_noise_std = 0.01  # rad/s
        
        # Gravity vector (assuming local NED frame)
        self.gravity = np.array([0, 0, 9.81])
    
    def preprocess_imu_data(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> dict:
        """
        Function: preprocess_imu_data
        Input: raw accelerometer and gyroscope data, sampling interval dt
        Output: Bias-corrected and integrated IMU pose increments
        
        Preprocess raw IMU data by applying bias correction and noise filtering.
        
        Args:
            accel: Raw accelerometer data as 3D numpy array (m/s^2)
            gyro: Raw gyroscope data as 3D numpy array (rad/s)
            dt: Time step in seconds
            
        Returns:
            Dictionary containing processed IMU data and pose increments
        """
        # Apply bias correction
        accel_corrected = accel - self.accel_bias
        gyro_corrected = gyro - self.gyro_bias
        
        # Apply low-pass filter to reduce noise
        accel_filtered = self._apply_low_pass_filter(accel_corrected)
        gyro_filtered = self._apply_low_pass_filter(gyro_corrected)
        
        # Integrate angular velocity to get orientation increment
        delta_orientation = self._integrate_angular_velocity(gyro_filtered, dt)
        
        # Transform acceleration to world frame and integrate
        accel_world = self.orientation @ accel_filtered
        accel_world -= self.gravity  # Remove gravity component
        
        # Integrate acceleration to get velocity and position increments
        delta_velocity = accel_world * dt
        delta_position = self.velocity * dt + 0.5 * accel_world * dt**2
        
        # Update internal state
        self.velocity += delta_velocity
        self.position += delta_position
        self.orientation = self.orientation @ delta_orientation
        
        return {
            'accel_filtered': accel_filtered,
            'gyro_filtered': gyro_filtered,
            'delta_position': delta_position,
            'delta_velocity': delta_velocity,
            'delta_orientation': delta_orientation,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy()
        }
    
    def _apply_low_pass_filter(self, data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """
        Apply simple low-pass filter to reduce noise in IMU data.
        
        Args:
            data: Input data array
            alpha: Filter parameter (0 < alpha < 1)
            
        Returns:
            Filtered data array
        """
        # Simple exponential moving average filter
        if not hasattr(self, '_prev_data'):
            self._prev_data = data.copy()
        
        filtered_data = alpha * data + (1 - alpha) * self._prev_data
        self._prev_data = filtered_data.copy()
        
        return filtered_data
    
    def _integrate_angular_velocity(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate angular velocity to get orientation increment as rotation matrix.
        
        Args:
            gyro: Angular velocity in rad/s
            dt: Time step in seconds
            
        Returns:
            Rotation matrix representing orientation increment
        """
        # Calculate rotation angle
        angle = np.linalg.norm(gyro) * dt
        
        if angle < 1e-8:
            # Small angle approximation
            return np.eye(3)
        
        # Rotation axis
        axis = gyro / np.linalg.norm(gyro)
        
        # Create rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R_increment = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return R_increment
    
    def euler_from_rotation_matrix(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract Euler angles (roll, pitch, yaw) from rotation matrix.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (roll, pitch, yaw) angles in radians
        """
        # Extract roll, pitch, yaw from rotation matrix
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return roll, pitch, yaw
    
    def calibrate_bias(self, accel_samples: np.ndarray, gyro_samples: np.ndarray) -> None:
        """
        Calibrate IMU bias using stationary measurements.
        
        Args:
            accel_samples: Array of accelerometer samples during stationary period
            gyro_samples: Array of gyroscope samples during stationary period
        """
        # Calculate mean bias
        self.accel_bias = np.mean(accel_samples, axis=0)
        self.gyro_bias = np.mean(gyro_samples, axis=0)
        
        # Remove gravity from accelerometer bias (assuming z-axis is up)
        self.accel_bias[2] -= 9.81
    
    def get_imu_covariance(self) -> np.ndarray:
        """
        Get IMU measurement covariance matrix.
        
        Returns:
            6x6 covariance matrix for [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        """
        covariance = np.zeros((6, 6))
        
        # Accelerometer covariance (first 3 elements)
        covariance[0:3, 0:3] = np.eye(3) * self.accel_noise_std**2
        
        # Gyroscope covariance (last 3 elements)
        covariance[3:6, 3:6] = np.eye(3) * self.gyro_noise_std**2
        
        return covariance
    
    def reset_integration(self) -> None:
        """
        Reset integration state (velocity, position, orientation).
        """
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.orientation = np.eye(3)
