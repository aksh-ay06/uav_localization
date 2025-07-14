"""
Visualization Utilities for UAV Localization System

This module provides visualization tools for plotting and analyzing
the UAV localization system results including trajectories, sensor data,
and estimation accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple
import matplotlib.patches as patches

class UAVVisualization:
    """
    Visualization tools for UAV localization system.
    
    This class provides methods for plotting trajectories, sensor data,
    and analyzing the performance of the sensor fusion system.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization system.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
    def plot_3d_trajectory(self, positions: List[np.ndarray], labels: List[str] = None, 
                          colors: List[str] = None, title: str = "3D Trajectory") -> None:
        """
        Plot 3D trajectory of UAV flight path.
        
        Args:
            positions: List of position arrays (Nx3) for each trajectory
            labels: Labels for each trajectory
            colors: Colors for each trajectory
            title: Plot title
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Default colors and labels
        if colors is None:
            colors = ['red', 'blue', 'green', 'orange', 'purple']
        if labels is None:
            labels = [f'Trajectory {i+1}' for i in range(len(positions))]
        
        # Plot each trajectory
        for i, pos in enumerate(positions):
            if len(pos) > 0:
                color = colors[i % len(colors)]
                label = labels[i] if i < len(labels) else f'Trajectory {i+1}'
                
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                       color=color, label=label, linewidth=2, alpha=0.8)
                
                # Mark start and end points
                ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], 
                          color=color, s=100, marker='o', alpha=1.0)
                ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], 
                          color=color, s=100, marker='s', alpha=1.0)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sensor_comparison(self, timestamps: np.ndarray, gps_data: np.ndarray, 
                              imu_data: np.ndarray, vo_data: np.ndarray, 
                              fused_data: np.ndarray) -> None:
        """
        Plot comparison of different sensor measurements and fused estimate.
        
        Args:
            timestamps: Time vector
            gps_data: GPS position data (Nx3)
            imu_data: IMU integrated position data (Nx3)
            vo_data: Visual odometry position data (Nx3)
            fused_data: EKF fused position data (Nx3)
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axis_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        
        for i in range(3):
            ax = axes[i]
            
            # Plot each sensor
            if len(gps_data) > 0:
                ax.plot(timestamps, gps_data[:, i], 'r--', label='GPS', alpha=0.7, linewidth=2)
            if len(imu_data) > 0:
                ax.plot(timestamps, imu_data[:, i], 'g:', label='IMU (integrated)', alpha=0.7, linewidth=2)
            if len(vo_data) > 0:
                ax.plot(timestamps, vo_data[:, i], 'b-.', label='Visual Odometry', alpha=0.7, linewidth=2)
            if len(fused_data) > 0:
                ax.plot(timestamps, fused_data[:, i], 'k-', label='EKF Fused', linewidth=3)
            
            ax.set_ylabel(axis_labels[i])
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if i == 2:  # Last subplot
                ax.set_xlabel('Time (s)')
        
        plt.suptitle('Sensor Fusion Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_estimation_uncertainty(self, timestamps: np.ndarray, positions: np.ndarray, 
                                   covariances: np.ndarray) -> None:
        """
        Plot position estimates with uncertainty bounds.
        
        Args:
            timestamps: Time vector
            positions: Position estimates (Nx3)
            covariances: Position covariance matrices (Nx3x3)
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axis_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        colors = ['red', 'green', 'blue']
        
        for i in range(3):
            ax = axes[i]
            
            # Extract standard deviation for this axis
            std_dev = np.sqrt(covariances[:, i, i])
            
            # Plot position estimate
            ax.plot(timestamps, positions[:, i], color=colors[i], 
                   linewidth=3, label=f'{axis_labels[i]} Estimate')
            
            # Plot uncertainty bounds (±3σ)
            ax.fill_between(timestamps, 
                           positions[:, i] - 3*std_dev,
                           positions[:, i] + 3*std_dev,
                           color=colors[i], alpha=0.3, 
                           label='99.7% Confidence (±3σ)')
            
            ax.set_ylabel(axis_labels[i])
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if i == 2:  # Last subplot
                ax.set_xlabel('Time (s)')
        
        plt.suptitle('Position Estimates with Uncertainty', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_imu_data(self, timestamps: np.ndarray, accel_data: np.ndarray, 
                     gyro_data: np.ndarray) -> None:
        """
        Plot IMU accelerometer and gyroscope data.
        
        Args:
            timestamps: Time vector
            accel_data: Accelerometer data (Nx3)
            gyro_data: Gyroscope data (Nx3)
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Accelerometer plots
        accel_labels = ['Accel X (m/s²)', 'Accel Y (m/s²)', 'Accel Z (m/s²)']
        for i in range(3):
            axes[0, i].plot(timestamps, accel_data[:, i], 'b-', linewidth=1)
            axes[0, i].set_title(accel_labels[i])
            axes[0, i].grid(True, alpha=0.3)
            if i == 0:
                axes[0, i].set_ylabel('Acceleration')
        
        # Gyroscope plots
        gyro_labels = ['Gyro X (rad/s)', 'Gyro Y (rad/s)', 'Gyro Z (rad/s)']
        for i in range(3):
            axes[1, i].plot(timestamps, gyro_data[:, i], 'r-', linewidth=1)
            axes[1, i].set_title(gyro_labels[i])
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].grid(True, alpha=0.3)
            if i == 0:
                axes[1, i].set_ylabel('Angular Velocity')
        
        plt.suptitle('IMU Data', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_attitude_estimates(self, timestamps: np.ndarray, attitudes: np.ndarray) -> None:
        """
        Plot attitude estimates (roll, pitch, yaw).
        
        Args:
            timestamps: Time vector
            attitudes: Attitude estimates in radians (Nx3) [roll, pitch, yaw]
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        attitude_labels = ['Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
        colors = ['red', 'green', 'blue']
        
        for i in range(3):
            axes[i].plot(timestamps, attitudes[:, i], color=colors[i], linewidth=2)
            axes[i].set_ylabel(attitude_labels[i])
            axes[i].grid(True, alpha=0.3)
            
            # Add horizontal lines at common angles
            if i < 2:  # Roll and pitch
                axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[i].axhline(y=np.pi/4, color='gray', linestyle=':', alpha=0.5)
                axes[i].axhline(y=-np.pi/4, color='gray', linestyle=':', alpha=0.5)
            
            if i == 2:  # Last subplot
                axes[i].set_xlabel('Time (s)')
        
        plt.suptitle('Attitude Estimates', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_error_analysis(self, timestamps: np.ndarray, true_positions: np.ndarray, 
                           estimated_positions: np.ndarray) -> None:
        """
        Plot error analysis comparing true and estimated positions.
        
        Args:
            timestamps: Time vector
            true_positions: Ground truth positions (Nx3)
            estimated_positions: Estimated positions (Nx3)
        """
        # Calculate errors
        position_errors = estimated_positions - true_positions
        error_magnitudes = np.linalg.norm(position_errors, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Individual axis errors
        axis_labels = ['X Error (m)', 'Y Error (m)', 'Z Error (m)']
        for i in range(3):
            row = i // 2
            col = i % 2
            if i < 2:
                axes[row, col].plot(timestamps, position_errors[:, i], linewidth=2)
                axes[row, col].set_title(axis_labels[i])
                axes[row, col].set_ylabel('Error (m)')
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Z error in bottom left
        axes[1, 0].plot(timestamps, position_errors[:, 2], 'g-', linewidth=2)
        axes[1, 0].set_title(axis_labels[2])
        axes[1, 0].set_ylabel('Error (m)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Error magnitude in bottom right
        axes[1, 1].plot(timestamps, error_magnitudes, 'r-', linewidth=2)
        axes[1, 1].set_title('Position Error Magnitude')
        axes[1, 1].set_ylabel('Error (m)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(error_magnitudes)
        rms_error = np.sqrt(np.mean(error_magnitudes**2))
        max_error = np.max(error_magnitudes)
        
        stats_text = f'Mean: {mean_error:.3f}m\nRMS: {rms_error:.3f}m\nMax: {max_error:.3f}m'
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Position Error Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_real_time_plot(self) -> Tuple:
        """
        Create real-time plotting setup for live visualization.
        
        Returns:
            Tuple of (figure, axes) for real-time updating
        """
        plt.ion()  # Turn on interactive mode
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Initialize empty plots
        line1, = axes[0, 0].plot([], [], 'b-', label='EKF Position')
        axes[0, 0].set_title('Position Estimate')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        line2, = axes[0, 1].plot([], [], 'r-')
        axes[0, 1].set_title('Altitude')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Z (m)')
        axes[0, 1].grid(True)
        
        line3, = axes[1, 0].plot([], [], 'g-')
        axes[1, 0].set_title('Velocity Magnitude')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Speed (m/s)')
        axes[1, 0].grid(True)
        
        line4, = axes[1, 1].plot([], [], 'purple')
        axes[1, 1].set_title('Yaw Angle')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Yaw (rad)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        return fig, axes, [line1, line2, line3, line4]
