"""
Sample Data Generator for UAV Localization System Testing

This module generates synthetic sensor data for testing and demonstrating
the UAV localization system capabilities.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
import os

class SyntheticDataGenerator:
    """
    Generate synthetic sensor data for UAV localization testing.
    
    This class creates realistic synthetic GPS, IMU, and stereo camera data
    for testing the localization system without requiring actual hardware.
    """
    
    def __init__(self, trajectory_type: str = "circular", noise_level: float = 1.0):
        """
        Initialize synthetic data generator.
        
        Args:
            trajectory_type: Type of trajectory ("circular", "linear", "figure8")
            noise_level: Noise scaling factor (1.0 = normal noise)
        """
        self.trajectory_type = trajectory_type
        self.noise_level = noise_level
        
        # Simulation parameters
        self.gravity = 9.81
        self.reference_lat = 40.7589  # NYC coordinates
        self.reference_lon = -73.9851
        self.reference_alt = 100.0
        
        # Noise parameters
        self.gps_noise_std = np.array([2.0, 2.0, 3.0]) * noise_level  # meters
        self.imu_accel_noise_std = 0.1 * noise_level  # m/s^2
        self.imu_gyro_noise_std = 0.01 * noise_level  # rad/s
        
        # IMU biases
        self.accel_bias = np.array([0.1, 0.05, 0.2]) * noise_level
        self.gyro_bias = np.array([0.01, -0.02, 0.005]) * noise_level
        
    def generate_trajectory(self, duration: float, dt: float) -> Dict:
        """
        Generate ground truth trajectory.
        
        Args:
            duration: Total duration in seconds
            dt: Time step in seconds
            
        Returns:
            Dictionary containing ground truth trajectory data
        """
        t = np.arange(0, duration, dt)
        n_points = len(t)
        
        # Initialize arrays
        positions = np.zeros((n_points, 3))
        velocities = np.zeros((n_points, 3))
        accelerations = np.zeros((n_points, 3))
        attitudes = np.zeros((n_points, 3))  # [roll, pitch, yaw]
        angular_velocities = np.zeros((n_points, 3))
        
        if self.trajectory_type == "circular":
            # Circular trajectory
            radius = 50.0  # meters
            angular_freq = 0.2  # rad/s (slow circle)
            altitude_variation = 10.0  # meters
            
            for i, time in enumerate(t):
                # Position
                positions[i, 0] = radius * np.cos(angular_freq * time)
                positions[i, 1] = radius * np.sin(angular_freq * time)
                positions[i, 2] = altitude_variation * np.sin(0.1 * time)
                
                # Velocity
                velocities[i, 0] = -radius * angular_freq * np.sin(angular_freq * time)
                velocities[i, 1] = radius * angular_freq * np.cos(angular_freq * time)
                velocities[i, 2] = altitude_variation * 0.1 * np.cos(0.1 * time)
                
                # Acceleration
                accelerations[i, 0] = -radius * angular_freq**2 * np.cos(angular_freq * time)
                accelerations[i, 1] = -radius * angular_freq**2 * np.sin(angular_freq * time)
                accelerations[i, 2] = -altitude_variation * 0.01 * np.sin(0.1 * time)
                
                # Attitude (banking in turns)
                attitudes[i, 0] = 0.2 * np.sin(angular_freq * time)  # roll
                attitudes[i, 1] = 0.1 * np.cos(angular_freq * time)  # pitch
                attitudes[i, 2] = angular_freq * time  # yaw
                
                # Angular velocity
                angular_velocities[i, 0] = 0.2 * angular_freq * np.cos(angular_freq * time)
                angular_velocities[i, 1] = -0.1 * angular_freq * np.sin(angular_freq * time)
                angular_velocities[i, 2] = angular_freq
                
        elif self.trajectory_type == "linear":
            # Linear trajectory with some curves
            speed = 10.0  # m/s
            
            for i, time in enumerate(t):
                # Position
                positions[i, 0] = speed * time + 5 * np.sin(0.1 * time)
                positions[i, 1] = 2 * np.sin(0.2 * time)
                positions[i, 2] = 5 * np.sin(0.05 * time)
                
                # Velocity (derivative of position)
                velocities[i, 0] = speed + 0.5 * np.cos(0.1 * time)
                velocities[i, 1] = 0.4 * np.cos(0.2 * time)
                velocities[i, 2] = 0.25 * np.cos(0.05 * time)
                
                # Acceleration (derivative of velocity)
                accelerations[i, 0] = -0.05 * np.sin(0.1 * time)
                accelerations[i, 1] = -0.08 * np.sin(0.2 * time)
                accelerations[i, 2] = -0.0125 * np.sin(0.05 * time)
                
                # Simple attitude
                attitudes[i, 0] = 0.1 * np.sin(0.1 * time)  # roll
                attitudes[i, 1] = 0.05 * np.sin(0.2 * time)  # pitch
                attitudes[i, 2] = 0.1 * time  # yaw
                
                # Angular velocity
                angular_velocities[i, 0] = 0.01 * np.cos(0.1 * time)
                angular_velocities[i, 1] = 0.01 * np.cos(0.2 * time)
                angular_velocities[i, 2] = 0.1
                
        elif self.trajectory_type == "figure8":
            # Figure-8 trajectory
            scale = 30.0  # meters
            angular_freq = 0.1  # rad/s
            
            for i, time in enumerate(t):
                # Position (Lissajous curve for figure-8)
                positions[i, 0] = scale * np.sin(angular_freq * time)
                positions[i, 1] = scale * np.sin(2 * angular_freq * time) / 2
                positions[i, 2] = 10 * np.sin(0.05 * time)
                
                # Velocity
                velocities[i, 0] = scale * angular_freq * np.cos(angular_freq * time)
                velocities[i, 1] = scale * angular_freq * np.cos(2 * angular_freq * time)
                velocities[i, 2] = 0.5 * np.cos(0.05 * time)
                
                # Acceleration
                accelerations[i, 0] = -scale * angular_freq**2 * np.sin(angular_freq * time)
                accelerations[i, 1] = -2 * scale * angular_freq**2 * np.sin(2 * angular_freq * time)
                accelerations[i, 2] = -0.025 * np.sin(0.05 * time)
                
                # Attitude
                attitudes[i, 0] = 0.3 * np.sin(2 * angular_freq * time)  # roll
                attitudes[i, 1] = 0.1 * np.sin(angular_freq * time)  # pitch
                attitudes[i, 2] = np.arctan2(velocities[i, 1], velocities[i, 0])  # yaw towards velocity
                
                # Angular velocity
                angular_velocities[i, 0] = 0.6 * angular_freq * np.cos(2 * angular_freq * time)
                angular_velocities[i, 1] = 0.1 * angular_freq * np.cos(angular_freq * time)
                # Yaw rate calculation would be complex, use simplified version
                angular_velocities[i, 2] = 0.1 * np.sin(angular_freq * time)
        
        return {
            'timestamps': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'attitudes': attitudes,
            'angular_velocities': angular_velocities
        }
    
    def generate_gps_data(self, ground_truth: Dict, gps_rate: float = 2.0) -> Dict:
        """
        Generate synthetic GPS data with noise and dropouts.
        
        Args:
            ground_truth: Ground truth trajectory data
            gps_rate: GPS measurement rate in Hz
            
        Returns:
            Dictionary containing GPS measurements
        """
        timestamps = ground_truth['timestamps']
        positions = ground_truth['positions']
        dt = timestamps[1] - timestamps[0]
        
        # Determine GPS measurement indices
        gps_interval = int(1.0 / (gps_rate * dt))
        gps_indices = np.arange(0, len(timestamps), gps_interval)
        
        gps_timestamps = timestamps[gps_indices]
        gps_positions_local = positions[gps_indices]
        
        # Convert to GPS coordinates (lat, lon, alt)
        gps_coords = []
        for pos in gps_positions_local:
            # Simple conversion (not geodetically accurate, for demo only)
            lat = self.reference_lat + pos[1] / 111320.0  # ~111km per degree lat
            lon = self.reference_lon + pos[0] / (111320.0 * np.cos(np.radians(self.reference_lat)))
            alt = self.reference_alt + pos[2]
            
            # Add noise
            lat += np.random.normal(0, self.gps_noise_std[1] / 111320.0)
            lon += np.random.normal(0, self.gps_noise_std[0] / (111320.0 * np.cos(np.radians(self.reference_lat))))
            alt += np.random.normal(0, self.gps_noise_std[2])
            
            gps_coords.append([lat, lon, alt])
        
        gps_coords = np.array(gps_coords)
        
        # Simulate occasional GPS dropouts (5% chance)
        valid_measurements = np.random.random(len(gps_coords)) > 0.05
        
        return {
            'timestamps': gps_timestamps[valid_measurements],
            'coordinates': gps_coords[valid_measurements],  # [lat, lon, alt]
            'valid_indices': gps_indices[valid_measurements]
        }
    
    def generate_imu_data(self, ground_truth: Dict) -> Dict:
        """
        Generate synthetic IMU data with noise and biases.
        
        Args:
            ground_truth: Ground truth trajectory data
            
        Returns:
            Dictionary containing IMU measurements
        """
        timestamps = ground_truth['timestamps']
        accelerations = ground_truth['accelerations']
        angular_velocities = ground_truth['angular_velocities']
        attitudes = ground_truth['attitudes']
        
        # Add gravity to accelerations (in body frame)
        accel_measurements = []
        gyro_measurements = []
        
        for i, (accel, att) in enumerate(zip(accelerations, attitudes)):
            # Transform gravity to body frame
            roll, pitch, yaw = att
            
            # Rotation matrix from world to body frame
            R_world_to_body = self._euler_to_rotation_matrix(-roll, -pitch, -yaw)
            gravity_body = R_world_to_body @ np.array([0, 0, self.gravity])
            
            # Add gravity to acceleration (IMU measures specific force)
            accel_with_gravity = accel + gravity_body
            
            # Add bias and noise
            accel_noisy = accel_with_gravity + self.accel_bias + \
                         np.random.normal(0, self.imu_accel_noise_std, 3)
            
            gyro_noisy = angular_velocities[i] + self.gyro_bias + \
                        np.random.normal(0, self.imu_gyro_noise_std, 3)
            
            accel_measurements.append(accel_noisy)
            gyro_measurements.append(gyro_noisy)
        
        return {
            'timestamps': timestamps,
            'accelerations': np.array(accel_measurements),
            'angular_velocities': np.array(gyro_measurements)
        }
    
    def generate_stereo_images(self, ground_truth: Dict, image_rate: float = 10.0) -> Dict:
        """
        Generate synthetic stereo camera images with features.
        
        Args:
            ground_truth: Ground truth trajectory data
            image_rate: Camera frame rate in Hz
            
        Returns:
            Dictionary containing stereo image data
        """
        timestamps = ground_truth['timestamps']
        positions = ground_truth['positions']
        attitudes = ground_truth['attitudes']
        dt = timestamps[1] - timestamps[0]
        
        # Determine camera measurement indices
        camera_interval = int(1.0 / (image_rate * dt))
        camera_indices = np.arange(0, len(timestamps), camera_interval)
        
        camera_timestamps = timestamps[camera_indices]
        
        # Generate simple synthetic images with features
        image_width, image_height = 640, 480
        stereo_data = []
        
        for i, idx in enumerate(camera_indices):
            pos = positions[idx]
            att = attitudes[idx]
            
            # Create synthetic images with features
            left_image = self._create_synthetic_image(pos, att, image_width, image_height, is_left=True)
            right_image = self._create_synthetic_image(pos, att, image_width, image_height, is_left=False)
            
            stereo_data.append({
                'left_image': left_image,
                'right_image': right_image,
                'position': pos,
                'attitude': att
            })
        
        return {
            'timestamps': camera_timestamps,
            'stereo_pairs': stereo_data,
            'camera_indices': camera_indices
        }
    
    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.
        
        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            
        Returns:
            3x3 rotation matrix
        """
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
    
    def _create_synthetic_image(self, position: np.ndarray, attitude: np.ndarray, 
                               width: int, height: int, is_left: bool = True) -> np.ndarray:
        """
        Create synthetic image with features for visual odometry.
        
        Args:
            position: Camera position
            attitude: Camera attitude
            width: Image width
            height: Image height
            is_left: Whether this is left camera (for stereo offset)
            
        Returns:
            Synthetic image as numpy array
        """
        # Create blank image
        image = np.zeros((height, width), dtype=np.uint8)
        
        # Add some synthetic features (dots and lines)
        num_features = 50
        
        for _ in range(num_features):
            # Random feature position influenced by camera pose
            x = int(np.random.uniform(50, width - 50))
            y = int(np.random.uniform(50, height - 50))
            
            # Add stereo disparity for left/right cameras
            if not is_left:
                disparity = int(np.random.uniform(5, 20))  # Simple disparity simulation
                x = max(0, min(width - 1, x - disparity))
            
            # Draw feature (circle)
            cv2.circle(image, (x, y), 3, 255, -1)
        
        # Add some noise
        noise = np.random.normal(0, 10, (height, width))
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def save_synthetic_dataset(self, output_dir: str, duration: float = 60.0, dt: float = 0.01) -> None:
        """
        Generate and save a complete synthetic dataset.
        
        Args:
            output_dir: Output directory for saved data
            duration: Duration of simulation in seconds
            dt: Time step in seconds
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating synthetic dataset for {duration}s simulation...")
        
        # Generate ground truth trajectory
        ground_truth = self.generate_trajectory(duration, dt)
        
        # Generate sensor data
        gps_data = self.generate_gps_data(ground_truth, gps_rate=2.0)
        imu_data = self.generate_imu_data(ground_truth)
        stereo_data = self.generate_stereo_images(ground_truth, image_rate=10.0)
        
        # Save data
        np.savez(os.path.join(output_dir, 'ground_truth.npz'), **ground_truth)
        np.savez(os.path.join(output_dir, 'gps_data.npz'), **gps_data)
        np.savez(os.path.join(output_dir, 'imu_data.npz'), **imu_data)
        
        # Save stereo images
        stereo_dir = os.path.join(output_dir, 'stereo_images')
        os.makedirs(stereo_dir, exist_ok=True)
        
        for i, frame_data in enumerate(stereo_data['stereo_pairs']):
            cv2.imwrite(os.path.join(stereo_dir, f'left_{i:06d}.png'), frame_data['left_image'])
            cv2.imwrite(os.path.join(stereo_dir, f'right_{i:06d}.png'), frame_data['right_image'])
        
        # Save stereo timestamps and poses
        np.savez(os.path.join(output_dir, 'stereo_data.npz'), 
                timestamps=stereo_data['timestamps'],
                camera_indices=stereo_data['camera_indices'])
        
        print(f"Dataset saved to {output_dir}")
        print(f"  Ground truth: {len(ground_truth['timestamps'])} points")
        print(f"  GPS measurements: {len(gps_data['timestamps'])} points")
        print(f"  IMU measurements: {len(imu_data['timestamps'])} points")
        print(f"  Stereo images: {len(stereo_data['timestamps'])} pairs")


def main():
    """Demonstrate synthetic data generation."""
    # Create data generator
    generator = SyntheticDataGenerator(trajectory_type="circular", noise_level=1.0)
    
    # Generate and save synthetic dataset
    output_dir = "/home/akshay/uav_localization/data/synthetic"
    generator.save_synthetic_dataset(output_dir, duration=30.0, dt=0.01)
    
    print("Synthetic data generation complete!")

if __name__ == "__main__":
    main()
