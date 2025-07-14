"""
Main UAV Localization System

This is the main application that orchestrates the sensor fusion process
using GPS, IMU, and Stereo Visual Odometry data with an Extended Kalman Filter.
"""

import numpy as np
import time
from typing import Dict, List, Optional
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensors.gps_processor import GPSProcessor
from sensors.imu_processor import IMUProcessor
from sensors.stereo_vo import StereoVisualOdometry, StereoCalibration
from fusion.ekf_fusion import EKFSensorFusion
from utils.visualization import UAVVisualization

class UAVLocalizationSystem:
    """
    Main UAV Localization System class.
    
    This class integrates all sensor processors and the EKF fusion algorithm
    to provide real-time UAV pose estimation.
    """
    
    def __init__(self, reference_gps: tuple, stereo_calibration: StereoCalibration):
        """
        Initialize UAV localization system.
        
        Args:
            reference_gps: Reference GPS coordinates (lat, lon, alt)
            stereo_calibration: Stereo camera calibration parameters
        """
        # Initialize sensor processors
        self.gps_processor = GPSProcessor(*reference_gps)
        self.imu_processor = IMUProcessor()
        self.stereo_vo = StereoVisualOdometry(stereo_calibration)
        
        # Initialize EKF sensor fusion
        self.ekf = EKFSensorFusion()
        
        # Initialize visualization
        self.visualizer = UAVVisualization()
        
        # Data storage for analysis
        self.data_log = {
            'timestamps': [],
            'gps_positions': [],
            'imu_positions': [],
            'vo_positions': [],
            'fused_positions': [],
            'fused_velocities': [],
            'fused_attitudes': [],
            'position_covariances': []
        }
        
        # System state
        self.is_initialized = False
        self.start_time = None
        
    def initialize_system(self, initial_gps: tuple, initial_imu_samples: Dict) -> None:
        """
        Initialize the localization system with initial measurements.
        
        Args:
            initial_gps: Initial GPS measurement (lat, lon, alt)
            initial_imu_samples: Dictionary with 'accel' and 'gyro' sample arrays for bias calibration
        """
        print("Initializing UAV Localization System...")
        
        # Calibrate IMU bias
        print("Calibrating IMU bias...")
        self.imu_processor.calibrate_bias(
            initial_imu_samples['accel'], 
            initial_imu_samples['gyro']
        )
        
        # Set initial position from GPS
        initial_position = np.array(self.gps_processor.convert_gps_to_cartesian(*initial_gps))
        self.ekf.set_initial_state(position=initial_position)
        
        # Initialize timing
        self.start_time = time.time()
        
        self.is_initialized = True
        print("System initialization complete!")
    
    def process_sensor_data(self, sensor_data: Dict) -> Dict:
        """
        Main Sensor Fusion Loop
        Inputs: synchronized IMU, GPS, Stereo data
        Outputs: EKF fused pose estimates for UAV localization
        
        Process a timestep of sensor data and return fused pose estimate.
        
        Args:
            sensor_data: Dictionary containing sensor measurements:
                - 'timestamp': Current timestamp
                - 'imu': {'accel': np.array, 'gyro': np.array}
                - 'gps': {'lat': float, 'lon': float, 'alt': float} (optional)
                - 'stereo': {'left_image': np.array, 'right_image': np.array} (optional)
                - 'dt': Time step since last measurement
        
        Returns:
            Dictionary containing fused pose estimate and sensor data
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        timestamp = sensor_data['timestamp']
        dt = sensor_data['dt']
        
        # Process IMU data
        imu_data = sensor_data['imu']
        imu_processed = self.imu_processor.preprocess_imu_data(
            imu_data['accel'], imu_data['gyro'], dt
        )
        
        # EKF prediction step using IMU
        self.ekf.ekf_predict(imu_processed, dt)
        
        # Process GPS if available
        gps_position = None
        if 'gps' in sensor_data and sensor_data['gps'] is not None:
            gps_data = sensor_data['gps']
            if self.gps_processor.validate_gps_data(gps_data['lat'], gps_data['lon'], gps_data['alt']):
                # Convert GPS to Cartesian
                gps_position = np.array(self.gps_processor.convert_gps_to_cartesian(
                    gps_data['lat'], gps_data['lon'], gps_data['alt']
                ))
                
                # GPS measurement covariance (simplified)
                gps_covariance = np.eye(3) * 4.0  # 2m standard deviation
                
                # EKF update with GPS
                self.ekf.ekf_update_gps(gps_position, gps_covariance)
        
        # Process stereo visual odometry if available
        vo_pose = None
        if 'stereo' in sensor_data and sensor_data['stereo'] is not None:
            stereo_data = sensor_data['stereo']
            # Compute stereo visual odometry pose
            vo_result = self.stereo_vo.compute_stereo_vo_pose(
                stereo_data['left_image'], stereo_data['right_image']
            )
            
            if vo_result['success']:
                # Extract pose: [x, y, z, roll, pitch, yaw]
                vo_pose = np.concatenate([
                    vo_result['translation'],
                    vo_result['euler_angles']
                ])
                
                # Get VO covariance based on number of features
                vo_covariance = self.stereo_vo.get_stereo_vo_covariance(vo_result['num_features'])
                
                # EKF update with visual odometry
                self.ekf.ekf_update_visual_odometry(vo_pose, vo_covariance)
        
        # Get fused state estimate
        state_estimate = self.ekf.get_state_estimate()
        
        # Log data for analysis
        self._log_data(timestamp, gps_position, imu_processed['position'], 
                      vo_pose, state_estimate)
        
        # Prepare result
        result = {
            'timestamp': timestamp,
            'fused_position': state_estimate['position'],
            'fused_velocity': state_estimate['velocity'],
            'fused_attitude': state_estimate['attitude'],
            'position_covariance': state_estimate['position_cov'],
            'gps_position': gps_position,
            'vo_pose': vo_pose,
            'imu_position': imu_processed['position'],
            'num_vo_features': vo_result['num_features'] if 'stereo' in sensor_data and vo_result['success'] else 0
        }
        
        return result
    
    def _log_data(self, timestamp: float, gps_pos: Optional[np.ndarray], 
                  imu_pos: np.ndarray, vo_pose: Optional[np.ndarray], 
                  state_estimate: Dict) -> None:
        """
        Log sensor data and estimates for later analysis.
        
        Args:
            timestamp: Current timestamp
            gps_pos: GPS position (or None)
            imu_pos: IMU integrated position
            vo_pose: Visual odometry pose (or None)
            state_estimate: EKF state estimate
        """
        self.data_log['timestamps'].append(timestamp)
        self.data_log['gps_positions'].append(gps_pos if gps_pos is not None else np.full(3, np.nan))
        self.data_log['imu_positions'].append(imu_pos)
        self.data_log['vo_positions'].append(vo_pose[:3] if vo_pose is not None else np.full(3, np.nan))
        self.data_log['fused_positions'].append(state_estimate['position'])
        self.data_log['fused_velocities'].append(state_estimate['velocity'])
        self.data_log['fused_attitudes'].append(state_estimate['attitude'])
        self.data_log['position_covariances'].append(state_estimate['position_cov'])
    
    def visualize_results(self) -> None:
        """
        Visualize and analyze the localization results.
        """
        if not self.data_log['timestamps']:
            print("No data to visualize.")
            return
        
        print("Generating visualization plots...")
        
        # Convert lists to numpy arrays
        timestamps = np.array(self.data_log['timestamps'])
        gps_positions = np.array([pos for pos in self.data_log['gps_positions'] if not np.any(np.isnan(pos))])
        imu_positions = np.array(self.data_log['imu_positions'])
        vo_positions = np.array([pos for pos in self.data_log['vo_positions'] if not np.any(np.isnan(pos))])
        fused_positions = np.array(self.data_log['fused_positions'])
        fused_attitudes = np.array(self.data_log['fused_attitudes'])
        position_covariances = np.array(self.data_log['position_covariances'])
        
        # Plot 3D trajectory comparison
        trajectories = []
        labels = []
        
        if len(gps_positions) > 0:
            trajectories.append(gps_positions)
            labels.append('GPS')
        
        trajectories.extend([imu_positions, fused_positions])
        labels.extend(['IMU (integrated)', 'EKF Fused'])
        
        if len(vo_positions) > 0:
            trajectories.append(vo_positions)
            labels.append('Visual Odometry')
        
        self.visualizer.plot_3d_trajectory(trajectories, labels, title="UAV Flight Trajectory")
        
        # Plot sensor comparison
        self.visualizer.plot_sensor_comparison(
            timestamps, gps_positions, imu_positions, vo_positions, fused_positions
        )
        
        # Plot estimation uncertainty
        self.visualizer.plot_estimation_uncertainty(
            timestamps, fused_positions, position_covariances
        )
        
        # Plot attitude estimates
        self.visualizer.plot_attitude_estimates(timestamps, fused_attitudes)
        
        print("Visualization complete!")
    
    def get_performance_metrics(self, ground_truth_positions: np.ndarray = None) -> Dict:
        """
        Calculate performance metrics for the localization system.
        
        Args:
            ground_truth_positions: Ground truth positions for error analysis (optional)
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.data_log['timestamps']:
            return {}
        
        fused_positions = np.array(self.data_log['fused_positions'])
        position_covariances = np.array(self.data_log['position_covariances'])
        
        metrics = {
            'total_time': self.data_log['timestamps'][-1] - self.data_log['timestamps'][0],
            'num_measurements': len(self.data_log['timestamps']),
            'average_position_uncertainty': np.mean(np.sqrt(np.diagonal(position_covariances, axis1=1, axis2=2))),
            'trajectory_length': np.sum(np.linalg.norm(np.diff(fused_positions, axis=0), axis=1))
        }
        
        if ground_truth_positions is not None and len(ground_truth_positions) == len(fused_positions):
            # Calculate error metrics
            position_errors = fused_positions - ground_truth_positions
            error_magnitudes = np.linalg.norm(position_errors, axis=1)
            
            metrics.update({
                'mean_position_error': np.mean(error_magnitudes),
                'rms_position_error': np.sqrt(np.mean(error_magnitudes**2)),
                'max_position_error': np.max(error_magnitudes),
                'position_error_std': np.std(error_magnitudes)
            })
            
            # Plot error analysis
            timestamps = np.array(self.data_log['timestamps'])
            self.visualizer.plot_error_analysis(timestamps, ground_truth_positions, fused_positions)
        
        return metrics


def create_sample_stereo_calibration() -> StereoCalibration:
    """
    Create sample stereo camera calibration parameters.
    
    Returns:
        StereoCalibration object with sample parameters
    """
    # Sample calibration parameters (replace with actual calibration)
    K_left = np.array([
        [718.8560, 0, 607.1928],
        [0, 718.8560, 185.2157],
        [0, 0, 1]
    ])
    
    K_right = np.array([
        [718.8560, 0, 607.1928],
        [0, 718.8560, 185.2157],
        [0, 0, 1]
    ])
    
    D_left = np.array([-0.3691, 0.1968, -0.0019, -0.0001, 0])
    D_right = np.array([-0.3691, 0.1968, -0.0019, -0.0001, 0])
    
    R = np.eye(3)  # Rotation between cameras
    T = np.array([0.12, 0, 0])  # Translation between cameras (12cm baseline)
    baseline = 0.12
    
    return StereoCalibration(K_left, K_right, D_left, D_right, R, T, baseline)


def main():
    """
    Main function demonstrating the UAV localization system.
    """
    print("UAV Localization System Demo")
    print("============================")
    
    # Create sample stereo calibration
    stereo_calibration = create_sample_stereo_calibration()
    
    # Initialize system with reference GPS coordinates
    reference_gps = (40.7589, -73.9851, 100.0)  # Sample coordinates (NYC)
    uav_system = UAVLocalizationSystem(reference_gps, stereo_calibration)
    
    # Sample initial IMU calibration data (stationary measurements)
    initial_imu_samples = {
        'accel': np.random.normal([0, 0, 9.81], 0.1, (100, 3)),  # 100 samples
        'gyro': np.random.normal([0, 0, 0], 0.01, (100, 3))      # 100 samples
    }
    
    # Initialize system
    initial_gps = (40.7589, -73.9851, 100.0)
    uav_system.initialize_system(initial_gps, initial_imu_samples)
    
    # Simulate sensor data processing
    print("Processing simulated sensor data...")
    
    dt = 0.1  # 10 Hz
    num_steps = 100
    
    for i in range(num_steps):
        timestamp = i * dt
        
        # Simulate sensor data
        sensor_data = {
            'timestamp': timestamp,
            'dt': dt,
            'imu': {
                'accel': np.random.normal([0, 0, 9.81], 0.1, 3),
                'gyro': np.random.normal([0, 0, 0], 0.01, 3)
            }
        }
        
        # Add GPS data every 5 steps (2 Hz)
        if i % 5 == 0:
            sensor_data['gps'] = {
                'lat': 40.7589 + np.random.normal(0, 0.00001),
                'lon': -73.9851 + np.random.normal(0, 0.00001),
                'alt': 100.0 + np.random.normal(0, 2.0)
            }
        
        # Process sensor data
        result = uav_system.process_sensor_data(sensor_data)
        
        if i % 20 == 0:  # Print every 2 seconds
            pos = result['fused_position']
            print(f"Time: {timestamp:.1f}s, Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]m")
    
    # Generate performance metrics
    metrics = uav_system.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize results
    uav_system.visualize_results()
    
    print("Demo complete!")


if __name__ == "__main__":
    main()
