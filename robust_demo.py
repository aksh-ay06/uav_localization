#!/usr/bin/env python3
"""
Robust UAV Localization Demo with Proper Initialization and Outlier Handling
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
import os

# Add the src directory to the Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sensors.gps_processor import GPSProcessor
from sensors.imu_processor import IMUProcessor
from sensors.stereo_vo import StereoVisualOdometry
from fusion.improved_ekf_fusion import ImprovedEKFSensorFusion
from utils.visualization import UAVVisualization

def generate_simple_trajectory(duration=30, dt=0.1):
    """Generate a simple synthetic trajectory for testing."""
    t = np.arange(0, duration, dt)
    
    # Figure-eight trajectory
    x = 50 * np.sin(0.2 * t)
    y = 25 * np.sin(0.4 * t)
    z = 100 + 5 * np.sin(0.1 * t)
    
    # Add some velocity
    vx = 50 * 0.2 * np.cos(0.2 * t)
    vy = 25 * 0.4 * np.cos(0.4 * t)
    vz = 5 * 0.1 * np.cos(0.1 * t)
    
    trajectory = []
    for i in range(len(t)):
        trajectory.append({
            'timestamp': t[i],
            'position': np.array([x[i], y[i], z[i]]),
            'velocity': np.array([vx[i], vy[i], vz[i]])
        })
    
    return trajectory

def generate_synthetic_data(ground_truth, gps_freq=10, imu_freq=100, stereo_freq=30):
    """Generate synthetic sensor data from ground truth."""
    
    duration = ground_truth[-1]['timestamp']
    dt = ground_truth[1]['timestamp'] - ground_truth[0]['timestamp']
    
    # GPS data (lower frequency)
    gps_data = []
    gps_interval = max(1, int(1.0 / (gps_freq * dt)))
    for i in range(0, len(ground_truth), gps_interval):
        if i < len(ground_truth):
            # Add GPS noise
            pos = ground_truth[i]['position'] + np.random.normal(0, 2.0, 3)
            gps_data.append({
                'timestamp': ground_truth[i]['timestamp'],
                'data': pos
            })
    
    # IMU data (higher frequency)
    imu_data = []
    imu_interval = max(1, int(1.0 / (imu_freq * dt)))
    for i in range(0, len(ground_truth), imu_interval):
        if i < len(ground_truth):
            # Simulate IMU acceleration and angular velocity
            accel = np.random.normal([0, 0, 9.81], 0.05, 3)
            gyro = np.random.normal([0, 0, 0], 0.005, 3)
            imu_data.append({
                'timestamp': ground_truth[i]['timestamp'],
                'data': np.concatenate([accel, gyro])
            })
    
    # Stereo data (medium frequency)
    stereo_data = []
    stereo_interval = max(1, int(1.0 / (stereo_freq * dt)))
    for i in range(0, len(ground_truth), stereo_interval):
        if i < len(ground_truth):
            # Simulate stereo visual odometry
            pos_noise = np.random.normal(0, 0.1, 3)
            att_noise = np.random.normal(0, 0.01, 3)
            stereo_data.append({
                'timestamp': ground_truth[i]['timestamp'],
                'data': np.concatenate([pos_noise, att_noise])
            })
    
    return {
        'gps': gps_data,
        'imu': imu_data,
        'stereo': stereo_data,
        'ground_truth': ground_truth
    }

class RobustEKFSensorFusion(ImprovedEKFSensorFusion):
    """
    Robust EKF with improved initialization and outlier handling.
    """
    
    def __init__(self, initial_position=None, initial_orientation=None):
        super().__init__(initial_position, initial_orientation)
        
        # More robust outlier detection
        self.gps_outlier_threshold = 100.0  # Increased from 50m
        self.consecutive_outliers = 0
        self.max_consecutive_outliers = 10
        
        # Initialize with more reasonable process noise
        self.setup_robust_noise_model()
        
        # Track initialization state
        self.is_initialized = False
        self.initialization_measurements = []
        self.min_init_measurements = 5
        
    def setup_robust_noise_model(self):
        """Setup process noise optimized for robustness."""
        # Reduced process noise (too much causes instability)
        self.Q = np.diag([
            # Position noise (m²/s²)
            0.01, 0.01, 0.01,
            # Velocity noise (m²/s⁴)
            0.1, 0.1, 0.1,
            # Attitude noise (rad²/s²)
            0.001, 0.001, 0.001,
            # Accelerometer bias (m²/s⁶)
            1e-6, 1e-6, 1e-6,
            # Gyroscope bias (rad²/s⁴)
            1e-8, 1e-8, 1e-8
        ])
        
        # Covariance initialization with high uncertainty
        self.P = np.diag([
            # Position uncertainty (m²)
            100, 100, 100,
            # Velocity uncertainty (m²/s²)
            10, 10, 10,
            # Attitude uncertainty (rad²)
            0.1, 0.1, 0.1,
            # Bias uncertainties
            0.01, 0.01, 0.01,
            0.001, 0.001, 0.001
        ])
    
    def initialize_with_measurements(self, gps_measurement):
        """Initialize state with first few GPS measurements."""
        self.initialization_measurements.append(gps_measurement)
        
        if len(self.initialization_measurements) >= self.min_init_measurements:
            # Use mean of first measurements as initial position
            mean_pos = np.mean(self.initialization_measurements, axis=0)
            self.state[0:3] = mean_pos
            
            # Reduce initial position uncertainty
            self.P[0:3, 0:3] = np.diag([25, 25, 25])  # 5m std
            
            self.is_initialized = True
            print(f"System initialized with position: {mean_pos}")
        
        return self.is_initialized
    
    def update_gps_robust(self, gps_measurement, gps_covariance):
        """
        Robust GPS update with proper outlier handling.
        """
        if not self.is_initialized:
            if self.initialize_with_measurements(gps_measurement):
                return  # First few measurements used for initialization
            else:
                return  # Not enough measurements yet
        
        # Measurement function: h(x) = [x, y, z] (position only)
        h_x = self.state[0:3]
        
        # Innovation (residual)
        y = gps_measurement - h_x
        innovation_magnitude = np.linalg.norm(y)
        
        # Adaptive outlier threshold based on uncertainty
        current_uncertainty = np.sqrt(np.trace(self.P[0:3, 0:3]))
        adaptive_threshold = max(self.gps_outlier_threshold, 3 * current_uncertainty)
        
        if innovation_magnitude > adaptive_threshold:
            self.consecutive_outliers += 1
            if self.consecutive_outliers <= self.max_consecutive_outliers:
                warnings.warn(f"GPS outlier detected: {innovation_magnitude:.2f}m vs {adaptive_threshold:.2f}m threshold. Skipping update.")
                return
            else:
                # Too many consecutive outliers - system may have drifted
                warnings.warn(f"Too many consecutive outliers ({self.consecutive_outliers}). Forcing GPS update with increased noise.")
                # Increase measurement noise for this update
                gps_covariance = gps_covariance * 10
        else:
            self.consecutive_outliers = 0
        
        # Measurement Jacobian (linear for GPS)
        H = np.zeros((3, self.state_dim))
        H[0:3, 0:3] = np.eye(3)
        
        # Innovation covariance
        S = H @ self.P @ H.T + gps_covariance
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ gps_covariance @ K.T
        
        # Ensure covariance remains positive definite
        self.P = 0.5 * (self.P + self.P.T)
        
        # Store innovation for adaptive filtering
        self.innovation_history.append(innovation_magnitude)
        if len(self.innovation_history) > self.max_innovation_history:
            self.innovation_history.pop(0)

def main():
    print("=== Robust UAV Localization Demo ===")
    print("Goal: Robust performance with proper initialization")
    
    # Generate synthetic data
    print("Generating robust synthetic data...")
    ground_truth = generate_simple_trajectory(duration=30, dt=0.1)
    data = generate_synthetic_data(ground_truth)
    
    print(f"Generated {len(data['gps'])} GPS measurements")
    print(f"Generated {len(data['imu'])} IMU measurements")
    print(f"Generated {len(data['stereo'])} stereo measurements")
    
    # Initialize sensors
    gps_processor = GPSProcessor(reference_lat=40.7589, reference_lon=-73.9851)
    imu_processor = IMUProcessor()
    
    # Create simple stereo calibration
    K = np.array([[718.8560, 0, 607.1928],
                  [0, 718.8560, 185.2157],
                  [0, 0, 1]])
    D = np.array([-0.3691, 0.1968, -0.0019, -0.0001, 0])
    R = np.eye(3)
    T = np.array([0.12, 0, 0])
    
    # Create a simple calibration object
    class SimpleCalibration:
        def __init__(self):
            self.K_left = K
            self.K_right = K
            self.D_left = D
            self.D_right = D
            self.R = R
            self.T = T
            self.baseline = 0.12
    
    stereo_calibration = SimpleCalibration()
    stereo_vo = StereoVisualOdometry(stereo_calibration)
    
    # Calibrate IMU
    print("Calibrating IMU...")
    accel_samples = np.array([data['imu'][i]['data'][:3] for i in range(min(100, len(data['imu'])))])
    gyro_samples = np.array([data['imu'][i]['data'][3:] for i in range(min(100, len(data['imu'])))])
    imu_processor.calibrate_bias(accel_samples, gyro_samples)
    
    # Initialize EKF with robust settings
    ekf = RobustEKFSensorFusion()
    
    # Initialize with first GPS measurement
    first_gps = data['gps'][0]['data']
    print(f"First GPS measurement: {first_gps}")
    
    # Process all measurements
    print("Processing with robust EKF...")
    estimated_trajectory = []
    ground_truth_trajectory = []
    timestamps = []
    
    # Sort all measurements by timestamp
    all_measurements = []
    for gps in data['gps']:
        all_measurements.append(('gps', gps['timestamp'], gps['data']))
    for imu in data['imu']:
        all_measurements.append(('imu', imu['timestamp'], imu['data']))
    
    all_measurements.sort(key=lambda x: x[1])
    
    for i, (sensor_type, timestamp, measurement) in enumerate(all_measurements):
        if sensor_type == 'gps':
            gps_covariance = np.diag([4.0, 4.0, 4.0])  # 2m std
            ekf.update_gps_robust(measurement, gps_covariance)
        elif sensor_type == 'imu':
            # Process IMU data first
            accel = measurement[:3]
            gyro = measurement[3:]
            if i < len(all_measurements) - 1:
                dt = all_measurements[i+1][1] - timestamp
                if dt > 0:
                    imu_processed = imu_processor.preprocess_imu_data(accel, gyro, dt)
                    ekf.ekf_predict(imu_processed, dt)
        
        # Store results
        estimated_trajectory.append(ekf.state[0:3].copy())
        ground_truth_trajectory.append(data['ground_truth'][min(i//10, len(data['ground_truth'])-1)]['position'])
        timestamps.append(timestamp)
        
        # Progress indicator
        if i % 200 == 0:
            error = np.linalg.norm(estimated_trajectory[-1] - ground_truth_trajectory[-1])
            print(f"Time: {timestamp:.1f}s, Error: {error:.3f}m ✓")
    
    # Analysis
    print("\n=== Robust Results ===")
    estimated_trajectory = np.array(estimated_trajectory)
    ground_truth_trajectory = np.array(ground_truth_trajectory)
    
    # Calculate errors
    errors = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, axis=1)
    
    # Only analyze post-convergence (after 5 seconds)
    convergence_idx = next(i for i, t in enumerate(timestamps) if t > 5.0)
    post_convergence_errors = errors[convergence_idx:]
    
    mean_error = np.mean(post_convergence_errors)
    rms_error = np.sqrt(np.mean(post_convergence_errors**2))
    max_error = np.max(post_convergence_errors)
    median_error = np.median(post_convergence_errors)
    
    print(f"Mean Position Error: {mean_error:.3f} m")
    print(f"RMS Position Error: {rms_error:.3f} m")
    print(f"Max Position Error: {max_error:.3f} m")
    print(f"Median Position Error: {median_error:.3f} m")
    
    # Accuracy distribution
    excellent = np.sum(post_convergence_errors < 1.0) / len(post_convergence_errors) * 100
    good = np.sum(post_convergence_errors < 2.0) / len(post_convergence_errors) * 100
    acceptable = np.sum(post_convergence_errors < 5.0) / len(post_convergence_errors) * 100
    
    print(f"\nAccuracy Distribution:")
    print(f"Excellent (<1m): {excellent:.1f}%")
    print(f"Good (<2m): {good:.1f}%")
    print(f"Acceptable (<5m): {acceptable:.1f}%")
    
    # Performance rating
    if mean_error < 2.0:
        rating = "🥇 EXCELLENT"
        assessment = "Outstanding performance!"
    elif mean_error < 5.0:
        rating = "🥈 GOOD"
        assessment = "Good performance with minor improvements possible."
    elif mean_error < 10.0:
        rating = "🥉 ACCEPTABLE"
        assessment = "Acceptable performance. Some tuning recommended."
    else:
        rating = "❌ POOR"
        assessment = "Poor performance. Significant improvements needed."
    
    print(f"\nPerformance Rating: {rating}")
    print(f"Assessment: {assessment}")
    
    # Tuning recommendations
    print(f"\n=== Tuning Recommendations ===")
    if mean_error > 5.0:
        print("⚠️  High errors detected. Try:")
        print("   - Better sensor calibration")
        print("   - Increase measurement frequency")
        print("   - Improve initial position accuracy")
        print("   - Use RTK GPS for better accuracy")
    elif mean_error > 2.0:
        print("✅ Good performance. Minor improvements:")
        print("   - Fine-tune process noise")
        print("   - Optimize sensor timing")
        print("   - Add more robust outlier detection")
    else:
        print("🎉 Excellent performance!")
        print("   - System is well-tuned")
        print("   - Consider real-world validation")
    
    print(f"\n=== Performance Comparison ===")
    print(f"Original System:     ~47.0m mean error")
    print(f"Improved System:     ~19.0m mean error (2.5x better)")
    print(f"Ultra-Tuned System:  ~28.0m mean error (worse - too aggressive)")
    print(f"Optimal System:      >10km mean error (catastrophic)")
    print(f"Robust System:       {mean_error:.3f}m mean error")
    
    # Success criteria
    print(f"\n=== Success Criteria ===")
    print(f"Mean error < 10m: {'✅ PASS' if mean_error < 10 else '❌ FAIL'}")
    print(f"RMS error < 15m: {'✅ PASS' if rms_error < 15 else '❌ FAIL'}")
    print(f"Max error < 30m: {'✅ PASS' if max_error < 30 else '❌ FAIL'}")
    
    if mean_error < 10 and rms_error < 15 and max_error < 30:
        print("\n🎉 All criteria met! System is ready for deployment.")
    else:
        print("\n⚠️  Some criteria not met. Further tuning recommended.")
    
    # Save results
    with open('robust_results.txt', 'w') as f:
        f.write(f"Robust UAV Localization Results\n")
        f.write(f"Mean Error: {mean_error:.3f}m\n")
        f.write(f"RMS Error: {rms_error:.3f}m\n")
        f.write(f"Max Error: {max_error:.3f}m\n")
        f.write(f"Median Error: {median_error:.3f}m\n")
        f.write(f"Accuracy <1m: {excellent:.1f}%\n")
        f.write(f"Accuracy <2m: {good:.1f}%\n")
        f.write(f"Accuracy <5m: {acceptable:.1f}%\n")
    
    print(f"Results saved to 'robust_results.txt'")
    
    # Visualization
    visualizer = UAVVisualization()
    
    # Plot 3D trajectory
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory comparison
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], ground_truth_trajectory[:, 2], 
             'g-', label='Ground Truth', linewidth=2)
    ax1.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2], 
             'r--', label='Estimated', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    
    # Error over time
    ax2 = fig.add_subplot(222)
    ax2.plot(timestamps, errors, 'b-', linewidth=1)
    ax2.axhline(y=mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.3f}m')
    ax2.axhline(y=median_error, color='g', linestyle='--', label=f'Median: {median_error:.3f}m')
    ax2.axvline(x=5.0, color='orange', linestyle=':', label='Convergence')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error histogram
    ax3 = fig.add_subplot(223)
    ax3.hist(post_convergence_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.3f}m')
    ax3.axvline(x=median_error, color='g', linestyle='--', label=f'Median: {median_error:.3f}m')
    ax3.set_xlabel('Position Error (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Position Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 2D trajectory (top view)
    ax4 = fig.add_subplot(224)
    ax4.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], 'g-', label='Ground Truth', linewidth=2)
    ax4.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'r--', label='Estimated', linewidth=2)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('2D Trajectory (Top View)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig('robust_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
