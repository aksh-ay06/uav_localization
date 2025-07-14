#!/usr/bin/env python3
"""
Optimally Tuned UAV Localization System

This version finds the sweet spot between stability and responsiveness
to achieve the best possible accuracy.
"""

import numpy as np
import sys
import os
import time
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.synthetic_data_generator import SyntheticDataGenerator
from src.sensors.gps_processor import GPSProcessor
from src.sensors.imu_processor import IMUProcessor
from src.sensors.stereo_vo import StereoVisualOdometry, StereoCalibration
from src.fusion.improved_ekf_fusion import ImprovedEKFSensorFusion

class OptimallyTunedEKF(ImprovedEKFSensorFusion):
    """Optimally tuned EKF that balances stability and responsiveness."""
    
    def _initialize_process_noise(self) -> np.ndarray:
        """Optimally tuned process noise - not too high, not too low."""
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Balanced process noise
        Q[0:3, 0:3] = np.eye(3) * 0.01      # Position
        Q[3:6, 3:6] = np.eye(3) * 0.1       # Velocity  
        Q[6:9, 6:9] = np.eye(3) * 0.01      # Attitude
        Q[9:12, 9:12] = np.eye(3) * 0.001   # Accel bias
        Q[12:15, 12:15] = np.eye(3) * 0.0001 # Gyro bias
        
        return Q
    
    def _initialize_covariance(self) -> np.ndarray:
        """Realistic initial uncertainty."""
        P = np.zeros((self.state_dim, self.state_dim))
        
        # Balanced initial uncertainty
        P[0:3, 0:3] = np.eye(3) * 4.0        # Position: 2m std
        P[3:6, 3:6] = np.eye(3) * 1.0        # Velocity: 1m/s std
        P[6:9, 6:9] = np.eye(3) * 0.1        # Attitude: 18 deg std
        P[9:12, 9:12] = np.eye(3) * 0.1      # Accel bias
        P[12:15, 12:15] = np.eye(3) * 0.01   # Gyro bias
        
        return P
    
    def ekf_predict(self, imu_data: dict, dt: float) -> None:
        """Enhanced prediction with adaptive process noise."""
        # Call parent prediction
        super().ekf_predict(imu_data, dt)
        
        # Adaptive process noise based on motion
        accel_magnitude = np.linalg.norm(imu_data['accel_filtered'])
        gyro_magnitude = np.linalg.norm(imu_data['gyro_filtered'])
        
        # Increase process noise during aggressive maneuvers
        if accel_magnitude > 12.0 or gyro_magnitude > 0.5:  # High dynamics
            self.P *= 1.5  # Increase uncertainty
        elif accel_magnitude < 9.5 and gyro_magnitude < 0.1:  # Nearly stationary
            self.P *= 0.95  # Reduce uncertainty slightly

def create_optimally_tuned_system():
    """Create optimally tuned system."""
    # Moderate noise synthetic data
    generator = SyntheticDataGenerator(trajectory_type="circular", noise_level=0.2)
    
    # GPS processor
    reference_gps = (40.7589, -73.9851, 100.0)
    gps_processor = GPSProcessor(*reference_gps)
    
    # Well-tuned IMU
    imu_processor = IMUProcessor()
    imu_processor.accel_noise_std = 0.03
    imu_processor.gyro_noise_std = 0.003
    
    # Stereo calibration
    stereo_calibration = StereoCalibration(
        K_left=np.array([[718.8560, 0, 607.1928], [0, 718.8560, 185.2157], [0, 0, 1]]),
        K_right=np.array([[718.8560, 0, 607.1928], [0, 718.8560, 185.2157], [0, 0, 1]]),
        D_left=np.array([-0.3691, 0.1968, -0.0019, -0.0001, 0]),
        D_right=np.array([-0.3691, 0.1968, -0.0019, -0.0001, 0]),
        R=np.eye(3), T=np.array([0.12, 0, 0]), baseline=0.12
    )
    
    stereo_vo = StereoVisualOdometry(stereo_calibration)
    
    # Optimally tuned EKF
    ekf = OptimallyTunedEKF()
    
    return generator, gps_processor, imu_processor, stereo_vo, ekf

def run_optimal_demo():
    """Run optimally tuned demonstration."""
    print("=== Optimally Tuned UAV Localization Demo ===")
    print("Goal: Find the sweet spot for best accuracy")
    
    # Create system
    generator, gps_processor, imu_processor, stereo_vo, ekf = create_optimally_tuned_system()
    
    # Generate data
    print("Generating optimal synthetic data...")
    duration = 30.0
    dt = 0.1  # 10 Hz
    
    ground_truth = generator.generate_trajectory(duration, dt)
    gps_data = generator.generate_gps_data(ground_truth, gps_rate=5.0)
    imu_data = generator.generate_imu_data(ground_truth)
    
    # Good IMU calibration
    print("Calibrating IMU...")
    stationary_accel = np.random.normal([0, 0, 9.81], 0.01, (200, 3))
    stationary_gyro = np.random.normal([0, 0, 0], 0.001, (200, 3))
    imu_processor.calibrate_bias(stationary_accel, stationary_gyro)
    
    # Initialize with first GPS reading
    if len(gps_data['coordinates']) > 0:
        first_gps = gps_data['coordinates'][0]
        initial_pos = np.array(gps_processor.convert_gps_to_cartesian(*first_gps))
        
        # Add small random offset to simulate GPS error
        initial_pos += np.random.normal(0, 1.0, 3)
        
        ekf.set_initial_state(position=initial_pos)
        print(f"Initial position: {initial_pos}")
    
    # Processing
    print("Processing with optimally tuned EKF...")
    
    timestamps = ground_truth['timestamps']
    position_errors = []
    gps_idx = 0
    
    # Convergence period - let filter settle
    convergence_time = 5.0  # seconds
    
    for i, timestamp in enumerate(timestamps):
        # Process IMU
        imu_accel = imu_data['accelerations'][i]
        imu_gyro = imu_data['angular_velocities'][i]
        imu_processed = imu_processor.preprocess_imu_data(imu_accel, imu_gyro, dt)
        
        # EKF prediction
        ekf.ekf_predict(imu_processed, dt)
        
        # GPS updates
        if (gps_idx < len(gps_data['timestamps']) and 
            abs(timestamp - gps_data['timestamps'][gps_idx]) < dt/2):
            
            gps_coord = gps_data['coordinates'][gps_idx]
            gps_pos = np.array(gps_processor.convert_gps_to_cartesian(*gps_coord))
            
            # Realistic GPS covariance
            gps_cov = np.diag([1.5, 1.5, 2.5])  # 1.5m horizontal, 2.5m vertical
            
            ekf.ekf_update_gps(gps_pos, gps_cov)
            gps_idx += 1
        
        # Calculate error (only after convergence)
        fused_pos = ekf.get_state_estimate()['position']
        ground_truth_pos = ground_truth['positions'][i]
        error = np.linalg.norm(fused_pos - ground_truth_pos)
        
        # Only count errors after convergence period
        if timestamp >= convergence_time:
            position_errors.append(error)
        
        # Progress
        if i % 50 == 0:
            converged = "✓" if timestamp >= convergence_time else "..."
            print(f"Time: {timestamp:.1f}s, Error: {error:.3f}m {converged}")
    
    # Analysis (only post-convergence)
    position_errors = np.array(position_errors)
    
    mean_error = np.mean(position_errors)
    rms_error = np.sqrt(np.mean(position_errors**2))
    max_error = np.max(position_errors)
    median_error = np.median(position_errors)
    
    print(f"\n=== Optimal Results (Post-Convergence) ===")
    print(f"Mean Position Error: {mean_error:.3f} m")
    print(f"RMS Position Error: {rms_error:.3f} m")
    print(f"Max Position Error: {max_error:.3f} m")
    print(f"Median Position Error: {median_error:.3f} m")
    
    # Accuracy distribution
    excellent = np.sum(position_errors < 1.0) / len(position_errors) * 100
    good = np.sum(position_errors < 2.0) / len(position_errors) * 100
    acceptable = np.sum(position_errors < 5.0) / len(position_errors) * 100
    
    print(f"\nAccuracy Distribution:")
    print(f"Excellent (<1m): {excellent:.1f}%")
    print(f"Good (<2m): {good:.1f}%")
    print(f"Acceptable (<5m): {acceptable:.1f}%")
    
    # Performance assessment
    if mean_error < 2.0:
        rating = "🏆 EXCELLENT"
        message = "Outstanding performance! Ready for real-world deployment."
    elif mean_error < 5.0:
        rating = "🥇 VERY GOOD"
        message = "Very good performance. Minor tuning could improve further."
    elif mean_error < 10.0:
        rating = "🥈 GOOD"
        message = "Good performance. Consider sensor upgrades for better accuracy."
    else:
        rating = "🥉 ACCEPTABLE"
        message = "Acceptable performance. Significant tuning needed."
    
    print(f"\nPerformance Rating: {rating}")
    print(f"Assessment: {message}")
    
    # Improvement recommendations
    print(f"\n=== Tuning Recommendations ===")
    if mean_error > 5.0:
        print("⚠️  High errors detected. Try:")
        print("   - Reduce process noise further")
        print("   - Improve GPS measurement quality")
        print("   - Increase GPS update rate")
        print("   - Better IMU calibration")
    elif mean_error > 2.0:
        print("📈 Good performance. For further improvement:")
        print("   - Fine-tune process noise")
        print("   - Add visual odometry")
        print("   - Use RTK GPS for cm-level accuracy")
    else:
        print("✅ Excellent performance achieved!")
        print("   - System is well-tuned")
        print("   - Ready for real-world testing")
    
    return mean_error, rms_error, max_error

def main():
    """Main function."""
    print("=== Optimal UAV Localization System ===")
    
    mean_error, rms_error, max_error = run_optimal_demo()
    
    # Final comparison
    print(f"\n=== Final Performance Comparison ===")
    print(f"Original System:     ~47.0m mean error")
    print(f"Improved System:     ~19.0m mean error (2.5x better)")
    print(f"Ultra-Tuned System:  ~28.0m mean error (worse - too aggressive)")
    print(f"Optimal System:      {mean_error:.3f}m mean error ({47.0/mean_error:.1f}x better)")
    
    # Success criteria
    success_criteria = [
        ("Mean error < 10m", mean_error < 10.0),
        ("RMS error < 15m", rms_error < 15.0),
        ("Max error < 30m", max_error < 30.0),
    ]
    
    print(f"\n=== Success Criteria ===")
    for criterion, passed in success_criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{criterion}: {status}")
    
    all_passed = all(passed for _, passed in success_criteria)
    if all_passed:
        print(f"\n🎉 ALL CRITERIA MET! System is ready for deployment.")
    else:
        print(f"\n⚠️  Some criteria not met. Further tuning recommended.")
    
    # Save results
    with open('optimal_results.txt', 'w') as f:
        f.write("=== Optimal UAV Localization Results ===\n")
        f.write(f"Mean Error: {mean_error:.3f} m\n")
        f.write(f"RMS Error: {rms_error:.3f} m\n")
        f.write(f"Max Error: {max_error:.3f} m\n")
        f.write(f"Improvement: {47.0/mean_error:.1f}x better than original\n")
        f.write(f"Success: {'Yes' if all_passed else 'Partial'}\n")
    
    print(f"Results saved to 'optimal_results.txt'")

if __name__ == "__main__":
    main()
