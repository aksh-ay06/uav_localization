"""
Unit tests for UAV Localization System components.

This module contains unit tests for the GPS processor, IMU processor,
stereo visual odometry, and EKF fusion components.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sensors.gps_processor import GPSProcessor
from sensors.imu_processor import IMUProcessor
from fusion.ekf_fusion import EKFSensorFusion

class TestGPSProcessor(unittest.TestCase):
    """Test cases for GPS processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gps_processor = GPSProcessor(40.7589, -73.9851, 100.0)
    
    def test_gps_validation(self):
        """Test GPS data validation."""
        # Valid GPS data
        self.assertTrue(self.gps_processor.validate_gps_data(40.7589, -73.9851, 100.0))
        
        # Invalid latitude
        self.assertFalse(self.gps_processor.validate_gps_data(91.0, -73.9851, 100.0))
        
        # Invalid longitude
        self.assertFalse(self.gps_processor.validate_gps_data(40.7589, 181.0, 100.0))
        
        # Invalid altitude
        self.assertFalse(self.gps_processor.validate_gps_data(40.7589, -73.9851, 15000.0))
    
    def test_coordinate_conversion(self):
        """Test GPS to Cartesian coordinate conversion."""
        # Convert reference point should give (0, 0, 0)
        x, y, z = self.gps_processor.convert_gps_to_cartesian(40.7589, -73.9851, 100.0)
        self.assertAlmostEqual(x, 0.0, places=1)
        self.assertAlmostEqual(y, 0.0, places=1)
        self.assertAlmostEqual(z, 0.0, places=1)
        
        # Test with offset coordinates
        x, y, z = self.gps_processor.convert_gps_to_cartesian(40.7590, -73.9850, 101.0)
        self.assertGreater(abs(x), 0)  # Should have non-zero x
        self.assertGreater(abs(y), 0)  # Should have non-zero y
        self.assertAlmostEqual(z, 1.0, places=1)  # 1m altitude difference

class TestIMUProcessor(unittest.TestCase):
    """Test cases for IMU processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.imu_processor = IMUProcessor()
    
    def test_bias_calibration(self):
        """Test IMU bias calibration."""
        # Generate stationary IMU samples with known bias
        accel_bias = np.array([0.1, 0.05, 0.2])
        gyro_bias = np.array([0.01, -0.02, 0.005])
        
        accel_samples = np.random.normal([0, 0, 9.81], 0.1, (100, 3)) + accel_bias
        gyro_samples = np.random.normal([0, 0, 0], 0.01, (100, 3)) + gyro_bias
        
        self.imu_processor.calibrate_bias(accel_samples, gyro_samples)
        
        # Check if calibrated bias is close to expected
        np.testing.assert_allclose(self.imu_processor.accel_bias, accel_bias, atol=0.1)
        np.testing.assert_allclose(self.imu_processor.gyro_bias, gyro_bias, atol=0.05)
    
    def test_imu_preprocessing(self):
        """Test IMU data preprocessing."""
        # Test with sample data
        accel = np.array([0.1, 0.2, 9.9])
        gyro = np.array([0.01, -0.02, 0.05])
        dt = 0.1
        
        result = self.imu_processor.preprocess_imu_data(accel, gyro, dt)
        
        # Check if result contains expected keys
        expected_keys = ['accel_filtered', 'gyro_filtered', 'delta_position', 
                        'delta_velocity', 'delta_orientation', 'position', 
                        'velocity', 'orientation']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check data types and shapes
        self.assertEqual(result['accel_filtered'].shape, (3,))
        self.assertEqual(result['gyro_filtered'].shape, (3,))
        self.assertEqual(result['delta_position'].shape, (3,))
        self.assertEqual(result['orientation'].shape, (3, 3))
    
    def test_euler_conversion(self):
        """Test conversion between rotation matrix and Euler angles."""
        # Test with known rotation matrix (45 degree rotation about z-axis)
        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        roll, pitch, yaw = self.imu_processor.euler_from_rotation_matrix(R)
        
        # Should have zero roll and pitch, pi/4 yaw
        self.assertAlmostEqual(roll, 0.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(yaw, angle, places=5)

class TestEKFSensorFusion(unittest.TestCase):
    """Test cases for EKF sensor fusion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ekf = EKFSensorFusion()
    
    def test_state_initialization(self):
        """Test EKF state initialization."""
        # Test default initialization
        self.assertEqual(self.ekf.state.shape, (15,))
        self.assertEqual(self.ekf.P.shape, (15, 15))
        
        # Test custom initialization
        initial_pos = np.array([1.0, 2.0, 3.0])
        initial_vel = np.array([0.1, 0.2, 0.3])
        initial_att = np.array([0.1, 0.05, 0.2])
        
        self.ekf.set_initial_state(initial_pos, initial_vel, initial_att)
        
        np.testing.assert_array_equal(self.ekf.state[0:3], initial_pos)
        np.testing.assert_array_equal(self.ekf.state[3:6], initial_vel)
        np.testing.assert_array_equal(self.ekf.state[6:9], initial_att)
    
    def test_gps_update(self):
        """Test GPS measurement update."""
        # Set initial state
        initial_pos = np.array([0.0, 0.0, 0.0])
        self.ekf.set_initial_state(initial_pos)
        
        # Simulate GPS measurement
        gps_measurement = np.array([1.0, 2.0, 3.0])
        gps_covariance = np.eye(3) * 4.0
        
        # Perform GPS update
        self.ekf.ekf_update_gps(gps_measurement, gps_covariance)
        
        # Position should be updated towards GPS measurement
        updated_pos = self.ekf.state[0:3]
        
        # Check that position moved towards GPS measurement
        for i in range(3):
            self.assertGreater(updated_pos[i], initial_pos[i])
            self.assertLess(updated_pos[i], gps_measurement[i])
    
    def test_angle_normalization(self):
        """Test angle normalization function."""
        # Test angles that need normalization
        angles = np.array([2*np.pi + 0.1, -2*np.pi - 0.1, 3*np.pi])
        normalized = self.ekf._normalize_angles(angles)
        
        # All angles should be in [-pi, pi] range
        self.assertTrue(np.all(normalized >= -np.pi))
        self.assertTrue(np.all(normalized <= np.pi))
        
        # Check specific values
        self.assertAlmostEqual(normalized[0], 0.1, places=5)
        self.assertAlmostEqual(normalized[1], -0.1, places=5)
        self.assertAlmostEqual(normalized[2], -np.pi, places=5)
    
    def test_euler_to_rotation_matrix(self):
        """Test Euler angles to rotation matrix conversion."""
        # Test with zero angles
        euler = np.array([0.0, 0.0, 0.0])
        R = self.ekf._euler_to_rotation_matrix(euler)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        
        # Test with 90 degree rotation about z-axis
        euler = np.array([0.0, 0.0, np.pi/2])
        R = self.ekf._euler_to_rotation_matrix(euler)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_basic_workflow(self):
        """Test basic sensor fusion workflow."""
        # This test verifies that all components work together
        # without actual sensor data
        
        # Initialize components
        gps_processor = GPSProcessor(40.7589, -73.9851, 100.0)
        imu_processor = IMUProcessor()
        ekf = EKFSensorFusion()
        
        # Set initial state
        initial_pos = np.array([0.0, 0.0, 0.0])
        ekf.set_initial_state(initial_pos)
        
        # Simulate one step of sensor fusion
        dt = 0.1
        
        # Simulate IMU data
        accel = np.array([0.1, 0.2, 9.9])
        gyro = np.array([0.01, -0.02, 0.05])
        imu_processed = imu_processor.preprocess_imu_data(accel, gyro, dt)
        
        # EKF predict
        ekf.ekf_predict(imu_processed, dt)
        
        # Simulate GPS update
        gps_pos = np.array([1.0, 1.0, 1.0])
        gps_cov = np.eye(3) * 4.0
        ekf.ekf_update_gps(gps_pos, gps_cov)
        
        # Get state estimate
        state = ekf.get_state_estimate()
        
        # Verify state estimate structure
        self.assertIn('position', state)
        self.assertIn('velocity', state)
        self.assertIn('attitude', state)
        self.assertEqual(state['position'].shape, (3,))
        self.assertEqual(state['velocity'].shape, (3,))
        self.assertEqual(state['attitude'].shape, (3,))

def run_tests():
    """Run all unit tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestGPSProcessor))
    test_suite.addTest(unittest.makeSuite(TestIMUProcessor))
    test_suite.addTest(unittest.makeSuite(TestEKFSensorFusion))
    test_suite.addTest(unittest.makeSuite(TestSystemIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
