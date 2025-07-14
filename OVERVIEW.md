# UAV Localization System - Complete Implementation

## Overview

This is a comprehensive UAV localization system that uses sensor fusion via an Extended Kalman Filter (EKF) to combine:

- **GPS**: Low-frequency global position estimates
- **IMU**: High-frequency inertial data (accelerations and angular velocities)  
- **Stereo Vision**: Medium-frequency visual odometry for local pose estimation

The system is implemented following GitHub Copilot best practices with clear, descriptive comments and modular design.

## Project Structure

```
uav_localization/
├── src/
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── gps_processor.py      # GPS coordinate conversion and processing
│   │   ├── imu_processor.py      # IMU data preprocessing and integration
│   │   └── stereo_vo.py          # Stereo visual odometry
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── ekf_fusion.py         # Extended Kalman Filter implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── visualization.py      # Plotting and visualization tools
│   └── main.py                   # Main localization system class
├── data/
│   └── synthetic_data_generator.py  # Generate test data
├── tests/
│   └── test_uav_localization.py     # Unit tests
├── demo.py                           # Complete demonstration
├── setup.sh                         # Setup script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Key Features

### 1. GPS Processor (`gps_processor.py`)
- Converts GPS coordinates (lat/lon/alt) to local Cartesian coordinates
- Validates GPS data for reasonable bounds
- Handles coordinate transformations using pyproj
- Provides accuracy estimates based on dilution of precision

### 2. IMU Processor (`imu_processor.py`)  
- Processes accelerometer and gyroscope data
- Applies bias correction and noise filtering
- Integrates IMU data for pose increments
- Handles gravity compensation and coordinate transformations

### 3. Stereo Visual Odometry (`stereo_vo.py`)
- Processes stereo camera images for visual odometry
- Detects and matches features between frames
- Estimates camera motion using PnP algorithm
- Provides pose estimates with uncertainty quantification

### 4. Extended Kalman Filter (`ekf_fusion.py`)
- **State vector**: [position(3), velocity(3), attitude(3), accel_bias(3), gyro_bias(3)]
- **Prediction step**: Uses IMU measurements to predict state evolution
- **Update steps**: Incorporates GPS and visual odometry measurements
- Handles non-linear motion models and measurement functions

### 5. Visualization Tools (`visualization.py`)
- 3D trajectory plotting
- Sensor comparison plots
- Uncertainty visualization
- Error analysis with statistics
- Real-time plotting capabilities

## Installation and Setup

1. **Clone and navigate to the project**:
   ```bash
   cd /home/akshay/uav_localization
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Or manually install**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Running the Demo

The complete system demonstration:
```bash
source .venv/bin/activate
python demo.py
```

This will:
- Generate synthetic sensor data
- Run the complete sensor fusion pipeline
- Display performance metrics
- Generate visualization plots

### Running Tests

Execute the unit test suite:
```bash
source .venv/bin/activate
python tests/test_uav_localization.py
```

### Using Individual Components

```python
from src.sensors.gps_processor import GPSProcessor
from src.sensors.imu_processor import IMUProcessor
from src.fusion.ekf_fusion import EKFSensorFusion

# Initialize GPS processor with reference coordinates
gps_processor = GPSProcessor(40.7589, -73.9851, 100.0)

# Convert GPS to Cartesian
x, y, z = gps_processor.convert_gps_to_cartesian(lat, lon, alt)

# Process IMU data
imu_processor = IMUProcessor()
imu_result = imu_processor.preprocess_imu_data(accel, gyro, dt)

# Run EKF sensor fusion
ekf = EKFSensorFusion()
ekf.ekf_predict(imu_result, dt)
ekf.ekf_update_gps(gps_position, gps_covariance)
```

## Implementation Highlights

### GitHub Copilot Best Practices Applied

1. **Clear Function Comments**:
   ```python
   # Function: convert_gps_to_cartesian
   # Input: Latitude, longitude, altitude
   # Output: Local Cartesian (X,Y,Z) coordinates relative to a reference point
   ```

2. **Descriptive Variable Names**:
   ```python
   # State vector definition:
   # [x, y, z, vx, vy, vz, roll, pitch, yaw, accel_bias_x, accel_bias_y, accel_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
   ```

3. **Modular Design**:
   - Single-purpose functions
   - Clear separation of concerns
   - Reusable components

4. **Incremental Development**:
   - Each component can be developed and tested independently
   - Clear interfaces between modules

### Sensor Fusion Main Loop

```python
# Main Sensor Fusion Loop
# Inputs: synchronized IMU, GPS, Stereo data
# Outputs: EKF fused pose estimates for UAV localization
for timestep in timestamps:
    # preprocess IMU data
    imu_processed = preprocess_imu_data(accel[timestep], gyro[timestep], dt)

    # stereo visual odometry pose
    vo_pose = compute_stereo_vo_pose(left_images[timestep], right_images[timestep])

    # GPS conversion
    gps_xyz = convert_gps_to_cartesian(lat[timestep], lon[timestep], alt[timestep])

    # EKF prediction and update
    ekf_predict(imu_processed)
    ekf_update(gps_xyz, vo_pose)

    # visualize or log fused pose
```

## Performance Metrics

The demo shows typical performance metrics:
- **Mean Position Error**: ~2-5 meters (depends on sensor noise levels)
- **RMS Position Error**: ~3-7 meters
- **Processing Rate**: Real-time capable (>10 Hz on typical hardware)

## Key Algorithms

### 1. Extended Kalman Filter Prediction
```python
# Predict attitude using gyroscope
att_new = att + gyro_corrected * dt

# Transform acceleration to world frame  
R_body_to_world = euler_to_rotation_matrix(att_new)
accel_world = R_body_to_world @ accel_corrected - gravity

# Predict velocity and position
vel_new = vel + accel_world * dt
pos_new = pos + vel * dt + 0.5 * accel_world * dt**2
```

### 2. GPS Update Step
```python
# Innovation
y = gps_measurement - state[0:3]  # GPS measures position directly

# Kalman gain and state update
K = P @ H.T @ inv(S)
state = state + K @ y
P = (I - K @ H) @ P
```

### 3. Visual Odometry Pose Estimation
```python
# Solve PnP problem for pose estimation
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    object_points, image_points, camera_matrix, distortion_coeffs
)
```

## Extension Points

The system is designed for easy extension:

1. **Additional Sensors**: Add magnetometer, barometer, LIDAR
2. **Advanced Algorithms**: Implement UKF, particle filters, or GraphSLAM
3. **Real Hardware**: Interface with actual GPS, IMU, and camera hardware
4. **Machine Learning**: Add learned components for bias estimation or feature matching

## Dependencies

- **NumPy**: Numerical computations
- **SciPy**: Scientific computing and linear algebra
- **OpenCV**: Computer vision and image processing
- **FilterPy**: Kalman filter implementations (reference)
- **Matplotlib**: Visualization and plotting
- **PyProj**: GPS coordinate transformations

## Testing

The system includes comprehensive unit tests covering:
- GPS coordinate conversion
- IMU data processing
- EKF state prediction and updates
- System integration workflows

## Troubleshooting

1. **Import Errors**: Ensure virtual environment is activated and packages are installed
2. **Visualization Issues**: Some environments may not support interactive plotting
3. **Performance**: Adjust noise levels and sensor rates for your specific use case

## Future Improvements

1. **Real-time Optimization**: Implement more efficient algorithms for embedded systems
2. **Robustness**: Add outlier detection and sensor health monitoring
3. **Calibration**: Automatic sensor calibration procedures
4. **Advanced Vision**: Deep learning-based visual odometry
5. **Multi-UAV**: Extend to collaborative localization scenarios

This implementation provides a solid foundation for UAV localization research and development, following modern software engineering practices and providing comprehensive documentation for future development.
