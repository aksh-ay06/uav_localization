# UAV Localization System

A UAV localization system using sensor fusion via an Extended Kalman Filter (EKF) that combines:
- GPS: Low-frequency global position estimates
- IMU: High-frequency inertial data (accelerations and angular velocities)
- Stereo Vision: Medium-frequency visual odometry for local pose estimation

## Project Structure

```
uav_localization/
├── src/
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── gps_processor.py
│   │   ├── imu_processor.py
│   │   └── stereo_vo.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── ekf_fusion.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── visualization.py
│   └── main.py
├── data/
│   ├── sample_data/
│   └── calibration/
├── tests/
├── requirements.txt
└── README.md
```

## Dependencies

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main localization system:
```bash
python src/main.py
```

## Components

### Sensors
- **GPS Processor**: Converts GPS coordinates to local Cartesian coordinates
- **IMU Processor**: Processes accelerometer and gyroscope data
- **Stereo Visual Odometry**: Estimates pose from stereo camera images

### Fusion
- **EKF Fusion**: Extended Kalman Filter for sensor fusion

### Utilities
- **Visualization**: Tools for plotting and visualizing results
