# UAV Localization System

A real-time UAV localization system using Extended Kalman Filter (EKF) for multi-sensor fusion.

## 🚁 Overview

This project implements a comprehensive UAV positioning system that fuses data from multiple sensors:
- **GPS**: Global position estimates (10Hz)
- **IMU**: High-frequency inertial measurements (100Hz)
- **Stereo Vision**: Visual odometry for pose estimation (30Hz)

## 📊 Performance

- **Positioning Accuracy**: 19.4m mean error (2.4x improvement over baseline)
- **Real-time Processing**: 100Hz sensor fusion
- **Robust Operation**: Handles sensor dropouts and outliers
- **Processing Rate**: 1300+ Hz data throughput

## 🛠️ Technologies

- **Python**: NumPy, SciPy, OpenCV, FilterPy, Matplotlib
- **Algorithms**: 15-state Extended Kalman Filter, Computer Vision, Signal Processing
- **Testing**: Comprehensive unit tests with 95% coverage
- **Coordinate Systems**: GPS/UTM conversion, stereo camera calibration

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/aksh-ay06/uav_localization.git
cd uav_localization

# Setup environment
chmod +x setup.sh
./setup.sh

# Run demo
source .venv/bin/activate
python demo.py
```

## 📁 Project Structure

```
uav_localization/
├── src/
│   ├── sensors/          # Sensor processing modules
│   │   ├── gps_processor.py
│   │   ├── imu_processor.py
│   │   └── stereo_vo.py
│   ├── fusion/           # EKF implementation
│   │   └── improved_ekf_fusion.py
│   └── utils/            # Visualization and utilities
│       └── visualization.py
├── tests/                # Unit tests
│   └── test_uav_localization.py
├── data/                 # Synthetic data generation
│   └── synthetic_data_generator.py
├── demo.py               # Main demonstration
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🧪 Testing

```bash
# Run all tests
python tests/test_uav_localization.py

# Check test coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Results

The system achieves:
- **Mean Position Error**: 19.4m (vs 47.0m baseline)
- **RMS Position Error**: 22.8m
- **Max Position Error**: 35.6m
- **Real-time capability**: 100Hz processing
- **Improvement factor**: 2.4x better than baseline
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
