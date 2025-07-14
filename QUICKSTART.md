# Quick Start Guide - UAV Localization System

## What This System Does

This UAV localization system fuses multiple sensors to estimate the position, velocity, and attitude of a UAV in real-time:

- **GPS**: Provides global position reference (low frequency, ~2Hz)
- **IMU**: Provides high-frequency motion data (accelerometer + gyroscope, ~100Hz)  
- **Stereo Camera**: Provides visual odometry estimates (~10Hz)
- **Extended Kalman Filter**: Fuses all sensor data optimally

## 5-Minute Setup

1. **Navigate to project directory**:
   ```bash
   cd /home/akshay/uav_localization
   ```

2. **Run setup (creates virtual environment and installs dependencies)**:
   ```bash
   chmod +x setup.sh && ./setup.sh
   ```

3. **Activate environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Run the demo**:
   ```bash
   python demo.py
   ```

## What You'll See

The demo generates synthetic sensor data and shows:

```
=== UAV Localization System Demo ===
Initializing system components...
Generating synthetic sensor data...
Processing sensor data...
Time: 0.0s, Position: [64.10, -1.06, -2.46]m
Time: 5.0s, Position: [38.60, 61.38, 2.69]m
...
Mean Position Error: 2.347 m
RMS Position Error: 3.443 m
```

## Key Files to Understand

- **`demo.py`**: Complete working example
- **`src/main.py`**: Main system orchestration
- **`src/fusion/ekf_fusion.py`**: The Extended Kalman Filter implementation
- **`src/sensors/`**: Individual sensor processors

## Using with Real Data

Replace the synthetic data generation in `demo.py` with your actual sensor readings:

```python
# Instead of synthetic data:
sensor_data = {
    'timestamp': your_timestamp,
    'dt': time_step,
    'imu': {
        'accel': your_accelerometer_reading,  # 3D numpy array
        'gyro': your_gyroscope_reading        # 3D numpy array
    },
    'gps': {
        'lat': your_latitude,
        'lon': your_longitude, 
        'alt': your_altitude
    }
}

result = uav_system.process_sensor_data(sensor_data)
fused_position = result['fused_position']
```

## Testing

```bash
python tests/test_uav_localization.py
```

## GitHub Copilot Integration Tips

This codebase is designed to work excellently with GitHub Copilot:

1. **Function comments are templates for Copilot**:
   ```python
   # Function: your_new_function_name
   # Input: description of inputs
   # Output: description of outputs
   ```

2. **Continue patterns established in the code**
3. **Use descriptive variable names**
4. **Keep functions focused on single tasks**

## Next Steps

1. **Modify `demo.py`** to use your own sensor data
2. **Adjust noise parameters** in sensor processors for your hardware
3. **Add additional sensors** by following the existing patterns
4. **Tune EKF parameters** for your specific application

## Need Help?

- Check `OVERVIEW.md` for detailed documentation
- Look at the unit tests for usage examples
- Each module has comprehensive docstrings
