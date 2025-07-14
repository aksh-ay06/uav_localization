"""
GPS Processor Module for UAV Localization System

This module handles GPS data processing and coordinate transformations
for the UAV localization system.
"""

import numpy as np
import pyproj
from typing import Tuple, Optional

class GPSProcessor:
    """
    GPS data processor for converting GPS coordinates to local Cartesian coordinates.
    
    This class handles the conversion of GPS latitude, longitude, and altitude
    to local Cartesian coordinates relative to a reference point.
    """
    
    def __init__(self, reference_lat: float, reference_lon: float, reference_alt: float = 0.0):
        """
        Initialize GPS processor with reference coordinates.
        
        Args:
            reference_lat: Reference latitude in degrees
            reference_lon: Reference longitude in degrees
            reference_alt: Reference altitude in meters (default: 0.0)
        """
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        self.reference_alt = reference_alt
        
        # Initialize coordinate transformation
        self.transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",  # WGS84 (GPS coordinates)
            "EPSG:3857",  # Web Mercator (meters)
            always_xy=True
        )
        
        # Get reference point in Cartesian coordinates
        self.ref_x, self.ref_y = self.transformer.transform(reference_lon, reference_lat)
    
    def convert_gps_to_cartesian(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """
        Function: convert_gps_to_cartesian
        Input: Latitude, longitude, altitude
        Output: Local Cartesian (X,Y,Z) coordinates relative to a reference point
        
        Convert GPS coordinates to local Cartesian coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            
        Returns:
            Tuple of (x, y, z) coordinates in meters relative to reference point
        """
        # Transform GPS coordinates to Cartesian
        x_world, y_world = self.transformer.transform(lon, lat)
        
        # Calculate relative position from reference point
        x_local = x_world - self.ref_x
        y_local = y_world - self.ref_y
        z_local = alt - self.reference_alt
        
        return x_local, y_local, z_local
    
    def get_gps_accuracy_estimate(self, hdop: float, vdop: float) -> Tuple[float, float]:
        """
        Estimate GPS measurement accuracy based on dilution of precision values.
        
        Args:
            hdop: Horizontal dilution of precision
            vdop: Vertical dilution of precision
            
        Returns:
            Tuple of (horizontal_accuracy, vertical_accuracy) in meters
        """
        # Typical GPS accuracy is around 3-5 meters
        base_accuracy = 3.0
        
        horizontal_accuracy = base_accuracy * hdop
        vertical_accuracy = base_accuracy * vdop
        
        return horizontal_accuracy, vertical_accuracy
    
    def validate_gps_data(self, lat: float, lon: float, alt: float) -> bool:
        """
        Validate GPS data for reasonable values.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            
        Returns:
            True if GPS data is valid, False otherwise
        """
        # Check latitude bounds
        if lat < -90.0 or lat > 90.0:
            return False
            
        # Check longitude bounds
        if lon < -180.0 or lon > 180.0:
            return False
            
        # Check altitude bounds (reasonable for UAV operations)
        if alt < -500.0 or alt > 10000.0:
            return False
            
        return True
    
    def apply_gps_noise_model(self, measurement: Tuple[float, float, float], 
                            noise_std: Tuple[float, float, float] = (2.0, 2.0, 3.0)) -> np.ndarray:
        """
        Apply noise model to GPS measurements for simulation purposes.
        
        Args:
            measurement: GPS measurement as (x, y, z) tuple
            noise_std: Standard deviation of noise for (x, y, z) components
            
        Returns:
            Noisy GPS measurement as numpy array
        """
        x, y, z = measurement
        noise_x, noise_y, noise_z = noise_std
        
        # Add Gaussian noise to each component
        noisy_x = x + np.random.normal(0, noise_x)
        noisy_y = y + np.random.normal(0, noise_y)
        noisy_z = z + np.random.normal(0, noise_z)
        
        return np.array([noisy_x, noisy_y, noisy_z])
