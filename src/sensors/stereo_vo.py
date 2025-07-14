"""
Stereo Visual Odometry Module for UAV Localization System

This module handles stereo vision processing for visual odometry,
including feature detection, matching, and pose estimation.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class StereoCalibration:
    """Stereo camera calibration parameters."""
    K_left: np.ndarray  # Left camera intrinsic matrix
    K_right: np.ndarray  # Right camera intrinsic matrix
    D_left: np.ndarray  # Left camera distortion coefficients
    D_right: np.ndarray  # Right camera distortion coefficients
    R: np.ndarray  # Rotation matrix between cameras
    T: np.ndarray  # Translation vector between cameras
    baseline: float  # Distance between cameras

class StereoVisualOdometry:
    """
    Stereo visual odometry processor for pose estimation from stereo images.
    
    This class processes stereo image pairs to estimate camera motion
    and provide pose estimates for the UAV localization system.
    """
    
    def __init__(self, calibration: StereoCalibration):
        """
        Initialize stereo visual odometry processor.
        
        Args:
            calibration: Stereo camera calibration parameters
        """
        self.calibration = calibration
        
        # Feature detection and matching parameters
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Pose estimation parameters
        self.min_features = 50
        self.ransac_threshold = 1.0
        self.ransac_confidence = 0.99
        
        # Previous frame data for motion estimation
        self.prev_keypoints_left = None
        self.prev_descriptors_left = None
        self.prev_pose = np.eye(4)  # Previous pose as 4x4 transformation matrix
        
        # Stereo matching parameters
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    
    def compute_stereo_vo_pose(self, left_image: np.ndarray, right_image: np.ndarray) -> dict:
        """
        Function: compute_stereo_vo_pose
        Input: Left and right stereo images
        Output: Estimated pose as translation (x,y,z) and rotation (quaternion or Euler angles)
        
        Compute pose estimate from stereo image pair.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            
        Returns:
            Dictionary containing pose estimation results
        """
        # Convert images to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            
        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image
        
        # Detect features in left image
        keypoints_left, descriptors_left = self.feature_detector.detectAndCompute(left_gray, None)
        
        # Initialize result dictionary
        result = {
            'pose': np.eye(4),
            'translation': np.zeros(3),
            'rotation_matrix': np.eye(3),
            'euler_angles': np.zeros(3),
            'num_features': len(keypoints_left),
            'success': False
        }
        
        # Check if enough features detected
        if len(keypoints_left) < self.min_features:
            return result
        
        # Compute 3D points from stereo correspondence
        points_3d = self._compute_3d_points(left_gray, right_gray, keypoints_left)
        
        # Perform temporal matching if previous frame exists
        if self.prev_keypoints_left is not None and self.prev_descriptors_left is not None:
            # Match features between consecutive frames
            matches = self._match_features(descriptors_left, self.prev_descriptors_left)
            
            if len(matches) >= self.min_features:
                # Estimate pose change using PnP
                pose_change = self._estimate_pose_change(matches, points_3d, keypoints_left)
                
                if pose_change is not None:
                    # Update pose
                    self.prev_pose = self.prev_pose @ pose_change
                    
                    # Extract translation and rotation
                    result['pose'] = self.prev_pose.copy()
                    result['translation'] = self.prev_pose[:3, 3]
                    result['rotation_matrix'] = self.prev_pose[:3, :3]
                    result['euler_angles'] = self._rotation_matrix_to_euler(result['rotation_matrix'])
                    result['success'] = True
        
        # Store current frame data for next iteration
        self.prev_keypoints_left = keypoints_left
        self.prev_descriptors_left = descriptors_left
        
        return result
    
    def _compute_3d_points(self, left_image: np.ndarray, right_image: np.ndarray, 
                          keypoints_left: List) -> np.ndarray:
        """
        Compute 3D points from stereo correspondence.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            keypoints_left: Keypoints detected in left image
            
        Returns:
            Array of 3D points in camera coordinate system
        """
        # Compute disparity map
        disparity = self.stereo_matcher.compute(left_image, right_image)
        
        # Convert disparity to depth
        Q = self._get_disparity_to_depth_matrix()
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        # Extract 3D coordinates for detected keypoints
        keypoint_3d = []
        for kp in keypoints_left:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < points_3d.shape[1] and 0 <= y < points_3d.shape[0]:
                point_3d = points_3d[y, x]
                if point_3d[2] > 0:  # Valid depth
                    keypoint_3d.append(point_3d)
        
        return np.array(keypoint_3d)
    
    def _get_disparity_to_depth_matrix(self) -> np.ndarray:
        """
        Get disparity-to-depth reprojection matrix Q.
        
        Returns:
            4x4 reprojection matrix Q
        """
        # Simplified Q matrix for stereo rectified images
        fx = self.calibration.K_left[0, 0]  # Focal length
        baseline = self.calibration.baseline
        cx = self.calibration.K_left[0, 2]  # Principal point x
        cy = self.calibration.K_left[1, 2]  # Principal point y
        
        Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, fx],
            [0, 0, -1/baseline, 0]
        ])
        
        return Q
    
    def _match_features(self, descriptors_current: np.ndarray, 
                       descriptors_previous: np.ndarray) -> List:
        """
        Match features between current and previous frames.
        
        Args:
            descriptors_current: Current frame descriptors
            descriptors_previous: Previous frame descriptors
            
        Returns:
            List of feature matches
        """
        # Perform feature matching
        matches = self.matcher.match(descriptors_current, descriptors_previous)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep only good matches (top 70%)
        good_matches = matches[:int(len(matches) * 0.7)]
        
        return good_matches
    
    def _estimate_pose_change(self, matches: List, points_3d: np.ndarray, 
                            keypoints_current: List) -> Optional[np.ndarray]:
        """
        Estimate pose change using PnP algorithm.
        
        Args:
            matches: Feature matches between frames
            points_3d: 3D points from previous frame
            keypoints_current: Current frame keypoints
            
        Returns:
            4x4 transformation matrix or None if estimation fails
        """
        if len(matches) < 6:  # Minimum points for PnP
            return None
        
        # Prepare 3D-2D correspondences
        object_points = []
        image_points = []
        
        for match in matches:
            if match.trainIdx < len(points_3d):
                object_points.append(points_3d[match.trainIdx])
                image_points.append(keypoints_current[match.queryIdx].pt)
        
        if len(object_points) < 6:
            return None
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Solve PnP problem
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, self.calibration.K_left, 
            self.calibration.D_left, confidence=self.ransac_confidence,
            reprojectionError=self.ransac_threshold
        )
        
        if not success or len(inliers) < 6:
            return None
        
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rmat
        transform[:3, 3] = tvec.flatten()
        
        return transform
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Euler angles as [roll, pitch, yaw] in radians
        """
        # Extract Euler angles from rotation matrix
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return np.array([roll, pitch, yaw])
    
    def get_stereo_vo_covariance(self, num_features: int) -> np.ndarray:
        """
        Get visual odometry measurement covariance based on number of features.
        
        Args:
            num_features: Number of features used in pose estimation
            
        Returns:
            6x6 covariance matrix for [x, y, z, roll, pitch, yaw]
        """
        # Base covariance values
        base_pos_std = 0.1  # meters
        base_rot_std = 0.05  # radians
        
        # Scale covariance based on number of features
        scale_factor = max(1.0, 100.0 / num_features)
        
        pos_var = (base_pos_std * scale_factor)**2
        rot_var = (base_rot_std * scale_factor)**2
        
        covariance = np.diag([pos_var, pos_var, pos_var, rot_var, rot_var, rot_var])
        
        return covariance
    
    def reset_tracking(self) -> None:
        """
        Reset visual odometry tracking state.
        """
        self.prev_keypoints_left = None
        self.prev_descriptors_left = None
        self.prev_pose = np.eye(4)
