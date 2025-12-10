"""
Data Association for SLAM
Determines which measurement corresponds to which landmark
"""
import numpy as np
from src.utils import compute_mahalanobis_distance, normalize_angle
from config.params import MAHALANOBIS_THRESHOLD, MAX_RANGE, FOV_ANGLE


class DataAssociation:
    """
    Handles data association using nearest neighbor with gating
    """
    
    def __init__(self, ekf_slam):
        """
        Initialize data association
        
        Args:
            ekf_slam: EKF_SLAM object for accessing state and covariance
        """
        self.ekf_slam = ekf_slam
    
    def get_visible_landmarks(self, robot_pose, true_landmarks):
        """
        Determine which landmarks are within sensor range and FOV
        
        Args:
            robot_pose: Current robot pose [x, y, theta]
            true_landmarks: True landmark positions (n x 2)
        
        Returns:
            List of landmark IDs that are visible
        """
        x, y, theta = robot_pose
        visible = []
        
        for i, landmark in enumerate(true_landmarks):
            lx, ly = landmark
            
            # Compute range
            dx = lx - x
            dy = ly - y
            range_val = np.sqrt(dx**2 + dy**2)
            
            # Check range
            if range_val > MAX_RANGE:
                continue
            
            # Compute bearing relative to robot
            bearing = normalize_angle(np.arctan2(dy, dx) - theta)
            
            # Check FOV
            if abs(bearing) > FOV_ANGLE / 2:
                continue
            
            visible.append(i)
        
        return visible
    
    def associate_measurements(self, measurements, visible_landmark_ids):
        """
        Associate measurements with landmarks using nearest neighbor
        
        Uses Mahalanobis distance with gating for robust association
        
        Args:
            measurements: List of measurements [range, bearing]
            visible_landmark_ids: List of potentially visible landmark IDs
        
        Returns:
            associations: List of (landmark_id, measurement) pairs
        """
        associations = []
        used_measurements = set()
        
        robot_pose = self.ekf_slam.get_robot_state()
        
        # For each visible landmark, find best matching measurement
        for landmark_id in visible_landmark_ids:
            # Skip if landmark not initialized
            if not self.ekf_slam.landmark_initialized[landmark_id]:
                # For uninitialized landmarks, use simple nearest neighbor
                best_idx = self._find_nearest_measurement(
                    measurements, landmark_id, used_measurements
                )
                if best_idx is not None:
                    associations.append((landmark_id, measurements[best_idx]))
                    used_measurements.add(best_idx)
                continue
            
            # Get predicted measurement for this landmark
            landmark_pos = self.ekf_slam.get_landmark_state(landmark_id)
            z_pred = self.ekf_slam.measurement_model(robot_pose, landmark_pos)
            
            # Get measurement Jacobian
            H_robot, H_landmark = self.ekf_slam.measurement_jacobian(robot_pose, landmark_pos)
            
            # Build full Jacobian
            state_dim = len(self.ekf_slam.mu)
            landmark_idx = 3 + 2 * landmark_id
            H = np.zeros((2, state_dim))
            H[:, 0:3] = H_robot
            H[:, landmark_idx:landmark_idx+2] = H_landmark
            
            # Innovation covariance
            from config.params import MEASUREMENT_NOISE_RANGE, MEASUREMENT_NOISE_BEARING
            Q = np.diag([MEASUREMENT_NOISE_RANGE**2, MEASUREMENT_NOISE_BEARING**2])
            S = H @ self.ekf_slam.sigma @ H.T + Q
            
            # Find best matching measurement
            best_match = None
            min_distance = float('inf')
            
            for idx, measurement in enumerate(measurements):
                if idx in used_measurements:
                    continue
                
                # Compute innovation
                innovation = measurement - z_pred
                innovation[1] = normalize_angle(innovation[1])
                
                # Compute Mahalanobis distance
                distance = compute_mahalanobis_distance(innovation, S)
                
                # Check gating threshold
                if distance < MAHALANOBIS_THRESHOLD and distance < min_distance:
                    min_distance = distance
                    best_match = idx
            
            # If found a match, add to associations
            if best_match is not None:
                associations.append((landmark_id, measurements[best_match]))
                used_measurements.add(best_match)
        
        return associations
    
    def _find_nearest_measurement(self, measurements, landmark_id, used_measurements):
        """
        Simple nearest neighbor for uninitialized landmarks
        
        Args:
            measurements: List of measurements
            landmark_id: Landmark ID
            used_measurements: Set of already used measurement indices
        
        Returns:
            Index of nearest measurement, or None
        """
        robot_pose = self.ekf_slam.get_robot_state()
        
        # Estimate landmark position from current robot pose and measurement
        # Since we don't have initialized position, just find unused measurement
        for idx, measurement in enumerate(measurements):
            if idx not in used_measurements:
                return idx
        
        return None
    
    def simulate_measurements(self, robot_pose, true_landmarks, add_noise=True):
        """
        Simulate sensor measurements from visible landmarks
        
        Args:
            robot_pose: True robot pose [x, y, theta]
            true_landmarks: True landmark positions (n x 2)
            add_noise: Whether to add measurement noise
        
        Returns:
            measurements: List of [range, bearing] measurements
            landmark_ids: Corresponding landmark IDs for each measurement
        """
        from config.params import MEASUREMENT_NOISE_RANGE, MEASUREMENT_NOISE_BEARING
        
        visible_ids = self.get_visible_landmarks(robot_pose, true_landmarks)
        
        measurements = []
        landmark_ids = []
        
        for landmark_id in visible_ids:
            landmark = true_landmarks[landmark_id]
            
            # Compute true measurement
            x, y, theta = robot_pose
            lx, ly = landmark
            
            dx = lx - x
            dy = ly - y
            
            range_true = np.sqrt(dx**2 + dy**2)
            bearing_true = normalize_angle(np.arctan2(dy, dx) - theta)
            
            # Add noise
            if add_noise:
                range_meas = range_true + np.random.normal(0, MEASUREMENT_NOISE_RANGE)
                bearing_meas = normalize_angle(
                    bearing_true + np.random.normal(0, MEASUREMENT_NOISE_BEARING)
                )
            else:
                range_meas = range_true
                bearing_meas = bearing_true
            
            measurements.append(np.array([range_meas, bearing_meas]))
            landmark_ids.append(landmark_id)
        
        return measurements, landmark_ids
