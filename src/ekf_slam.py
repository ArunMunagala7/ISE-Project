"""
EKF-SLAM Implementation
Extended Kalman Filter for Simultaneous Localization and Mapping
"""
import numpy as np
from config.params import (
    MOTION_NOISE_V, MOTION_NOISE_W,
    MEASUREMENT_NOISE_RANGE, MEASUREMENT_NOISE_BEARING,
    INITIAL_LANDMARK_COV, MAX_RANGE, FOV_ANGLE
)
from src.utils import normalize_angle


class EKF_SLAM:
    """
    Extended Kalman Filter SLAM implementation
    
    State vector: [x, y, θ, lx_1, ly_1, lx_2, ly_2, ..., lx_n, ly_n]
    - Robot pose: (x, y, θ)
    - Landmarks: (lx_i, ly_i) for i = 1..n
    """
    
    def __init__(self, initial_state, num_landmarks, initial_cov):
        """
        Initialize EKF-SLAM
        
        Args:
            initial_state: Initial robot state [x, y, theta]
            num_landmarks: Number of landmarks in environment
            initial_cov: Initial covariance for robot state (3x3)
        """
        self.num_landmarks = num_landmarks
        
        # State dimension: 3 (robot) + 2*num_landmarks
        state_dim = 3 + 2 * num_landmarks
        
        # Initialize state vector
        self.mu = np.zeros(state_dim)
        self.mu[0:3] = initial_state
        
        # Initialize covariance matrix
        self.sigma = np.zeros((state_dim, state_dim))
        self.sigma[0:3, 0:3] = initial_cov
        
        # Initialize landmarks with moderate uncertainty
        for i in range(num_landmarks):
            idx = 3 + 2*i
            # Use smaller initial uncertainty to avoid numerical issues
            self.sigma[idx:idx+2, idx:idx+2] = np.eye(2) * INITIAL_LANDMARK_COV
        
        # Track which landmarks have been observed
        self.landmark_initialized = np.zeros(num_landmarks, dtype=bool)
        
        # Store history for analysis
        self.mu_history = [self.mu.copy()]
        self.sigma_history = [self.sigma.copy()]
    
    def predict(self, v, w, dt):
        """
        EKF Prediction Step
        
        Predict the next state based on motion model and update covariance
        
        Args:
            v: Linear velocity
            w: Angular velocity
            dt: Time step
        """
        # Current robot pose
        x, y, theta = self.mu[0:3]
        
        # --- Motion Model ---
        # Predict new robot pose
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = normalize_angle(theta + w * dt)
        
        # Update state estimate (robot moves, landmarks stay)
        self.mu[0] = x_new
        self.mu[1] = y_new
        self.mu[2] = theta_new
        
        # --- Jacobian of Motion Model ---
        # Jacobian with respect to robot state
        G_x = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0,  1]
        ])
        
        # Full Jacobian (state dimension)
        state_dim = len(self.mu)
        G = np.eye(state_dim)
        G[0:3, 0:3] = G_x
        
        # --- Process Noise ---
        # Motion noise covariance (control input noise)
        R = np.array([
            [(MOTION_NOISE_V * dt)**2, 0],
            [0, (MOTION_NOISE_W * dt)**2]
        ])
        
        # Jacobian with respect to control inputs
        V = np.array([
            [np.cos(theta) * dt, 0],
            [np.sin(theta) * dt, 0],
            [0, dt]
        ])
        
        # Process noise in state space
        Q = np.zeros((state_dim, state_dim))
        Q[0:3, 0:3] = V @ R @ V.T
        
        # --- Update Covariance ---
        self.sigma = G @ self.sigma @ G.T + Q
        
        # Ensure symmetry and prevent overflow
        self.sigma = (self.sigma + self.sigma.T) / 2
        self.sigma = np.clip(self.sigma, -1e10, 1e10)
        
        # Ensure symmetry and clip to prevent overflow
        self.sigma = (self.sigma + self.sigma.T) / 2
        self.sigma = np.clip(self.sigma, -1e10, 1e10)
    
    def measurement_model(self, robot_pose, landmark_pos):
        """
        Measurement model: Range-Bearing sensor
        
        Args:
            robot_pose: [x, y, theta]
            landmark_pos: [lx, ly]
        
        Returns:
            Expected measurement [range, bearing]
        """
        x, y, theta = robot_pose
        lx, ly = landmark_pos
        
        # Difference in position
        dx = lx - x
        dy = ly - y
        
        # Range (distance)
        q = dx**2 + dy**2
        range_pred = np.sqrt(q)
        
        # Bearing (angle relative to robot heading)
        bearing_pred = normalize_angle(np.arctan2(dy, dx) - theta)
        
        return np.array([range_pred, bearing_pred])
    
    def measurement_jacobian(self, robot_pose, landmark_pos):
        """
        Jacobian of measurement model
        
        H = ∂h/∂state where h is the measurement model
        
        Args:
            robot_pose: [x, y, theta]
            landmark_pos: [lx, ly]
        
        Returns:
            Jacobian matrix H (2 x state_dim)
        """
        x, y, theta = robot_pose
        lx, ly = landmark_pos
        
        dx = lx - x
        dy = ly - y
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)
        
        # Partial derivatives with respect to robot pose
        H_robot = np.array([
            [-dx/sqrt_q, -dy/sqrt_q, 0],
            [dy/q, -dx/q, -1]
        ])
        
        # Partial derivatives with respect to landmark position
        H_landmark = np.array([
            [dx/sqrt_q, dy/sqrt_q],
            [-dy/q, dx/q]
        ])
        
        return H_robot, H_landmark
    
    def update(self, landmark_id, measurement):
        """
        EKF Update Step
        
        Update state estimate based on landmark observation
        
        Args:
            landmark_id: Index of observed landmark (0 to num_landmarks-1)
            measurement: Observed [range, bearing]
        """
        # Check if this is the first observation of this landmark
        if not self.landmark_initialized[landmark_id]:
            self.initialize_landmark(landmark_id, measurement)
            return
        
        # Get current estimates
        robot_pose = self.mu[0:3]
        landmark_idx = 3 + 2 * landmark_id
        landmark_pos = self.mu[landmark_idx:landmark_idx+2]
        
        # --- Predicted Measurement ---
        z_pred = self.measurement_model(robot_pose, landmark_pos)
        
        # --- Innovation (measurement residual) ---
        innovation = measurement - z_pred
        innovation[1] = normalize_angle(innovation[1])  # Normalize bearing
        
        # --- Measurement Jacobian ---
        H_robot, H_landmark = self.measurement_jacobian(robot_pose, landmark_pos)
        
        # Full Jacobian (2 x state_dim)
        state_dim = len(self.mu)
        H = np.zeros((2, state_dim))
        H[:, 0:3] = H_robot
        H[:, landmark_idx:landmark_idx+2] = H_landmark
        
        # --- Measurement Noise ---
        Q_meas = np.array([
            [MEASUREMENT_NOISE_RANGE**2, 0],
            [0, MEASUREMENT_NOISE_BEARING**2]
        ])
        
        # --- Innovation Covariance ---
        S = H @ self.sigma @ H.T + Q_meas
        
        # Add small regularization for numerical stability
        S = S + np.eye(2) * 1e-6
        
        # --- Kalman Gain ---
        K = self.sigma @ H.T @ np.linalg.inv(S)
        
        # --- Update State ---
        self.mu = self.mu + K @ innovation
        self.mu[2] = normalize_angle(self.mu[2])  # Normalize robot heading
        
        # --- Update Covariance (Joseph form for numerical stability) ---
        I = np.eye(state_dim)
        I_KH = I - K @ H
        # Joseph form: P = (I-KH)P(I-KH)' + KQK'
        self.sigma = I_KH @ self.sigma @ I_KH.T + K @ Q_meas @ K.T
        
        # Ensure symmetry and numerical stability
        self.sigma = (self.sigma + self.sigma.T) / 2
        
        # Store history
        self.mu_history.append(self.mu.copy())
        self.sigma_history.append(self.sigma.copy())
    
    def initialize_landmark(self, landmark_id, measurement):
        """
        Initialize landmark position on first observation
        
        Args:
            landmark_id: Index of landmark
            measurement: First observation [range, bearing]
        """
        range_meas, bearing_meas = measurement
        x, y, theta = self.mu[0:3]
        
        # Convert range-bearing to Cartesian coordinates
        lx = x + range_meas * np.cos(bearing_meas + theta)
        ly = y + range_meas * np.sin(bearing_meas + theta)
        
        # Update landmark position in state
        landmark_idx = 3 + 2 * landmark_id
        self.mu[landmark_idx] = lx
        self.mu[landmark_idx + 1] = ly
        
        # Update covariance for this landmark
        # This should account for uncertainty in robot pose and measurement
        # Use reasonable initial uncertainty based on measurement noise
        from config.params import MEASUREMENT_NOISE_RANGE
        init_cov = (MEASUREMENT_NOISE_RANGE * 2) ** 2
        self.sigma[landmark_idx:landmark_idx+2, landmark_idx:landmark_idx+2] = np.eye(2) * init_cov
        
        # Mark as initialized
        self.landmark_initialized[landmark_id] = True
    
    def get_robot_state(self):
        """Get estimated robot state"""
        return self.mu[0:3].copy()
    
    def get_robot_covariance(self):
        """Get robot state covariance"""
        return self.sigma[0:3, 0:3].copy()
    
    def get_landmark_state(self, landmark_id):
        """Get estimated landmark position"""
        idx = 3 + 2 * landmark_id
        return self.mu[idx:idx+2].copy()
    
    def get_landmark_covariance(self, landmark_id):
        """Get landmark covariance"""
        idx = 3 + 2 * landmark_id
        return self.sigma[idx:idx+2, idx:idx+2].copy()
    
    def get_all_landmarks(self):
        """Get all landmark positions"""
        landmarks = []
        for i in range(self.num_landmarks):
            idx = 3 + 2*i
            landmarks.append(self.mu[idx:idx+2])
        return np.array(landmarks)
