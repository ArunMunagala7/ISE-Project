"""
Simulation Parameters for SLAM
"""
import numpy as np

# Simulation parameters - Coverage-based exploration
DT = 0.1  # Time step (seconds)
SIM_TIME = 120.0  # Extended time for complete coverage

# Robot parameters
ROBOT_RADIUS = 0.5  # Robot radius for visualization

# Motion noise (process noise) - Balanced
MOTION_NOISE_V = 0.1  # Linear velocity noise
MOTION_NOISE_W = 0.05  # Angular velocity noise

# Measurement noise (observation noise) - Realistic
MEASUREMENT_NOISE_RANGE = 0.3  # Range measurement noise std (meters)
MEASUREMENT_NOISE_BEARING = 0.1  # Bearing measurement noise std (radians)

# Sensor parameters - Good coverage
MAX_RANGE = 8.0  # Maximum sensor range (meters) - increased for outer landmarks
FOV_ANGLE = 2*np.pi  # Field of view angle - full 360° for better coverage

# Trajectory parameters - Exploration pattern
TRAJECTORY_TYPE = "figure8"  # Use figure-8 for area coverage
CIRCLE_RADIUS = 5.0  # Radius for circular trajectory
LINEAR_VELOCITY = 1.0  # Desired linear velocity (m/s)
FIGURE8_SCALE = 8.0  # Scale for figure-8 trajectory (increased to reach all landmarks)

# Landmark parameters - Strategic distribution for coverage
NUM_LANDMARKS = 20  # Balanced number for exploration
LANDMARK_AREA_SIZE = 15.0  # Area that figure-8 can reach

# Initial state - Start at origin
INITIAL_STATE = np.array([0.0, 0.0, 0.0])  # [x, y, theta]

# Initial covariance - Moderate uncertainty
INITIAL_STATE_COV = np.diag([0.1, 0.1, 0.05])**2
INITIAL_LANDMARK_COV = 25.0  # Default landmark uncertainty (reduced for numerical stability)

# Data association
MAHALANOBIS_THRESHOLD = 9.21  # Chi-square threshold for 95% confidence (2 DOF)

# Control parameters
CONTROL_TYPE = "feedback"  # Use feedback control for exploration

# Q-Learning parameters (only used when CONTROL_TYPE = "qlearning")
QL_NUM_BINS = 10  # Number of state discretization bins
QL_NUM_ACTIONS = 5  # Number of discrete actions (angular velocity offsets)
QL_ALPHA = 0.1  # Learning rate (how fast to learn)
QL_GAMMA = 0.9  # Discount factor (future reward importance)
QL_EPSILON = 0.2  # Exploration rate (probability of random action)
QL_TRAINING = True  # Whether to train during simulation (True) or use pre-trained model (False)
QL_MODEL_PATH = "outputs/qlearning_model.npy"  # Path to save/load Q-table

# Visualization
PLOT_INTERVAL = 10  # Update plot every N steps (balance between speed and smoothness)
SAVE_VIDEO = True
VIDEO_FPS = 10
ANIMATION_ENABLED = True  # Enable real-time animation
SHOW_ANIMATION = True  # Display live updates
