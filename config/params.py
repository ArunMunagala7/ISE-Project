"""
Simulation Parameters for SLAM
"""
import numpy as np

# Simulation parameters - LONGER for complex trajectory
DT = 0.1  # Time step (seconds)
SIM_TIME = 70.0  # Total simulation time (seconds) - increased from 50

# Robot parameters
ROBOT_RADIUS = 0.5  # Robot radius for visualization

# Motion noise (process noise) - INCREASED for more challenge
MOTION_NOISE_V = 0.2  # Linear velocity noise standard deviation (m/s)
MOTION_NOISE_W = 0.1  # Angular velocity noise standard deviation (rad/s)

# Measurement noise (observation noise) - INCREASED for more challenge
MEASUREMENT_NOISE_RANGE = 0.5  # Range measurement noise std (meters)
MEASUREMENT_NOISE_BEARING = 0.15  # Bearing measurement noise std (radians)

# Sensor parameters - REDUCED for more challenge
MAX_RANGE = 8.0  # Maximum sensor range (meters) - reduced from 10
FOV_ANGLE = 2*np.pi/3  # Field of view angle (radians) - 120 degrees (was 180)

# Trajectory parameters - MORE COMPLEX path
TRAJECTORY_TYPE = "figure8"  # Options: "circle", "figure8", "racetrack"
CIRCLE_RADIUS = 6.0  # Radius for circular trajectory
LINEAR_VELOCITY = 1.2  # Desired linear velocity (m/s) - slightly faster
FIGURE8_SCALE = 6.0  # Scale for figure-8 trajectory

# Landmark parameters - SCATTERED widely for challenge
NUM_LANDMARKS = 30  # Number of landmarks in the environment (more landmarks)
LANDMARK_AREA_SIZE = 20.0  # Larger area for better scattering

# Initial state
INITIAL_STATE = np.array([0.0, 0.0, 0.0])  # [x, y, theta]

# Initial covariance
INITIAL_STATE_COV = np.diag([0.1, 0.1, 0.05])  # Initial uncertainty in robot pose
INITIAL_LANDMARK_COV = 100.0  # Moderate uncertainty for uninitialized landmarks (was too large)

# Data association
MAHALANOBIS_THRESHOLD = 9.21  # Chi-square threshold for 95% confidence (2 DOF)

# Control parameters
CONTROL_TYPE = "qlearning"  # Options: "feedback" (original), "qlearning" (reinforcement learning)

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
