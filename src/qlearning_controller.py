"""
Q-Learning Reinforcement Learning Controller for SLAM
Learns to follow trajectory while robot simultaneously performs SLAM
"""
import numpy as np
from src.utils import normalize_angle

class QLearningController:
    """
    Q-Learning agent that learns to control robot by trial and error
    
    State: (lateral_error, heading_error) discretized into bins
    Actions: Angular velocity offsets to add to base control
    Reward: -(|lateral_error| + 0.5*|heading_error|)
    """
    
    def __init__(self, num_bins=10, num_actions=5, alpha=0.1, gamma=0.9, epsilon=0.2):
        """
        Initialize Q-Learning controller
        
        Args:
            num_bins: Number of discrete bins for state space
            num_actions: Number of discrete actions (angular velocity offsets)
            alpha: Learning rate (how quickly to update Q-values)
            gamma: Discount factor (how much to value future rewards)
            epsilon: Exploration rate (probability of random action)
        """
        self.num_bins = num_bins
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Define discrete action space (angular velocity offsets)
        self.actions = np.linspace(-0.3, 0.3, num_actions)  # rad/s offsets
        
        # Q-table: [lateral_error_bin, heading_error_bin, action] -> expected reward
        self.q_table = np.zeros((num_bins, num_bins, num_actions))
        
        # State discretization bins
        self.lat_bins = np.linspace(-3, 3, num_bins)  # lateral error range
        self.head_bins = np.linspace(-np.pi/2, np.pi/2, num_bins)  # heading error range
        
        # Learning statistics
        self.episode_rewards = []
        self.total_steps = 0
        
    def discretize_state(self, lateral_error, heading_error):
        """
        Convert continuous state to discrete bin indices
        
        Args:
            lateral_error: Distance from desired path
            heading_error: Angle difference from desired heading
            
        Returns:
            (lat_idx, head_idx): Discrete state indices
        """
        lat_idx = np.digitize(lateral_error, self.lat_bins) - 1
        head_idx = np.digitize(heading_error, self.head_bins) - 1
        
        # Clip to valid range
        lat_idx = np.clip(lat_idx, 0, self.num_bins - 1)
        head_idx = np.clip(head_idx, 0, self.num_bins - 1)
        
        return lat_idx, head_idx
    
    def choose_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Tuple (lat_idx, head_idx)
            training: If True, use epsilon-greedy. If False, always greedy.
            
        Returns:
            action_idx: Index of selected action
        """
        if training and np.random.rand() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.num_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule
        
        Q(s,a) <- Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state (lat_idx, head_idx)
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        self.total_steps += 1
    
    def compute_reward(self, lateral_error, heading_error):
        """
        Compute reward based on tracking errors
        
        Args:
            lateral_error: Distance from path
            heading_error: Angle from desired heading
            
        Returns:
            reward: Scalar reward value (higher is better)
        """
        # Standard reward: negative sum of weighted errors
        reward = -(abs(lateral_error) + 0.5 * abs(heading_error))
        
        return reward
    
    def get_control_offset(self, lateral_error, heading_error, training=True):
        """
        Get angular velocity offset from Q-Learning policy
        
        Args:
            lateral_error: Current lateral tracking error
            heading_error: Current heading tracking error
            training: Whether to update Q-table
            
        Returns:
            omega_offset: Angular velocity offset to add to base control
            state: Current discrete state (for later update)
        """
        # Discretize state
        state = self.discretize_state(lateral_error, heading_error)
        
        # Choose action
        action_idx = self.choose_action(state, training=training)
        
        # Get angular velocity offset
        omega_offset = self.actions[action_idx]
        
        return omega_offset, state, action_idx
    
    def save_model(self, filepath):
        """Save Q-table to file"""
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")
    
    def load_model(self, filepath):
        """Load Q-table from file"""
        self.q_table = np.load(filepath)
        print(f"Q-table loaded from {filepath}")
    
    def get_statistics(self):
        """Get learning statistics"""
        return {
            'total_steps': self.total_steps,
            'q_table_coverage': np.count_nonzero(self.q_table) / self.q_table.size,
            'avg_q_value': np.mean(self.q_table[self.q_table != 0]) if np.any(self.q_table) else 0
        }


def compute_tracking_errors(robot_state, target_point, trajectory_points):
    """
    Compute lateral and heading errors for Q-Learning
    
    Args:
        robot_state: [x, y, theta]
        target_point: [target_x, target_y] on trajectory
        trajectory_points: Array of trajectory waypoints for finding nearest point
        
    Returns:
        lateral_error: Signed distance from trajectory
        heading_error: Angle difference from desired heading
    """
    x, y, theta = robot_state[0], robot_state[1], robot_state[2]
    target_x, target_y = target_point
    
    # Heading error: difference between desired and actual heading
    desired_heading = np.arctan2(target_y - y, target_x - x)
    heading_error = normalize_angle(desired_heading - theta)
    
    # Lateral error: perpendicular distance to trajectory
    # Find nearest point on trajectory
    distances = np.sqrt((trajectory_points[:, 0] - x)**2 + 
                       (trajectory_points[:, 1] - y)**2)
    nearest_idx = np.argmin(distances)
    
    # Compute signed lateral error using cross product
    if nearest_idx < len(trajectory_points) - 1:
        p1 = trajectory_points[nearest_idx]
        p2 = trajectory_points[nearest_idx + 1]
        
        # Vector from p1 to p2 (trajectory direction)
        trajectory_vec = p2 - p1
        # Vector from p1 to robot
        robot_vec = np.array([x - p1[0], y - p1[1]])
        
        # Cross product gives signed distance
        cross = trajectory_vec[0] * robot_vec[1] - trajectory_vec[1] * robot_vec[0]
        lateral_error = cross / np.linalg.norm(trajectory_vec)
    else:
        lateral_error = distances[nearest_idx]
    
    return lateral_error, heading_error
