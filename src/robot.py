"""
Robot Motion Model - Unicycle Dynamics
"""
import numpy as np
from config.params import MOTION_NOISE_V, MOTION_NOISE_W
from src.utils import normalize_angle


class Robot:
    """
    Robot with unicycle motion model
    """
    
    def __init__(self, initial_state, dt):
        """
        Initialize robot
        
        Args:
            initial_state: [x, y, theta] initial pose
            dt: Time step
        """
        self.state = np.array(initial_state, dtype=float)
        self.dt = dt
        self.true_trajectory = [self.state.copy()]
        
    def motion_model(self, state, v, w, dt):
        """
        Unicycle motion model (deterministic)
        
        x_{k+1} = x_k + v*cos(θ)*dt
        y_{k+1} = y_k + v*sin(θ)*dt
        θ_{k+1} = θ_k + w*dt
        
        Args:
            state: Current state [x, y, theta]
            v: Linear velocity
            w: Angular velocity
            dt: Time step
        
        Returns:
            Next state [x, y, theta]
        """
        x, y, theta = state
        
        # Update position and orientation
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = normalize_angle(theta + w * dt)
        
        return np.array([x_new, y_new, theta_new])
    
    def motion_model_jacobian(self, state, v, dt):
        """
        Jacobian of motion model with respect to state
        
        This is used for linearizing the nonlinear motion model
        in the EKF prediction step.
        
        Args:
            state: Current state [x, y, theta]
            v: Linear velocity
            dt: Time step
        
        Returns:
            Jacobian matrix (3x3)
        """
        theta = state[2]
        
        # Jacobian G_t
        G = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0,  1]
        ])
        
        return G
    
    def move(self, v, w, add_noise=True):
        """
        Execute motion command with noise
        
        Args:
            v: Linear velocity command
            w: Angular velocity command
            add_noise: Whether to add motion noise
        
        Returns:
            Actual control executed (with noise)
        """
        # Add noise to control inputs
        if add_noise:
            v_noisy = v + np.random.normal(0, MOTION_NOISE_V)
            w_noisy = w + np.random.normal(0, MOTION_NOISE_W)
        else:
            v_noisy = v
            w_noisy = w
        
        # Apply motion model
        self.state = self.motion_model(self.state, v_noisy, w_noisy, self.dt)
        self.true_trajectory.append(self.state.copy())
        
        return v_noisy, w_noisy
    
    def get_state(self):
        """Get current robot state"""
        return self.state.copy()
    
    def get_trajectory(self):
        """Get full trajectory history"""
        return np.array(self.true_trajectory)


class TrajectoryController:
    """
    Hybrid controller supporting both feedback and Q-Learning control
    """
    
    def __init__(self, trajectory_type, params, control_type="feedback", qlearning_controller=None):
        """
        Initialize controller
        
        Args:
            trajectory_type: Type of trajectory ("circle", "figure8")
            params: Dictionary of trajectory parameters
            control_type: "feedback" (original PID-like) or "qlearning" (RL-based)
            qlearning_controller: QLearningController instance (if using Q-Learning)
        """
        self.trajectory_type = trajectory_type
        self.params = params
        self.control_type = control_type
        self.qlearning_controller = qlearning_controller
        
        # For Q-Learning: precompute trajectory waypoints
        if control_type == "qlearning" and qlearning_controller is not None:
            self.trajectory_waypoints = self._generate_trajectory_waypoints()
            self.prev_state_action = None  # For Q-Learning updates (state, action_idx)
            self.prev_errors = None  # For reward calculation
        
    def _generate_trajectory_waypoints(self, num_points=500):
        """Generate waypoints along the trajectory for Q-Learning reference"""
        import numpy as np
        waypoints = []
        
        if self.trajectory_type == "circle":
            radius = self.params.get('radius', 5.0)
            for i in range(num_points):
                theta = 2 * np.pi * i / num_points
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                waypoints.append([x, y])
                
        elif self.trajectory_type == "figure8":
            # Generate waypoints for complex racetrack
            scale = self.params.get('scale', 6.0)
            omega = 0.08
            duration = 2 * np.pi / omega
            
            for i in range(num_points):
                t = duration * i / num_points
                phase = omega * t
                section = (phase % (2 * np.pi))
                
                if section < np.pi / 2:  # Sharp right turn
                    angle = section * 4
                    radius = scale * 1.2
                    x = radius * np.cos(angle) + scale * 0.5
                    y = radius * np.sin(angle) - scale * 0.3
                    
                elif section < np.pi:  # Straightaway
                    progress = (section - np.pi/2) / (np.pi/2)
                    x = scale * (1.7 - 3.0 * progress)
                    y = scale * (0.9 + 0.4 * np.sin(progress * np.pi))
                    
                elif section < 3 * np.pi / 2:  # Hairpin turn
                    angle = (section - np.pi) * 3 + np.pi
                    radius = scale * 0.6
                    x = radius * np.cos(angle) - scale * 1.2
                    y = radius * np.sin(angle) + scale * 0.8
                    
                else:  # S-curve back
                    progress = (section - 3*np.pi/2) / (np.pi/2)
                    x = scale * (-1.8 + 2.3 * progress)
                    y = scale * (0.8 - 1.6 * progress + 0.5 * np.sin(progress * 2 * np.pi))
                
                # Add wobble
                wobble_x = scale * 0.15 * np.sin(phase * 5)
                wobble_y = scale * 0.1 * np.cos(phase * 7)
                x += wobble_x
                y += wobble_y
                
                waypoints.append([x, y])
        
        return np.array(waypoints)
        
    def get_control(self, current_state, t):
        """
        Compute control inputs to follow desired trajectory
        
        Args:
            current_state: Current robot state [x, y, theta]
            t: Current time
        
        Returns:
            (v, w): Linear and angular velocities
        """
        # Use Q-Learning control if enabled
        if self.control_type == "qlearning" and self.qlearning_controller is not None:
            return self._get_qlearning_control(current_state, t)
        
        # Otherwise use original feedback control
        return self._get_feedback_control(current_state, t)
    
    def _get_qlearning_control(self, current_state, t):
        """
        Q-Learning based control (learns while running)
        Combines base feedback control with learned angular velocity offset
        """
        from src.qlearning_controller import compute_tracking_errors
        
        # Get base feedback control first
        v_base, w_base = self._get_feedback_control(current_state, t)
        
        # Find nearest target point on trajectory
        distances = np.sqrt((self.trajectory_waypoints[:, 0] - current_state[0])**2 + 
                          (self.trajectory_waypoints[:, 1] - current_state[1])**2)
        nearest_idx = np.argmin(distances)
        look_ahead = min(nearest_idx + 10, len(self.trajectory_waypoints) - 1)
        target_point = self.trajectory_waypoints[look_ahead]
        
        # Compute tracking errors for Q-Learning
        lateral_error, heading_error = compute_tracking_errors(
            current_state, target_point, self.trajectory_waypoints
        )
        
        # Get Q-Learning action (angular velocity offset)
        omega_offset, state, action_idx = self.qlearning_controller.get_control_offset(
            lateral_error, heading_error, training=True
        )
        
        # Update Q-table if we have previous state-action pair
        if self.prev_state_action is not None:
            prev_state, prev_action = self.prev_state_action
            prev_lat_err, prev_head_err = self.prev_errors
            
            # Compute reward based on improvement
            reward = self.qlearning_controller.compute_reward(lateral_error, heading_error)
            
            # Update Q-table
            self.qlearning_controller.update_q_table(prev_state, prev_action, reward, state)
        
        # Save current state-action for next update
        self.prev_state_action = (state, action_idx)
        self.prev_errors = (lateral_error, heading_error)
        
        # Apply learned offset to base control
        w = w_base + omega_offset
        
        # Clip to safe limits
        w = np.clip(w, -2.0, 2.0)
        
        return v_base, w
    
    def _get_feedback_control(self, current_state, t):
        """
        Original feedback control method (kept as fallback)
        
        Args:
            current_state: Current robot state [x, y, theta]
            t: Current time
        
        Returns:
            (v, w): Linear and angular velocities
        """
        if self.trajectory_type == "circle":
            # Untitled7.ipynb circular trajectory control
            # Keep robot on a circle of given radius
            radius = self.params.get('radius', 5.0)
            v_desired = self.params.get('velocity', 1.0)
            
            # Control gains from Untitled7
            k_p = 2.0  # Proportional gain for heading error
            k_r = 0.5  # Cross-track error gain
            
            # Current estimated position
            est_x, est_y = current_state[0], current_state[1]
            est_d = np.sqrt(est_x**2 + est_y**2)
            
            # Cross-track error (distance from circle)
            cross_error = est_d - radius
            
            # Tangent direction (perpendicular to radial)
            tangent_x = -est_y
            tangent_y = est_x
            norm = np.sqrt(tangent_x**2 + tangent_y**2)
            
            if norm == 0:
                desired_theta = current_state[2]
            else:
                desired_theta = np.arctan2(tangent_y, tangent_x)
            
            # Heading error
            theta_error = normalize_angle(desired_theta - current_state[2])
            
            # Nominal angular velocity for circular motion
            omega_nom = v_desired / radius
            
            # Control law from Untitled7
            omega = omega_nom + k_p * theta_error + k_r * cross_error
            v = v_desired
            
            return v, omega
            
        elif self.trajectory_type == "figure8":
            # Figure-8 trajectory (Lemniscate curve) for area coverage
            scale = self.params.get('scale', 6.0)
            v_desired = self.params.get('velocity', 1.0)
            
            # Parametric figure-8: x = a*sin(t), y = a*sin(t)*cos(t)
            omega = 0.15  # Angular frequency for good figure-8 shape
            
            # Desired position on figure-8
            desired_x = scale * np.sin(omega * t)
            desired_y = scale * np.sin(omega * t) * np.cos(omega * t)
            
            # Velocity on the curve (derivatives)
            desired_vx = scale * omega * np.cos(omega * t)
            desired_vy = scale * omega * (np.cos(2 * omega * t))
            
            # Desired heading (tangent to curve)
            desired_theta = np.arctan2(desired_vy, desired_vx)
            
            # Current errors
            error_x = desired_x - current_state[0]
            error_y = desired_y - current_state[1]
            error_theta = normalize_angle(desired_theta - current_state[2])
            
            # Proportional feedback control
            kp_v = 2.0  # Position gain
            kp_w = 3.0  # Heading gain
            
            # Compute controls
            error_dist = np.sqrt(error_x**2 + error_y**2)
            v = v_desired + kp_v * error_dist
            w = kp_w * error_theta
            
            # Limit controls
            v = np.clip(v, 0, 3.0)
            w = np.clip(w, -2.0, 2.0)
            
            return v, w
        
        else:
            # Default: constant velocity
            return self.params.get('velocity', 1.0), 0.0
