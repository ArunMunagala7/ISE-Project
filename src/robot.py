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
            # Circular trajectory
            radius = self.params.get('radius', 5.0)
            v_desired = self.params.get('velocity', 1.0)
            
            # For a circle: v = r*omega, so omega = v/r
            w = v_desired / radius
            v = v_desired
            
            # Simple feedback: adjust based on current position
            desired_x = radius * np.cos(t * w)
            desired_y = radius * np.sin(t * w)
            
            error_x = desired_x - current_state[0]
            error_y = desired_y - current_state[1]
            
            # Proportional feedback gains
            kp_v = 0.5
            kp_w = 1.0
            
            # Adjust control based on error
            error_dist = np.sqrt(error_x**2 + error_y**2)
            v += kp_v * error_dist
            
            desired_theta = np.arctan2(desired_y, desired_x) + np.pi/2
            error_theta = normalize_angle(desired_theta - current_state[2])
            w += kp_w * error_theta
            
            return v, w
            
        elif self.trajectory_type == "figure8":
            # Complex racetrack with multiple distinct sections
            scale = self.params.get('scale', 6.0)
            v_desired = self.params.get('velocity', 1.2)
            
            # Use time to create distinct track sections
            omega = 0.08  # Slower for more variation
            phase = omega * t
            
            # Section 1: Large outer loop (0 to 2π)
            section = (phase % (2 * np.pi))
            
            if section < np.pi / 2:  # First quarter - sharp right turn
                angle = section * 4
                radius = scale * 1.2
                desired_x = radius * np.cos(angle) + scale * 0.5
                desired_y = radius * np.sin(angle) - scale * 0.3
                
            elif section < np.pi:  # Second quarter - straightaway
                progress = (section - np.pi/2) / (np.pi/2)
                desired_x = scale * (1.7 - 3.0 * progress)
                desired_y = scale * (0.9 + 0.4 * np.sin(progress * np.pi))
                
            elif section < 3 * np.pi / 2:  # Third quarter - hairpin turn
                angle = (section - np.pi) * 3 + np.pi
                radius = scale * 0.6
                desired_x = radius * np.cos(angle) - scale * 1.2
                desired_y = radius * np.sin(angle) + scale * 0.8
                
            else:  # Fourth quarter - S-curve back
                progress = (section - 3*np.pi/2) / (np.pi/2)
                desired_x = scale * (-1.8 + 2.3 * progress)
                desired_y = scale * (0.8 - 1.6 * progress + 0.5 * np.sin(progress * 2 * np.pi))
            
            # Add wobble for realism
            wobble_x = scale * 0.15 * np.sin(phase * 5)
            wobble_y = scale * 0.1 * np.cos(phase * 7)
            
            desired_x += wobble_x
            desired_y += wobble_y
            
            # Compute velocity (numerical derivative)
            dt = 0.01
            phase_next = omega * (t + dt)
            section_next = (phase_next % (2 * np.pi))
            
            if section_next < np.pi / 2:
                angle_next = section_next * 4
                radius = scale * 1.2
                x_next = radius * np.cos(angle_next) + scale * 0.5
                y_next = radius * np.sin(angle_next) - scale * 0.3
            elif section_next < np.pi:
                progress_next = (section_next - np.pi/2) / (np.pi/2)
                x_next = scale * (1.7 - 3.0 * progress_next)
                y_next = scale * (0.9 + 0.4 * np.sin(progress_next * np.pi))
            elif section_next < 3 * np.pi / 2:
                angle_next = (section_next - np.pi) * 3 + np.pi
                radius = scale * 0.6
                x_next = radius * np.cos(angle_next) - scale * 1.2
                y_next = radius * np.sin(angle_next) + scale * 0.8
            else:
                progress_next = (section_next - 3*np.pi/2) / (np.pi/2)
                x_next = scale * (-1.8 + 2.3 * progress_next)
                y_next = scale * (0.8 - 1.6 * progress_next + 0.5 * np.sin(progress_next * 2 * np.pi))
            
            wobble_x_next = scale * 0.15 * np.sin(phase_next * 5)
            wobble_y_next = scale * 0.1 * np.cos(phase_next * 7)
            x_next += wobble_x_next
            y_next += wobble_y_next
            
            desired_vx = (x_next - desired_x) / dt
            desired_vy = (y_next - desired_y) / dt
            
            # Desired heading
            desired_theta = np.arctan2(desired_vy, desired_vx)
            
            # Errors
            error_x = desired_x - current_state[0]
            error_y = desired_y - current_state[1]
            error_theta = normalize_angle(desired_theta - current_state[2])
            
            # Aggressive control for complex path
            kp_v = 3.0
            kp_w = 5.0
            
            error_dist = np.sqrt(error_x**2 + error_y**2)
            v = v_desired + kp_v * error_dist
            w = kp_w * error_theta
            
            # Wider limits for aggressive maneuvering
            v = np.clip(v, 0, 4.0)
            w = np.clip(w, -3.0, 3.0)
            
            return v, w
        
        else:
            # Default: constant velocity
            return self.params.get('velocity', 1.0), 0.0
