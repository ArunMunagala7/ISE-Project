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
    Simple feedback controller to follow a desired trajectory
    """
    
    def __init__(self, trajectory_type, params):
        """
        Initialize controller
        
        Args:
            trajectory_type: Type of trajectory ("circle", "figure8")
            params: Dictionary of trajectory parameters
        """
        self.trajectory_type = trajectory_type
        self.params = params
        
    def get_control(self, current_state, t):
        """
        Compute control inputs to follow desired trajectory
        
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
            # Figure-8 trajectory (Lemniscate curve)
            scale = self.params.get('scale', 6.0)
            v_desired = self.params.get('velocity', 1.2)
            
            # Parametric figure-8: x = a*sin(t), y = a*sin(t)*cos(t)
            # Period adjusted for visible loops
            omega = 0.15  # Fixed angular frequency for good figure-8 shape
            
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
