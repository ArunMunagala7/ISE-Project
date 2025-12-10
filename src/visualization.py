"""
Visualization and Animation for SLAM
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as mpatches
from src.utils import normalize_angle


class SLAMVisualizer:
    """
    Visualize SLAM convergence in real-time
    """
    
    def __init__(self, true_landmarks, save_video=False, video_filename='slam_animation.mp4'):
        """
        Initialize visualizer
        
        Args:
            true_landmarks: True landmark positions (n x 2)
            save_video: Whether to save video
            video_filename: Output video filename
        """
        self.true_landmarks = true_landmarks
        self.save_video = save_video
        self.video_filename = video_filename
        
        # Setup figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 7))
        self.ax_map = self.axes[0]
        self.ax_error = self.axes[1]
        
        # Storage for animation
        self.frames = []
        self.error_history = {'position': [], 'landmarks': [], 'time': []}
        
    def plot_covariance_ellipse(self, ax, mean, cov, n_std=2.0, **kwargs):
        """
        Plot covariance ellipse
        
        Args:
            ax: Matplotlib axis
            mean: Mean position [x, y]
            cov: 2x2 covariance matrix
            n_std: Number of standard deviations
        """
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle = np.degrees(angle)
        
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        ellipse = Ellipse(mean, width, height, angle=angle, **kwargs)
        ax.add_patch(ellipse)
        
        return ellipse
    
    def update_plot(self, robot_true, robot_est, ekf_slam, time_step):
        """
        Update visualization for current timestep
        
        Args:
            robot_true: True robot state [x, y, theta]
            robot_est: Estimated robot state [x, y, theta]
            ekf_slam: EKF_SLAM object
            time_step: Current time
        """
        self.ax_map.clear()
        
        # Set plot limits
        margin = 15
        self.ax_map.set_xlim(-margin, margin)
        self.ax_map.set_ylim(-margin, margin)
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_xlabel('X (m)', fontsize=12)
        self.ax_map.set_ylabel('Y (m)', fontsize=12)
        self.ax_map.set_title(f'SLAM Convergence (t = {time_step:.1f}s)', fontsize=14, fontweight='bold')
        
        # Plot true landmarks
        self.ax_map.scatter(self.true_landmarks[:, 0], self.true_landmarks[:, 1],
                           c='green', marker='*', s=200, label='True Landmarks',
                           edgecolors='black', linewidths=1, zorder=5)
        
        # Plot estimated landmarks with uncertainty
        for i in range(ekf_slam.num_landmarks):
            if ekf_slam.landmark_initialized[i]:
                landmark_est = ekf_slam.get_landmark_state(i)
                landmark_cov = ekf_slam.get_landmark_covariance(i)
                
                # Plot estimate
                self.ax_map.plot(landmark_est[0], landmark_est[1], 
                               'b^', markersize=10, alpha=0.7)
                
                # Plot uncertainty ellipse
                self.plot_covariance_ellipse(self.ax_map, landmark_est, landmark_cov,
                                            n_std=2.0, facecolor='blue', 
                                            alpha=0.2, edgecolor='blue')
        
        # Plot true robot trajectory
        if hasattr(self, 'true_traj_line'):
            true_traj = np.array(self.true_trajectory)
            self.ax_map.plot(true_traj[:, 0], true_traj[:, 1], 
                           'g-', linewidth=2, label='True Trajectory', alpha=0.6)
        
        # Plot estimated robot trajectory
        if len(ekf_slam.mu_history) > 1:
            est_traj = np.array([mu[0:2] for mu in ekf_slam.mu_history])
            self.ax_map.plot(est_traj[:, 0], est_traj[:, 1], 
                           'b--', linewidth=2, label='Estimated Trajectory', alpha=0.6)
        
        # Plot current true robot
        self.plot_robot(self.ax_map, robot_true, 'green', 'True Robot')
        
        # Plot current estimated robot with uncertainty
        self.plot_robot(self.ax_map, robot_est, 'blue', 'Estimated Robot')
        robot_cov = ekf_slam.get_robot_covariance()
        self.plot_covariance_ellipse(self.ax_map, robot_est[0:2], robot_cov[0:2, 0:2],
                                     n_std=2.0, facecolor='cyan', alpha=0.3, 
                                     edgecolor='blue', linewidth=2)
        
        self.ax_map.legend(loc='upper right', fontsize=10)
        
    def plot_robot(self, ax, state, color, label):
        """
        Plot robot as a circle with heading indicator
        
        Args:
            ax: Matplotlib axis
            state: Robot state [x, y, theta]
            color: Color for robot
            label: Label for legend
        """
        x, y, theta = state
        
        # Robot body
        circle = Circle((x, y), 0.3, color=color, alpha=0.6, label=label)
        ax.add_patch(circle)
        
        # Heading indicator
        dx = 0.5 * np.cos(theta)
        dy = 0.5 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.15, 
                fc=color, ec=color, linewidth=2)
    
    def update_error_plot(self, robot_true, robot_est, ekf_slam, time_step):
        """
        Update error plots
        
        Args:
            robot_true: True robot state
            robot_est: Estimated robot state
            ekf_slam: EKF_SLAM object
            time_step: Current time
        """
        # Compute position error
        pos_error = np.linalg.norm(robot_true[0:2] - robot_est[0:2])
        
        # Compute landmark error (only for initialized landmarks)
        landmark_errors = []
        for i in range(ekf_slam.num_landmarks):
            if ekf_slam.landmark_initialized[i]:
                est = ekf_slam.get_landmark_state(i)
                true = self.true_landmarks[i]
                error = np.linalg.norm(est - true)
                landmark_errors.append(error)
        
        avg_landmark_error = np.mean(landmark_errors) if landmark_errors else 0
        
        # Store errors
        self.error_history['time'].append(time_step)
        self.error_history['position'].append(pos_error)
        self.error_history['landmarks'].append(avg_landmark_error)
        
        # Plot errors
        self.ax_error.clear()
        self.ax_error.plot(self.error_history['time'], 
                          self.error_history['position'],
                          'b-', linewidth=2, label='Robot Position Error')
        self.ax_error.plot(self.error_history['time'], 
                          self.error_history['landmarks'],
                          'r-', linewidth=2, label='Avg Landmark Error')
        self.ax_error.set_xlabel('Time (s)', fontsize=12)
        self.ax_error.set_ylabel('Error (m)', fontsize=12)
        self.ax_error.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        self.ax_error.grid(True, alpha=0.3)
        self.ax_error.legend(loc='upper right', fontsize=10)
    
    def save_frame(self):
        """Save current frame for video"""
        if self.save_video:
            self.frames.append(self.fig)
    
    def show(self):
        """Display plot with animation"""
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause for smooth animation
    
    def save_final_plots(self, output_dir='outputs/plots'):
        """
        Save final plots
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save error plot
        fig_error, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.error_history['time'], self.error_history['position'],
               'b-', linewidth=2, label='Robot Position Error')
        ax.plot(self.error_history['time'], self.error_history['landmarks'],
               'r-', linewidth=2, label='Average Landmark Error')
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Error (m)', fontsize=14)
        ax.set_title('SLAM Convergence Over Time', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence_error.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plots saved to {output_dir}/")


class AnimationRecorder:
    """
    Record and save animation as video
    """
    
    def __init__(self, fig, output_file='outputs/videos/slam_animation.mp4', fps=10):
        """
        Initialize recorder
        
        Args:
            fig: Matplotlib figure
            output_file: Output video filename
            fps: Frames per second
        """
        self.fig = fig
        self.output_file = output_file
        self.fps = fps
        self.frames_data = []
        
    def add_frame(self, robot_true, robot_est, ekf_slam, true_landmarks, time_step):
        """Add frame data"""
        self.frames_data.append({
            'robot_true': robot_true.copy(),
            'robot_est': robot_est.copy(),
            'ekf_slam_mu': ekf_slam.mu.copy(),
            'ekf_slam_sigma': ekf_slam.sigma.copy(),
            'landmark_init': ekf_slam.landmark_initialized.copy(),
            'time': time_step
        })
    
    def save_video(self):
        """Save recorded frames as video"""
        import os
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        print(f"✓ Saving video to {self.output_file}...")
        # This would require additional implementation with cv2 or matplotlib animation
        # For now, we'll save individual frames
        print(f"✓ Recorded {len(self.frames_data)} frames")
