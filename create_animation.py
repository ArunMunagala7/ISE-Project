"""
Create animated GIF of SLAM simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import os

from config.params import *
from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.data_association import DataAssociation
from src.utils import generate_random_landmarks
from src.visualization import SLAMVisualizer

def create_slam_gif():
    """Create an animated GIF of SLAM convergence"""
    
    print("="*60)
    print("Creating SLAM Animation GIF")
    print("="*60)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Setup
    print("\n1. Initializing simulation...")
    num_landmarks = NUM_LANDMARKS  # Use from config (now 25)
    true_landmarks = generate_random_landmarks(num_landmarks, LANDMARK_AREA_SIZE)
    robot = Robot(INITIAL_STATE, DT)
    controller = TrajectoryController(TRAJECTORY_TYPE, {
        'radius': CIRCLE_RADIUS,
        'velocity': LINEAR_VELOCITY,
        'scale': FIGURE8_SCALE
    })
    ekf_slam = EKF_SLAM(INITIAL_STATE, num_landmarks, INITIAL_STATE_COV)
    data_assoc = DataAssociation(ekf_slam)
    
    # Run simulation and store frames
    print("2. Running simulation...")
    num_steps = int(SIM_TIME / DT)
    frames_data = []
    
    for step in range(num_steps):
        t = step * DT
        robot_state = robot.get_state()
        v, w = controller.get_control(robot_state, t)
        robot.move(v, w, add_noise=True)
        ekf_slam.predict(v, w, DT)
        
        measurements, measured_landmark_ids = data_assoc.simulate_measurements(
            robot.get_state(), true_landmarks, add_noise=True
        )
        
        for landmark_id, measurement in zip(measured_landmark_ids, measurements):
            ekf_slam.update(landmark_id, measurement)
        
        # Store every 35th frame for GIF (balances detail with file size)
        if step % 35 == 0:
            # Compute errors
            robot_est = ekf_slam.get_robot_state()
            pos_error = np.linalg.norm(robot.get_state()[0:2] - robot_est[0:2])
            
            landmark_errors = []
            for i in range(num_landmarks):
                if ekf_slam.landmark_initialized[i]:
                    est = ekf_slam.get_landmark_state(i)
                    true = true_landmarks[i]
                    error = np.linalg.norm(est - true)
                    landmark_errors.append(error)
            avg_lm_error = np.mean(landmark_errors) if landmark_errors else 0
            
            frames_data.append({
                'robot_true': robot.get_state().copy(),
                'robot_est': robot_est.copy(),
                'robot_cov': ekf_slam.get_robot_covariance().copy(),
                'landmarks_est': ekf_slam.get_all_landmarks().copy(),
                'landmark_init': ekf_slam.landmark_initialized.copy(),
                'true_traj': robot.get_trajectory().copy(),
                'est_traj': np.array([mu[0:2] for mu in ekf_slam.mu_history]),
                'time': t,
                'pos_error': pos_error,
                'lm_error': avg_lm_error,
                'num_init': np.sum(ekf_slam.landmark_initialized)
            })
        
        if step % 50 == 0:
            print(f"   Progress: {100*step//num_steps}%")
    
    print(f"   ✓ Simulation complete. Collected {len(frames_data)} frames")
    
    # Create animated figure
    print("3. Creating animated GIF...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    def plot_covariance_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
        """Plot covariance ellipse"""
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle = np.degrees(angle)
        width, height = 2 * n_std * np.sqrt(np.abs(eigenvalues))
        ellipse = Ellipse(mean, width, height, angle=angle, **kwargs)
        ax.add_patch(ellipse)
    
    # Store error history for right plot
    all_times = [f['time'] for f in frames_data]
    all_pos_errors = [f['pos_error'] for f in frames_data]
    all_lm_errors = [f['lm_error'] for f in frames_data]
    
    def animate(frame_idx):
        data = frames_data[frame_idx]
        
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # --- LEFT PLOT: MAP VIEW ---
        ax1.set_xlim(-18, 18)
        ax1.set_ylim(-18, 18)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        ax1.set_title(f'SLAM - {TRAJECTORY_TYPE.title()} Path (t={data["time"]:.1f}s, LM:{data["num_init"]}/{num_landmarks})',
                     fontsize=14, fontweight='bold')
        
        # True landmarks
        ax1.scatter(true_landmarks[:, 0], true_landmarks[:, 1],
                   c='green', marker='*', s=250, label='True Landmarks',
                   edgecolors='black', linewidths=1.5, zorder=5)
        
        # Estimated landmarks with uncertainty
        for i in range(num_landmarks):
            if data['landmark_init'][i]:
                lm = data['landmarks_est'][i]
                lm_cov = ekf_slam.get_landmark_covariance(i)
                
                # Landmark estimate
                ax1.plot(lm[0], lm[1], 'b^', markersize=12, alpha=0.8, markeredgecolor='darkblue', markeredgewidth=1.5)
                
                # Uncertainty ellipse
                try:
                    plot_covariance_ellipse(ax1, lm, lm_cov, n_std=2.0,
                                          facecolor='blue', alpha=0.15, edgecolor='blue', linewidth=1.5)
                except:
                    pass  # Skip if covariance is singular
        
        # True trajectory
        if len(data['true_traj']) > 1:
            ax1.plot(data['true_traj'][:, 0], data['true_traj'][:, 1],
                    'g-', linewidth=2.5, label='True Trajectory', alpha=0.7)
        
        # Estimated trajectory
        if len(data['est_traj']) > 1:
            ax1.plot(data['est_traj'][:, 0], data['est_traj'][:, 1],
                    'b--', linewidth=2.5, label='Estimated Trajectory', alpha=0.7)
        
        # Current robots
        rt = data['robot_true']
        re = data['robot_est']
        
        # True robot
        circle_true = Circle((rt[0], rt[1]), 0.4, color='green', alpha=0.7,
                            edgecolor='darkgreen', linewidth=2, label='True Robot')
        ax1.add_patch(circle_true)
        dx = 0.6 * np.cos(rt[2])
        dy = 0.6 * np.sin(rt[2])
        ax1.arrow(rt[0], rt[1], dx, dy, head_width=0.25, head_length=0.2,
                 fc='darkgreen', ec='darkgreen', linewidth=2.5)
        
        # Estimated robot with uncertainty
        circle_est = Circle((re[0], re[1]), 0.4, color='blue', alpha=0.7,
                           edgecolor='darkblue', linewidth=2, label='Est Robot')
        ax1.add_patch(circle_est)
        dx = 0.6 * np.cos(re[2])
        dy = 0.6 * np.sin(re[2])
        ax1.arrow(re[0], re[1], dx, dy, head_width=0.25, head_length=0.2,
                 fc='darkblue', ec='darkblue', linewidth=2.5)
        
        # Robot uncertainty ellipse
        try:
            plot_covariance_ellipse(ax1, re[0:2], data['robot_cov'][0:2, 0:2],
                                  n_std=2.0, facecolor='cyan', alpha=0.25,
                                  edgecolor='blue', linewidth=2)
        except:
            pass
        
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # --- RIGHT PLOT: ERROR CONVERGENCE ---
        ax2.set_xlim(0, SIM_TIME)
        ax2.set_ylim(0, max(max(all_pos_errors[:frame_idx+1]), max(all_lm_errors[:frame_idx+1])) * 1.1 + 0.1)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Error (m)', fontsize=12)
        ax2.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        
        # Plot errors up to current frame
        times_so_far = all_times[:frame_idx+1]
        pos_errors_so_far = all_pos_errors[:frame_idx+1]
        lm_errors_so_far = all_lm_errors[:frame_idx+1]
        
        ax2.plot(times_so_far, pos_errors_so_far, 'b-', linewidth=2.5,
                label=f'Robot Error: {data["pos_error"]:.2f}m')
        ax2.plot(times_so_far, lm_errors_so_far, 'r-', linewidth=2.5,
                label=f'Landmark Error: {data["lm_error"]:.2f}m')
        
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
    
    # Create animation (slower interval for better viewing)
    anim = FuncAnimation(fig, animate, frames=len(frames_data),
                        interval=500, repeat=True, blit=False)  # 500ms = 0.5 sec per frame
    
    # Save as GIF
    os.makedirs('outputs/videos', exist_ok=True)
    output_file = 'outputs/videos/slam_animation.gif'
    
    print(f"4. Saving to {output_file}...")
    print("   (This will take 30-60 seconds...)")
    
    # Slower FPS for easier viewing
    writer = PillowWriter(fps=2)  # 2 frames per second = slower, easier to see
    anim.save(output_file, writer=writer)
    
    print(f"\n✓ Animation saved to: {output_file}")
    print(f"✓ File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    print(f"\nTo view the animation:")
    print(f"  open {output_file}")
    print("="*60)
    
    plt.close()

if __name__ == "__main__":
    create_slam_gif()
