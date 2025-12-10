"""
Create a smooth video animation of SLAM simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse
import os

from config.params import *
from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.data_association import DataAssociation
from src.utils import generate_random_landmarks

print("Creating SLAM animation video...")
print("This will take a minute...")

# Set seed
np.random.seed(42)

# Setup
true_landmarks = generate_random_landmarks(12, LANDMARK_AREA_SIZE)
robot = Robot(INITIAL_STATE, DT)
controller = TrajectoryController("circle", {'radius': CIRCLE_RADIUS, 'velocity': LINEAR_VELOCITY})
ekf_slam = EKF_SLAM(INITIAL_STATE, 12, INITIAL_STATE_COV)
data_assoc = DataAssociation(ekf_slam)

# Storage for frames
frames_data = []
num_steps = int(SIM_TIME / DT)

# Run simulation and store data
print("Running simulation...")
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
    
    # Store frame data every 5 steps
    if step % 5 == 0:
        frames_data.append({
            'robot_true': robot.get_state().copy(),
            'robot_est': ekf_slam.get_robot_state().copy(),
            'landmarks_est': ekf_slam.get_all_landmarks().copy(),
            'landmark_init': ekf_slam.landmark_initialized.copy(),
            'true_traj': robot.get_trajectory().copy(),
            'est_traj': np.array([mu[0:2] for mu in ekf_slam.mu_history]),
            'time': t
        })
    
    if step % 50 == 0:
        print(f"  Progress: {100*step//num_steps}%")

print(f"✓ Simulation complete. Creating video from {len(frames_data)} frames...")

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

def init():
    return []

def animate(frame_idx):
    data = frames_data[frame_idx]
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # Plot map
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-12, 12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'SLAM (t = {data["time"]:.1f}s)', fontsize=14, fontweight='bold')
    
    # True landmarks
    ax1.scatter(true_landmarks[:, 0], true_landmarks[:, 1],
               c='green', marker='*', s=200, label='True Landmarks',
               edgecolors='black', linewidths=1, zorder=5)
    
    # Estimated landmarks
    for i in range(12):
        if data['landmark_init'][i]:
            lm = data['landmarks_est'][i]
            ax1.plot(lm[0], lm[1], 'b^', markersize=10, alpha=0.7)
    
    # Trajectories
    if len(data['true_traj']) > 1:
        ax1.plot(data['true_traj'][:, 0], data['true_traj'][:, 1],
                'g-', linewidth=2, label='True Trajectory', alpha=0.6)
    
    if len(data['est_traj']) > 1:
        ax1.plot(data['est_traj'][:, 0], data['est_traj'][:, 1],
                'b--', linewidth=2, label='Estimated Trajectory', alpha=0.6)
    
    # Current robots
    rt = data['robot_true']
    re = data['robot_est']
    
    # True robot
    circle_true = Circle((rt[0], rt[1]), 0.3, color='green', alpha=0.6, label='True Robot')
    ax1.add_patch(circle_true)
    dx = 0.5 * np.cos(rt[2])
    dy = 0.5 * np.sin(rt[2])
    ax1.arrow(rt[0], rt[1], dx, dy, head_width=0.2, head_length=0.15,
             fc='green', ec='green', linewidth=2)
    
    # Estimated robot
    circle_est = Circle((re[0], re[1]), 0.3, color='blue', alpha=0.6, label='Est Robot')
    ax1.add_patch(circle_est)
    dx = 0.5 * np.cos(re[2])
    dy = 0.5 * np.sin(re[2])
    ax1.arrow(re[0], re[1], dx, dy, head_width=0.2, head_length=0.15,
             fc='blue', ec='blue', linewidth=2)
    
    ax1.legend(loc='upper right', fontsize=9)
    
    # Error plot (simple version)
    ax2.text(0.5, 0.5, f'Frame {frame_idx}/{len(frames_data)}\\nTime: {data["time"]:.1f}s',
            ha='center', va='center', fontsize=20, transform=ax2.transAxes)
    ax2.set_title('Animation Progress', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    return []

print("Creating animation object...")
anim = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(frames_data), interval=100,
                              blit=True, repeat=True)

# Save
os.makedirs('outputs/videos', exist_ok=True)
output_file = 'outputs/videos/slam_animation.mp4'

print(f"Saving to {output_file}...")
print("(This may take 1-2 minutes...)")

Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='SLAM'), bitrate=1800)
anim.save(output_file, writer=writer)

print(f"✓ Video saved to: {output_file}")
print(f"  You can open it with: open {output_file}")
