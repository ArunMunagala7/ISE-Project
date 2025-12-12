"""
Create a 3D animation of the SLAM simulation
Shows robot trajectory and landmarks in 3D space with time as the third dimension
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from config.params import *
from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.data_association import DataAssociation
from src.utils import generate_random_landmarks

def create_3d_slam_animation():
    """Create 3D visualization with time as Z-axis"""
    print("="*60)
    print("Creating 3D SLAM Animation")
    print("="*60)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Initialize components
    print("\n1. Initializing simulation...")
    num_landmarks = NUM_LANDMARKS
    true_landmarks = generate_random_landmarks(num_landmarks, LANDMARK_AREA_SIZE)
    
    robot = Robot(INITIAL_STATE, DT)
    controller = TrajectoryController(TRAJECTORY_TYPE, {
        'radius': CIRCLE_RADIUS,
        'velocity': LINEAR_VELOCITY,
        'scale': FIGURE8_SCALE
    })
    ekf_slam = EKF_SLAM(INITIAL_STATE, num_landmarks, INITIAL_STATE_COV)
    data_assoc = DataAssociation(ekf_slam)
    
    # Run simulation and collect data
    print("2. Running simulation...")
    num_steps = int(SIM_TIME / DT)
    
    # Storage for 3D visualization
    true_trajectory_3d = []
    est_trajectory_3d = []
    landmark_observations_3d = []  # When each landmark was first seen
    
    for step in range(num_steps):
        t = step * DT
        
        # Get control and move robot
        robot_state = robot.get_state()
        v, w = controller.get_control(robot_state, t)
        robot.move(v, w, add_noise=True)
        
        # EKF prediction
        ekf_slam.predict(v, w, DT)
        
        # Get measurements and update
        measurements, measured_landmark_ids = data_assoc.simulate_measurements(
            robot.get_state(), true_landmarks, add_noise=True
        )
        
        for landmark_id, measurement in zip(measured_landmark_ids, measurements):
            # Record first observation time for this landmark
            if not ekf_slam.landmark_initialized[landmark_id]:
                landmark_observations_3d.append({
                    'id': landmark_id,
                    'time': t,
                    'true_pos': true_landmarks[landmark_id].copy(),
                    'robot_pos': robot.get_state()[0:2].copy()
                })
            ekf_slam.update(landmark_id, measurement)
        
        # Store trajectory points with time
        true_trajectory_3d.append([
            robot.get_state()[0],
            robot.get_state()[1],
            t
        ])
        
        est_state = ekf_slam.get_robot_state()
        est_trajectory_3d.append([
            est_state[0],
            est_state[1],
            t
        ])
        
        if step % 100 == 0:
            print(f"  Progress: {100*step/num_steps:.0f}%", end='\r')
    
    print(f"  Progress: 100% - Complete!       ")
    
    # Convert to arrays
    true_traj_3d = np.array(true_trajectory_3d)
    est_traj_3d = np.array(est_trajectory_3d)
    
    print("3. Creating 3D animated visualization...")
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=(16, 10))
    
    # Main 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('X Position (m)', fontsize=10)
    ax1.set_ylabel('Y Position (m)', fontsize=10)
    ax1.set_zlabel('Time (s)', fontsize=10)
    ax1.set_title('3D SLAM: Position Over Time', fontsize=14, fontweight='bold')
    
    # Side view (XZ plane) - shows X position over time
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('X Position (m)', fontsize=10)
    ax2.set_ylabel('Time (s)', fontsize=10)
    ax2.set_title('Side View: X vs Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Top view (XY plane) - shows traditional 2D map
    ax3 = fig.add_subplot(224)
    ax3.set_xlabel('X Position (m)', fontsize=10)
    ax3.set_ylabel('Y Position (m)', fontsize=10)
    ax3.set_title('Top View: 2D Trajectory', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot landmarks as vertical lines in 3D (from t=0 to t=max)
    for i in range(num_landmarks):
        if ekf_slam.landmark_initialized[i]:
            lm = true_landmarks[i]
            ax1.plot([lm[0], lm[0]], [lm[1], lm[1]], [0, SIM_TIME], 
                    'g--', alpha=0.3, linewidth=1)
            ax1.scatter([lm[0]], [lm[1]], [0], c='green', marker='*', 
                       s=200, edgecolors='black', linewidths=1, zorder=10)
    
    # Animation data
    num_frames = 25
    frame_indices = np.linspace(0, len(true_traj_3d)-1, num_frames, dtype=int)
    
    # Initialize plot elements
    true_line_3d, = ax1.plot([], [], [], 'g-', linewidth=3, label='True Path', alpha=0.8)
    est_line_3d, = ax1.plot([], [], [], 'b--', linewidth=2, label='Estimated Path', alpha=0.8)
    robot_point_3d = ax1.scatter([], [], [], c='red', marker='o', s=200, 
                                  edgecolors='black', linewidths=2, zorder=20)
    
    # Side view elements
    true_line_xz, = ax2.plot([], [], 'g-', linewidth=2, label='True', alpha=0.8)
    est_line_xz, = ax2.plot([], [], 'b--', linewidth=2, label='Estimated', alpha=0.8)
    
    # Top view elements
    true_line_xy, = ax3.plot([], [], 'g-', linewidth=2, label='True', alpha=0.8)
    est_line_xy, = ax3.plot([], [], 'b--', linewidth=2, label='Estimated', alpha=0.8)
    ax3.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='green', 
               marker='*', s=200, edgecolors='black', linewidths=1, zorder=5)
    
    # Add legends
    ax1.legend(loc='upper right', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax3.legend(loc='upper right', fontsize=9)
    
    # Set consistent axis limits
    margin = 5
    x_min, x_max = true_traj_3d[:, 0].min() - margin, true_traj_3d[:, 0].max() + margin
    y_min, y_max = true_traj_3d[:, 1].min() - margin, true_traj_3d[:, 1].max() + margin
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(0, SIM_TIME)
    
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, SIM_TIME)
    
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    
    # Time text
    time_text = ax1.text2D(0.02, 0.95, '', transform=ax1.transAxes, 
                           fontsize=12, fontweight='bold')
    
    def init():
        """Initialize animation"""
        true_line_3d.set_data([], [])
        true_line_3d.set_3d_properties([])
        est_line_3d.set_data([], [])
        est_line_3d.set_3d_properties([])
        robot_point_3d._offsets3d = ([], [], [])
        
        true_line_xz.set_data([], [])
        est_line_xz.set_data([], [])
        
        true_line_xy.set_data([], [])
        est_line_xy.set_data([], [])
        
        time_text.set_text('')
        
        return (true_line_3d, est_line_3d, robot_point_3d, 
                true_line_xz, est_line_xz, true_line_xy, est_line_xy, time_text)
    
    def animate(frame):
        """Update animation frame"""
        idx = frame_indices[frame]
        
        # Get data up to current frame
        true_data = true_traj_3d[:idx+1]
        est_data = est_traj_3d[:idx+1]
        
        if len(true_data) > 0:
            # Update 3D plot
            true_line_3d.set_data(true_data[:, 0], true_data[:, 1])
            true_line_3d.set_3d_properties(true_data[:, 2])
            
            est_line_3d.set_data(est_data[:, 0], est_data[:, 1])
            est_line_3d.set_3d_properties(est_data[:, 2])
            
            # Current robot position
            robot_point_3d._offsets3d = ([true_data[-1, 0]], 
                                         [true_data[-1, 1]], 
                                         [true_data[-1, 2]])
            
            # Update side view (X vs Time)
            true_line_xz.set_data(true_data[:, 0], true_data[:, 2])
            est_line_xz.set_data(est_data[:, 0], est_data[:, 2])
            
            # Update top view (X vs Y)
            true_line_xy.set_data(true_data[:, 0], true_data[:, 1])
            est_line_xy.set_data(est_data[:, 0], est_data[:, 1])
            
            # Update time text
            time_text.set_text(f'Time: {true_data[-1, 2]:.1f}s')
            
            # Rotate 3D view slightly
            ax1.view_init(elev=20, azim=45 + frame * 2)
        
        return (true_line_3d, est_line_3d, robot_point_3d, 
                true_line_xz, est_line_xz, true_line_xy, est_line_xy, time_text)
    
    # Create animation
    print("  Rendering frames...")
    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames,
                        interval=200, blit=False, repeat=True)
    
    # Save as GIF
    print("4. Saving 3D animation...")
    output_path = 'outputs/videos/slam_3d_animation.gif'
    writer = PillowWriter(fps=3)
    anim.save(output_path, writer=writer)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ 3D Animation saved: {file_size:.1f} MB")
    print(f"  Location: {output_path}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("3D ANIMATION COMPLETE!")
    print("="*60)
    print("\nWhat the 3D animation shows:")
    print("  • Left: Full 3D view (X, Y, Time) - rotating camera")
    print("  • Top Right: Side view showing X position over time")
    print("  • Bottom Right: Top-down view (traditional 2D SLAM)")
    print("  • Green vertical lines: Landmark positions extended through time")
    print("  • Green path: True robot trajectory")
    print("  • Blue dashed path: Estimated trajectory")
    print("  • Red dot: Current robot position")
    print(f"\n  View the animation: {output_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import os
    os.makedirs('outputs/videos', exist_ok=True)
    create_3d_slam_animation()
